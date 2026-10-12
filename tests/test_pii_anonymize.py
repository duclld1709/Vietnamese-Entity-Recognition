import pytest

from vietner.anonymize import anonymize, deanonymize
from vietner.pii import find_pii, resolve_overlaps
from vietner.schemas import Entity


def labels_in(text):
    return [(e.label, e.text) for e in resolve_overlaps(find_pii(text))]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Gọi 0912345678 nhé", ("PHONE", "0912345678")),
        ("SĐT: 0912.345.678.", ("PHONE", "0912.345.678")),
        ("hotline +84 987 654 321", ("PHONE", "+84 987 654 321")),
        ("bàn 024 3826 1234", ("PHONE", "024 3826 1234")),
        ("mail: a.b+c@fpt.com.vn", ("EMAIL", "a.b+c@fpt.com.vn")),
        ("CCCD 079090001234", ("ID_NUMBER", "079090001234")),
        ("số chứng minh nhân dân 123456789", ("ID_NUMBER", "123456789")),
        ("MST: 0312345678-001", ("TAX_CODE", "0312345678-001")),
        ("STK 0071 0012 34567 Vietcombank", ("BANK_ACCOUNT", "0071 0012 34567")),
        ("thẻ 4111 1111 1111 1111", ("CREDIT_CARD", "4111 1111 1111 1111")),
        ("biển số 51G-123.45", ("LICENSE_PLATE", "51G-123.45")),
        ("sinh ngày 12/05/1990", ("DATE", "12/05/1990")),
        ("ngày 2 tháng 9 năm 1945", ("DATE", "ngày 2 tháng 9 năm 1945")),
        ("xem https://example.com/a?b=1.", ("URL", "https://example.com/a?b=1")),
    ],
)
def test_recognizers_find(text, expected):
    assert expected in labels_in(text)


@pytest.mark.parametrize(
    "text",
    [
        "Dân số 0123456789012 người",  # 13 digits: not a phone, fails Luhn
        "mã đơn 123456789",  # 9 digits without ID context
        "giá 1312345678 đồng",  # 10 digits, not a phone, no tax/bank context
        "CCCD 999090001234",  # 999 is not a province code
        "số 0612345678",  # 06x is not a mobile prefix
        "thẻ 4111 1111 1111 1112",  # Luhn fails
        "ngày 45/13/1990",  # invalid date
        "01234567891",  # 11 digits starting 01: not a phone
    ],
)
def test_recognizers_reject(text):
    assert labels_in(text) == []


def test_context_raises_score():
    no_ctx = find_pii("0912345678")[0]
    ctx = find_pii("SĐT 0912345678")[0]
    assert ctx.score > no_ctx.score


def test_overlap_prefers_rules_and_longer_spans():
    model = Entity(label="LOC", text="gmail.com", start=6, end=15, score=0.8, source="model")
    rule = find_pii("email a@gmail.com")
    chosen = resolve_overlaps([model, *rule])
    assert [e.label for e in chosen] == ["EMAIL"]


def ents(text, *pairs):
    out, cursor = [], 0
    for label, value in pairs:
        start = text.index(value, cursor)
        out.append(Entity(label=label, text=value, start=start, end=start + len(value), source="model"))
        cursor = start + len(value)
    return out


TEXT = "Anh Nam gặp chị Lan ở Hà Nội. Sau đó anh NAM gọi 0912345678."


def test_tag_strategy_is_consistent_and_reversible():
    entities = ents(TEXT, ("PER", "Nam"), ("PER", "Lan"), ("LOC", "Hà Nội"), ("PER", "NAM"), ("PHONE", "0912345678"))
    result = anonymize(TEXT, entities, "tag")
    assert result.text == "Anh [PER_1] gặp chị [PER_2] ở [LOC_1]. Sau đó anh [PER_1] gọi [PHONE_1]."
    assert result.mapping == {"[PER_1]": "Nam", "[PER_2]": "Lan", "[LOC_1]": "Hà Nội", "[PHONE_1]": "0912345678"}
    llm_answer = "[PER_1] nên gọi lại cho [PER_2] tại [LOC_1]."
    assert deanonymize(llm_answer, result.mapping) == "Nam nên gọi lại cho Lan tại Hà Nội."


def test_placeholders_do_not_collide_past_nine():
    text = " ".join(f"n{i}" for i in range(12))
    result = anonymize(text, ents(text, *[("PER", f"n{i}") for i in range(12)]), "tag")
    assert deanonymize(result.text, result.mapping) == text


def test_mask_strategy():
    text = "Nguyễn Văn Nam, 0912345678, nam.nguyen@fpt.vn, CCCD 079090001234"
    entities = ents(
        text, ("PER", "Nguyễn Văn Nam"), ("PHONE", "0912345678"), ("EMAIL", "nam.nguyen@fpt.vn"),
        ("ID_NUMBER", "079090001234"),
    )  # fmt: skip
    assert anonymize(text, entities, "mask").text == "N***** V** N**, 09******78, n*********@fpt.vn, CCCD *********234"


def test_redact_and_label_filter():
    entities = ents(TEXT, ("PER", "Nam"), ("LOC", "Hà Nội"))
    result = anonymize(TEXT, entities, "redact", labels=["PER"])
    assert result.text.startswith("Anh [PER] gặp chị Lan ở Hà Nội.") and result.mapping == {}


def test_existing_placeholders_warn():
    assert anonymize("đã có [PER_1] rồi", [], "tag").warnings
    with pytest.raises(ValueError):
        anonymize("x", [], "shuffle")
