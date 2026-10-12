import pytest
from fastapi.testclient import TestClient

from tests.conftest import V1_CHECKPOINT, StubTagger
from vietner.api import create_app
from vietner.models import NERTagger
from vietner.pipeline import VietNER
from vietner.render import render_html
from vietner.schemas import Entity

TEXT = (
    "Ông Nguyễn Văn Nam, giám đốc FPT, sống ở Hà Nội. Liên hệ SĐT 0912.345.678.\n"
    "Ông Nguyễn Văn Nam có tài khoản 0071001234567 tại Vietcombank."
)


@pytest.fixture
def ner():
    return VietNER(StubTagger())


def test_entities_have_exact_character_offsets(ner):
    result = ner.analyze([TEXT])[0]
    got = [(e.label, e.text, e.source) for e in result.entities]
    assert got == [
        ("PER", "Nguyễn Văn Nam", "model"),
        ("ORG", "FPT", "model"),
        ("LOC", "Hà Nội", "model"),
        ("PHONE", "0912.345.678", "rule"),
        ("PER", "Nguyễn Văn Nam", "model"),
        ("BANK_ACCOUNT", "0071001234567", "rule"),
        ("ORG", "Vietcombank", "model"),
    ]
    assert all(result.text[e.start : e.end] == e.text for e in result.entities)


def test_batches_sentences_from_all_texts_in_one_call(ner):
    ner.analyze(["Câu một. Câu hai.", "Câu ba."])
    assert len(ner.tagger.calls) == 1 and len(ner.tagger.calls[0]) == 3


def test_anonymize_end_to_end(ner):
    result = ner.anonymize(TEXT)
    assert "Nguyễn" not in result.text and "0912" not in result.text
    assert result.text.count("[PER_1]") == 2


def test_rules_only_mode_warns():
    result = VietNER(None, unavailable_reason="offline").analyze(["Gọi 0912345678"])[0]
    assert [e.label for e in result.entities] == ["PHONE"]
    assert "offline" in result.warnings[0]


def test_from_checkpoint_degrades_gracefully(tmp_path):
    ner = VietNER.from_checkpoint(tmp_path / "missing.pt")
    assert not ner.ner_available and "FileNotFoundError" in ner.unavailable_reason
    with pytest.raises(FileNotFoundError):
        VietNER.from_checkpoint(tmp_path / "missing.pt", strict=True)


def test_rejects_huge_text(ner):
    with pytest.raises(ValueError):
        ner.analyze(["a" * 20_001])


def test_real_checkpoint_with_pyvi(encoder768):
    tagger, _ = NERTagger.load(V1_CHECKPOINT, encoder=encoder768)
    result = VietNER(tagger).analyze([TEXT + " " + "Câu rất dài. " * 50])[0]
    assert all(result.text[e.start : e.end] == e.text for e in result.entities)


def test_render_escapes_html():
    text = "<script>alert(1)</script> Nam"
    html = render_html(text, [Entity(label="PER", text="Nam", start=26, end=29, source="model")])
    assert "<script>" not in html and "&lt;script&gt;" in html and "PER" in html


@pytest.fixture
def client(ner):
    with TestClient(create_app(lambda: ner)) as c:
        yield c


def test_api_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_api_entities_and_validation(client):
    res = client.post("/v1/entities", json={"texts": [TEXT, "Gọi 0912345678"]})
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 2 and body[1]["entities"][0]["label"] == "PHONE"
    assert client.post("/v1/entities", json={"texts": []}).status_code == 422
    assert client.post("/v1/entities", json={"texts": ["a" * 20_001]}).status_code == 413


def test_api_anonymize_roundtrip(client):
    res = client.post("/v1/anonymize", json={"text": TEXT, "strategy": "tag"}).json()
    restored = client.post("/v1/deanonymize", json={"text": res["text"], "mapping": res["mapping"]}).json()
    assert restored["text"] == TEXT
    assert client.post("/v1/anonymize", json={"text": TEXT, "strategy": "nope"}).status_code == 422


def test_api_degraded_health():
    with TestClient(create_app(lambda: VietNER(None, unavailable_reason="offline"))) as c:
        assert c.get("/health").json() == {
            "status": "degraded",
            "ner_model": False,
            "detail": "offline",
            "strategies": ["tag", "mask", "redact"],
        }
