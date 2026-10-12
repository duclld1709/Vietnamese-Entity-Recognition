import random
import unicodedata

import pytest
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision

from vietner.labels import TagSpan, bio_to_spans, spans_to_bio
from vietner.metrics import entity_report, error_breakdown, token_report
from vietner.text import Word, align, normalize, segment, split_sentences


def test_normalize_composes_vietnamese():
    nfd = unicodedata.normalize("NFD", "Hà Nội")
    assert len(nfd) > len("Hà Nội") and normalize(nfd) == "Hà Nội"


def test_split_sentences_offsets():
    text = "  Câu một. Câu hai!\n\nCâu ba…  Hết "
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["Câu một.", "Câu hai!", "Câu ba…", "Hết"]


@pytest.mark.parametrize(
    "text",
    [
        "Nguyễn Văn A đang làm việc tại Hà Nội.",
        "Bà Lê Thị Hoà sinh ngày 12/05/1990 tại TP.HCM, gọi 0912.345.678!",
        'Email: abc@gmail.com (công ty "ABC") – 100% ✔',
        "snake_case_name và   nhiều    khoảng trắng",
    ],
)
def test_pyvi_segmentation_aligns_to_original_offsets(text):
    words = segment(text, offset=5)
    assert words
    for w in words:
        surface = text[w.start - 5 : w.end - 5]
        assert surface == w.text or surface.split() == w.text.split("_")


def test_align_rejects_changed_text_and_segment_falls_back():
    assert align("abc def", "abc xyz") is None
    words = segment("abc def", segmenter=lambda s: "totally different")
    assert words == [Word("abc", 0, 3), Word("def", 4, 7)]


def test_bio_to_spans_is_lenient_like_conlleval():
    tags = ["B-PER", "I-PER", "O", "I-LOC", "B-LOC", "I-LOC", "I-ORG"]
    assert bio_to_spans(tags) == [
        TagSpan("PER", 0, 2),
        TagSpan("LOC", 3, 4),
        TagSpan("LOC", 4, 6),
        TagSpan("ORG", 6, 7),
    ]
    assert bio_to_spans(spans_to_bio([TagSpan("PER", 1, 3)], 4)) == [TagSpan("PER", 1, 3)]


def test_entity_report_matches_seqeval():
    rng = random.Random(0)
    tags = ["O", "O", "O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    gold = [[rng.choice(tags) for _ in range(rng.randint(1, 15))] for _ in range(300)]
    pred = [[t if rng.random() < 0.8 else rng.choice(tags) for t in g] for g in gold]
    report = entity_report(gold, pred)
    assert report["micro"]["f1"] == pytest.approx(seqeval_f1(gold, pred))
    assert report["micro"]["precision"] == pytest.approx(seqeval_precision(gold, pred))


def test_token_metric_is_much_higher_than_entity_metric_on_o_heavy_data():
    gold = [["O"] * 18 + ["B-ORG", "I-ORG"]] * 10
    pred = [["O"] * 18 + ["B-ORG", "O"]] * 10  # every entity truncated -> all wrong
    assert entity_report(gold, pred)["micro"]["f1"] == 0.0
    assert token_report(gold, pred)["f1"] > 0.5


def test_error_breakdown():
    gold = [["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O"]]
    pred = [["B-PER", "O", "O", "B-ORG", "O", "O", "B-LOC"]]
    assert error_breakdown(gold, pred) == {
        "correct": 0,
        "wrong_boundary": 1,
        "wrong_type": 1,
        "missed": 1,
        "spurious": 1,
    }
