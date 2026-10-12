"""Tag schemes and BIO <-> span conversion."""

from __future__ import annotations

from dataclasses import dataclass

# Tag ids used by the VLSP2016 dump in data/raw_data.csv. 7/8 are the fourth entity
# type (MISC; called "NAT" in the v1 EDA).
VLSP_TAGS: tuple[str, ...] = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC")
# v1 mapped MISC to O, so the published checkpoint has 7 tags.
LEGACY_TAGS: tuple[str, ...] = VLSP_TAGS[:7]


@dataclass(frozen=True)
class TagSpan:
    """Entity over word indices ``[start, end)``."""

    label: str
    start: int
    end: int


def bio_to_spans(tags: list[str]) -> list[TagSpan]:
    """Decode BIO tags leniently (conlleval style): an ``I-X`` that does not continue an
    ``X`` entity starts a new one instead of being dropped."""

    spans: list[TagSpan] = []
    label, start = None, 0
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, kind = tag.partition("-")
        continues = prefix == "I" and kind == label
        if label is not None and not continues:
            spans.append(TagSpan(label, start, i))
            label = None
        if prefix in ("B", "I") and not continues:
            label, start = kind, i
    return spans


def spans_to_bio(spans: list[TagSpan], length: int) -> list[str]:
    tags = ["O"] * length
    for s in spans:
        tags[s.start] = f"B-{s.label}"
        for i in range(s.start + 1, s.end):
            tags[i] = f"I-{s.label}"
    return tags


def entity_types(tags: tuple[str, ...]) -> list[str]:
    return sorted({t[2:] for t in tags if t != "O"})
