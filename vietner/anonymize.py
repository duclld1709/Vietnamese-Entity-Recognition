"""Replace detected entities in text.

Strategies:

* ``tag``    - ``[PER_1]``-style placeholders, consistent per entity, reversible with
  :func:`deanonymize`. Use it to send documents to an external LLM and restore the
  names in its answer.
* ``mask``   - partial masking that keeps the text readable (``09*******78``,
  ``n***@gmail.com``, ``N**** V** A``).
* ``redact`` - irreversible ``[PER]`` labels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from vietner.pii import resolve_overlaps
from vietner.schemas import AnonymizeResult, Entity, Replacement

Strategy = Literal["tag", "mask", "redact"]
STRATEGIES: tuple[str, ...] = ("tag", "mask", "redact")

# Labels anonymised by default. MISC (nationalities, events, ...) is rarely personal data.
DEFAULT_LABELS: frozenset[str] = frozenset(
    {"PER", "ORG", "LOC", "EMAIL", "PHONE", "ID_NUMBER", "TAX_CODE", "BANK_ACCOUNT", "CREDIT_CARD", "LICENSE_PLATE",
     "DATE", "URL"}
)  # fmt: skip
# Digit-based labels: how many trailing digits stay visible when masking (letters/separators are kept).
_KEEP_LAST_DIGITS = {
    "PHONE": 2, "ID_NUMBER": 3, "BANK_ACCOUNT": 3, "CREDIT_CARD": 4, "TAX_CODE": 0, "DATE": 4, "LICENSE_PLATE": 0,
}  # fmt: skip
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


def _canonical(s: str) -> str:
    return " ".join(s.casefold().split())


def _mask(entity: Entity) -> str:
    value = entity.text
    if entity.label == "EMAIL":
        local, _, domain = value.partition("@")
        return f"{local[:1]}{'*' * max(1, len(local) - 1)}@{domain}"
    if entity.label in _KEEP_LAST_DIGITS:
        keep = _KEEP_LAST_DIGITS[entity.label]
        digits_seen, total = 0, sum(c.isdigit() for c in value)
        out = []
        for c in value:
            if c.isdigit():
                digits_seen += 1
                # keep the first two digits of phones (carrier prefix) and the last `keep`
                visible = digits_seen > total - keep or (entity.label == "PHONE" and digits_seen <= 2)
                out.append(c if visible else "*")
            else:
                out.append(c)
        return "".join(out)
    # names, places, ...: keep the first letter of each word
    return " ".join(w[:1] + "*" * (len(w) - 1) for w in value.split())


def anonymize(
    text: str,
    entities: Iterable[Entity],
    strategy: Strategy = "tag",
    labels: Iterable[str] | None = None,
) -> AnonymizeResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    allowed = set(labels) if labels is not None else DEFAULT_LABELS
    selected = resolve_overlaps(e for e in entities if e.label in allowed)

    warnings = []
    if strategy == "tag" and _PLACEHOLDER.search(text):
        warnings.append("Input already contains [LABEL_N] placeholders; de-anonymisation may be ambiguous.")

    counters: dict[str, int] = {}
    tags: dict[tuple[str, str], str] = {}
    mapping: dict[str, str] = {}
    pieces: list[str] = []
    replacements: list[Replacement] = []
    cursor = 0
    for e in selected:
        if strategy == "tag":
            key = (e.label, _canonical(e.text))
            if key not in tags:
                counters[e.label] = counters.get(e.label, 0) + 1
                tags[key] = f"[{e.label}_{counters[e.label]}]"
                mapping[tags[key]] = e.text
            new = tags[key]
        elif strategy == "mask":
            new = _mask(e)
        else:
            new = f"[{e.label}]"
        pieces += [text[cursor : e.start], new]
        replacements.append(Replacement(label=e.label, original=e.text, replacement=new, start=e.start, end=e.end))
        cursor = e.end
    pieces.append(text[cursor:])
    return AnonymizeResult(text="".join(pieces), replacements=replacements, mapping=mapping, warnings=warnings)


def deanonymize(text: str, mapping: dict[str, str]) -> str:
    """Put the originals back into text produced from a ``tag``-anonymised document."""

    if not mapping:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group()], text)
