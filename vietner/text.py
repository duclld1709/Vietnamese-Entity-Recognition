"""Text normalisation, sentence splitting and word segmentation with character offsets.

PhoBERT expects word-segmented input ("Hà_Nội"). We segment with pyvi and align
every segmented word back to the original string, so entities can be returned as
character spans of the user's text and replaced in place by the anonymiser.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sentence boundary: end punctuation followed by whitespace, or a line break.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass(frozen=True)
class Word:
    """A segmented word (syllables joined by ``_``) and its ``[start, end)`` in the text."""

    text: str
    start: int
    end: int


def normalize(text: str) -> str:
    """NFC-normalise. Vietnamese text often arrives decomposed (NFD) from macOS or PDFs,
    which PhoBERT's vocabulary does not cover."""

    return unicodedata.normalize("NFC", text)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``[start, end)`` offsets of non-empty sentences."""

    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def _pyvi_segment(sentence: str) -> str:
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize(sentence)


def align(text: str, segmented: str, offset: int = 0) -> list[Word] | None:
    """Map tokens of ``segmented`` (pyvi output) back onto ``text``.

    pyvi only inserts spaces (around punctuation) and joins syllables with ``_``, so each
    syllable must appear in order with nothing but whitespace or a previous token
    between. Returns ``None`` if that assumption breaks.
    """

    words: list[Word] = []
    cursor = 0
    for token in segmented.split():
        # A literal underscore in the input stays as is.
        parts = [token] if text.startswith(token, cursor) else token.split("_")
        start = None
        for part in parts:
            if not part:
                continue
            idx = text.find(part, cursor)
            if idx < 0 or text[cursor:idx].strip():
                return None
            if start is None:
                start = idx
            cursor = idx + len(part)
        if start is not None:
            words.append(Word(token, offset + start, offset + cursor))
    if text[cursor:].strip():
        return None
    return words


def segment(text: str, offset: int = 0, segmenter=_pyvi_segment) -> list[Word]:
    """Word-segment one sentence; fall back to whitespace tokens if alignment fails."""

    try:
        words = align(text, segmenter(text), offset)
    except Exception:  # segmenter crash on odd input
        logger.exception("word segmentation failed")
        words = None
    if words is None:
        logger.warning("could not align segmentation, falling back to whitespace tokens: %r", text[:80])
        words = [Word(m.group(), offset + m.start(), offset + m.end()) for m in re.finditer(r"\S+", text)]
    return words
