"""High-level API: text -> entities with character offsets -> anonymised text."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import torch

from vietner.anonymize import Strategy, anonymize
from vietner.labels import bio_to_spans
from vietner.pii import find_pii, resolve_overlaps
from vietner.schemas import AnalyzeResult, AnonymizeResult, Entity
from vietner.text import Word, normalize, segment, split_sentences

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "best_epoch_16.pt"
# The CRF gives no calibrated confidence; model entities get a fixed score that ranks
# them below validated rule matches when spans overlap.
MODEL_SCORE = 0.8
MAX_CHARS = 20_000


class Tagger(Protocol):
    tags: tuple[str, ...]

    def predict(self, words_batch: list[list[str]], batch_size: int = 16) -> list[list[str]]: ...


class VietNER:
    """Named entities + rule-based PII for Vietnamese text.

    ``tagger=None`` runs in rules-only mode (e.g. when PhoBERT cannot be downloaded),
    and every result carries a warning saying so.
    """

    def __init__(
        self,
        tagger: Tagger | None,
        *,
        segmenter: Callable[[str, int], list[Word]] = segment,
        batch_size: int = 16,
        max_chars: int = MAX_CHARS,
        unavailable_reason: str | None = None,
    ):
        self.tagger = tagger
        self.segmenter = segmenter
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.unavailable_reason = unavailable_reason

    @classmethod
    def from_checkpoint(
        cls, path: str | Path = DEFAULT_CHECKPOINT, *, device: str = "auto", strict: bool = False, **kwargs
    ) -> VietNER:
        """Load the tagger; with ``strict=False`` fall back to rules-only on failure."""

        from vietner.models import NERTagger

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tagger, _ = NERTagger.load(path, device=device)
        except Exception as exc:
            if strict:
                raise
            logger.exception("NER model unavailable, running rules-only")
            return cls(None, unavailable_reason=f"{type(exc).__name__}: {exc}", **kwargs)
        return cls(tagger, **kwargs)

    @property
    def ner_available(self) -> bool:
        return self.tagger is not None

    # ------------------------------------------------------------------ analysis

    def analyze(self, texts: Iterable[str], *, ner: bool = True, rules: bool = True) -> list[AnalyzeResult]:
        texts = [normalize(t) for t in texts]
        for t in texts:
            if len(t) > self.max_chars:
                raise ValueError(f"text of {len(t)} characters exceeds the limit of {self.max_chars}")

        model_entities: list[list[Entity]] = [[] for _ in texts]
        warnings: list[str] = []
        if ner and self.tagger is not None:
            sentences: list[tuple[int, list[Word]]] = []
            for ti, text in enumerate(texts):
                for start, end in split_sentences(text):
                    words = self.segmenter(text[start:end], start)
                    if words:
                        sentences.append((ti, words))
            tags = self.tagger.predict([[w.text for w in ws] for _, ws in sentences], batch_size=self.batch_size)
            for (ti, words), seq in zip(sentences, tags, strict=True):
                for span in bio_to_spans(seq):
                    s, e = words[span.start].start, words[span.end - 1].end
                    model_entities[ti].append(
                        Entity(label=span.label, text=texts[ti][s:e], start=s, end=e, score=MODEL_SCORE, source="model")
                    )
        elif ner:
            warnings.append(f"NER model unavailable; only rule-based entities returned ({self.unavailable_reason}).")

        results = []
        for ti, text in enumerate(texts):
            found = model_entities[ti] + (find_pii(text) if rules else [])
            results.append(AnalyzeResult(text=text, entities=resolve_overlaps(found), warnings=warnings))
        return results

    def anonymize(self, text: str, strategy: Strategy = "tag", labels: Iterable[str] | None = None) -> AnonymizeResult:
        analysis = self.analyze([text])[0]
        result = anonymize(analysis.text, analysis.entities, strategy, labels)
        result.warnings = analysis.warnings + result.warnings
        return result
