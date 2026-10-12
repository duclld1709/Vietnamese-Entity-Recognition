"""Entity-level evaluation.

v1 reported token-level macro F1 *including the O tag* (0.91), which rewards the
~94% of tokens that are O and does not check entity boundaries. The standard for
NER (CoNLL/VLSP) is entity-level P/R/F1 where an entity counts only if both its
span and its type are exactly right. That is what :func:`entity_report` computes.
"""

from __future__ import annotations

from collections import Counter

from sklearn.metrics import precision_recall_fscore_support

from vietner.labels import TagSpan, bio_to_spans


def entity_report(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Micro/macro and per-type entity-level precision, recall and F1 (exact match)."""

    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred, strict=True):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        for s in gs & ps:
            tp[s.label] += 1
        for s in ps - gs:
            fp[s.label] += 1
        for s in gs - ps:
            fn[s.label] += 1

    def prf(t: int, f_p: int, f_n: int) -> dict:
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": t + f_n}

    labels = sorted(set(tp) | set(fp) | set(fn))
    per_type = {k: prf(tp[k], fp[k], fn[k]) for k in labels}
    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro = {
        m: sum(v[m] for v in per_type.values()) / len(per_type) if per_type else 0.0
        for m in ("precision", "recall", "f1")
    }
    return {"micro": micro, "macro": macro, "per_type": per_type}


def token_report(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """v1's metric (token-level macro over all tags incl. O), kept for comparison."""

    g = [t for s in gold for t in s]
    p = [t for s in pred for t in s]
    precision, recall, f1, _ = precision_recall_fscore_support(g, p, average="macro", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def error_breakdown(gold: list[list[str]], pred: list[list[str]]) -> dict[str, int]:
    """Classify every mistake (useful for error analysis):

    * ``wrong_type``     - exact span, wrong label
    * ``wrong_boundary`` - overlapping span, same label
    * ``wrong_both``     - overlapping span, different label
    * ``missed``         - gold entity with no overlapping prediction
    * ``spurious``       - prediction with no overlapping gold entity
    """

    counts = Counter()
    for g, p in zip(gold, pred, strict=True):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        counts["correct"] += len(gs & ps)
        for s in gs - ps:
            overlaps = [q for q in ps if _overlap(s, q)]
            if not overlaps:
                counts["missed"] += 1
            elif any(q.start == s.start and q.end == s.end for q in overlaps):
                counts["wrong_type"] += 1
            elif any(q.label == s.label for q in overlaps):
                counts["wrong_boundary"] += 1
            else:
                counts["wrong_both"] += 1
        counts["spurious"] += sum(1 for q in ps - gs if not any(_overlap(q, s) for s in gs))
    return dict(counts)


def _overlap(a: TagSpan, b: TagSpan) -> bool:
    return a.start < b.end and b.start < a.end
