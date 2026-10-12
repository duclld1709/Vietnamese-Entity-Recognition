"""Evaluate a checkpoint on the held-out test set with entity-level metrics.

    python -m training.evaluate --checkpoint models/best_epoch_16.pt --output-dir reports/v1_probe

Writes ``metrics.json`` (entity-level P/R/F1 per type, v1's token-level metric for
comparison, an error breakdown, throughput) and ``errors.jsonl`` (every test sentence
the model got wrong, for error analysis).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from training.data import Sentence, load_vlsp_csv, remove_test_overlap, split_random, split_v1
from vietner.labels import bio_to_spans
from vietner.metrics import entity_report, error_breakdown, token_report

logger = logging.getLogger("evaluate")


def evaluate_tagger(tagger, sentences: list[Sentence], batch_size: int = 32) -> tuple[dict, list[list[str]]]:
    start = time.perf_counter()
    pred = tagger.predict([list(s.tokens) for s in sentences], batch_size=batch_size)
    seconds = time.perf_counter() - start
    gold = [list(s.tags) for s in sentences]
    metrics = {
        "entity": entity_report(gold, pred),
        "token_v1_metric": token_report(gold, pred),
        "errors": error_breakdown(gold, pred),
        "n_sentences": len(sentences),
        "sentences_per_second": len(sentences) / seconds if seconds else None,
    }
    return metrics, pred


def _spans(tokens, tags) -> list[tuple[str, str]]:
    return [(x.label, " ".join(tokens[x.start : x.end])) for x in bio_to_spans(list(tags))]


def write_errors(path: Path, sentences: list[Sentence], pred: list[list[str]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s, p in zip(sentences, pred, strict=True):
            if list(s.tags) == p:
                continue
            record = {"text": " ".join(s.tokens), "gold": _spans(s.tokens, s.tags), "pred": _spans(s.tokens, p)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_splits(data: str, scheme: str, split: str, tokenizer, seed: int):
    sentences = load_vlsp_csv(data, scheme)
    splits = split_v1(sentences, tokenizer, seed=seed) if split == "v1" else split_random(sentences, seed=seed)
    splits, removed = remove_test_overlap(splits)
    logger.info(
        "split=%s sizes: %s (removed as test duplicates: %s)", split, {k: len(v) for k, v in splits.items()}, removed
    )
    return splits, removed


def main(argv=None) -> None:
    from vietner.models import NERTagger

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/raw_data.csv")
    p.add_argument("--split", choices=["v1", "random"], default="v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tagger, meta = NERTagger.load(args.checkpoint, device=args.device)
    scheme = "legacy" if len(tagger.tags) == 7 else "vlsp"
    splits, _ = build_splits(args.data, scheme, args.split, tagger.encoder.tokenizer, args.seed)
    metrics, pred = evaluate_tagger(tagger, splits["test"])
    metrics.update(checkpoint=str(args.checkpoint), checkpoint_metadata=meta, split=args.split, scheme=scheme)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    n_err = write_errors(out / "errors.jsonl", splits["test"], pred)
    e = metrics["entity"]["micro"]
    print(f"entity-level micro F1 {e['f1']:.4f} (P {e['precision']:.4f} R {e['recall']:.4f}); "
          f"v1 token metric {metrics['token_v1_metric']['f1']:.4f}; {n_err} sentences with errors")  # fmt: skip


if __name__ == "__main__":
    main()
