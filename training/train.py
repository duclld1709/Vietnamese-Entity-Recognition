"""Train a PhoBERT-CRF tagger.

    # v1 recipe (frozen PhoBERT, Linear-CRF head), now selected on entity-level F1
    python -m training.train --mode probe --output-dir runs/probe

    # end-to-end fine-tuning
    python -m training.train --mode finetune --epochs 5 --batch-size 16 --output-dir runs/finetune

Both modes use the v1 test split (see training/data.py) with test duplicates removed
from train/val, so results are directly comparable with the published checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from training.data import Sentence, tag_ids
from training.evaluate import build_splits, evaluate_tagger, write_errors
from vietner.encoder import DEFAULT_ENCODER, WordEncoder
from vietner.labels import LEGACY_TAGS, VLSP_TAGS
from vietner.metrics import entity_report, token_report
from vietner.models import NERTagger, TaggerConfig, decode_features, pad_features, pad_tags

logger = logging.getLogger("train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw_data.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    p.add_argument("--scheme", choices=["legacy", "vlsp"], default="legacy", help="legacy = 7 tags (MISC -> O)")
    p.add_argument("--split", choices=["v1", "random"], default="v1")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--epochs", type=int, default=None, help="default: 20 (probe) / 5 (finetune)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--encoder-lr", type=float, default=3e-5, help="PhoBERT learning rate (finetune)")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="bf16/fp16 autocast on CUDA (finetune)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N sentences (debugging)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunk_for_encoder(sentences: list[Sentence], encoder: WordEncoder) -> list[Sentence]:
    """Split sentences longer than PhoBERT's window (v1 would have crashed on them)."""

    out = []
    for s in sentences:
        start = 0
        for chunk in encoder.chunk(list(s.tokens)):
            out.append(Sentence(s.tokens[start : start + len(chunk)], s.tags[start : start + len(chunk)]))
            start += len(chunk)
    return out


class Logger:
    def __init__(self, enabled: bool, config: dict):
        self.run = None
        if enabled:
            import wandb

            self.run = wandb.init(project="vietner", config=config)

    def log(self, data: dict) -> None:
        if self.run:
            self.run.log(data)

    def finish(self) -> None:
        if self.run:
            self.run.finish()


def train_probe(model: NERTagger, splits, args, device, log: Logger, out: Path) -> None:
    """Frozen encoder: embed every sentence once, then train only the head (fast)."""

    tags = model.tags
    feats = {
        k: model.encoder.encode([list(s.tokens) for s in v], batch_size=64) for k, v in splits.items() if k != "test"
    }
    labels = {k: tag_ids(splits[k], tags) for k in feats}
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    order = list(range(len(feats["train"])))
    best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.head.train()
        random.shuffle(order)
        total = 0.0
        for b in range(0, len(order), args.batch_size):
            idx = order[b : b + args.batch_size]
            x, mask = pad_features([feats["train"][i] for i in idx], device)
            loss = model.head.loss(x, pad_tags([labels["train"][i] for i in idx], x.shape[1], device), mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.head.eval()
        pred = decode_features(model.head, feats["val"], device)
        val_pred = [[tags[i] for i in seq] for seq in pred]
        best, stale = _end_epoch(model, splits, epoch, total / len(order), best, stale, log, out, args, val_pred)
        if stale >= args.patience:
            logger.info("early stopping")
            break


def train_finetune(model: NERTagger, splits, args, device, log: Logger, out: Path) -> None:
    from transformers import get_linear_schedule_with_warmup

    train = splits["train"]
    ids = tag_ids(train, model.tags)
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=0.01,
    )
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)
    use_amp = args.amp and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    order = list(range(len(train)))
    best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(order)
        total = 0.0
        for step, b in enumerate(range(0, len(order), args.batch_size)):
            idx = order[b : b + args.batch_size]
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                feats, mask = model.encoder([list(train[i].tokens) for i in idx])
            loss = model.head.loss(feats.float(), pad_tags([ids[i] for i in idx], feats.shape[1], device), mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
            if step % 100 == 0:
                logger.info("  step %d loss %.4f", step, loss.item())
        best, stale = _end_epoch(model, splits, epoch, total / len(order), best, stale, log, out, args)
        if stale >= args.patience:
            logger.info("early stopping")
            break


def _end_epoch(model, splits, epoch, train_loss, best, stale, log, out, args, val_pred=None):
    if val_pred is None:
        val, _ = evaluate_tagger(model, splits["val"])
    else:  # probe: predictions decoded from cached features
        gold = [list(s.tags) for s in splits["val"]]
        val = {"entity": entity_report(gold, val_pred), "token_v1_metric": token_report(gold, val_pred)}
    f1 = val["entity"]["micro"]["f1"]
    logger.info("epoch %d: train loss %.4f | val entity F1 %.4f", epoch, train_loss, f1)
    log.log(
        {"epoch": epoch, "train/loss": train_loss, "val/entity_f1": f1, "val/token_f1": val["token_v1_metric"]["f1"]}
    )
    with (out / "history.jsonl").open("a") as f:
        f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "val": val}) + "\n")
    if f1 > best:
        model.save(out / "best.pt", epoch=epoch, val_entity_f1=f1, args=vars(args))
        return f1, 0
    return best, stale + 1


def main(argv=None, encoder: WordEncoder | None = None) -> dict:
    """``encoder`` lets tests inject a tiny model instead of downloading PhoBERT."""

    args = parse_args(argv)
    args.epochs = args.epochs or (20 if args.mode == "probe" else 5)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.jsonl").unlink(missing_ok=True)

    encoder = encoder or WordEncoder.from_pretrained(args.encoder)
    config = TaggerConfig(
        encoder_name=args.encoder,
        tags=LEGACY_TAGS if args.scheme == "legacy" else VLSP_TAGS,
        mode=args.mode,
        dropout=args.dropout,
    )
    model = NERTagger(encoder, config).to(device)

    splits, removed = build_splits(args.data, args.scheme, args.split, encoder.tokenizer, args.seed)
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    splits["train"] = chunk_for_encoder(splits["train"], encoder)
    if args.mode == "probe":
        model.encoder.eval()

    log = Logger(args.wandb, vars(args))
    start = time.time()
    (train_probe if args.mode == "probe" else train_finetune)(model, splits, args, device, log, out)

    best, meta = NERTagger.load(out / "best.pt", device=device, encoder=encoder)
    test, pred = evaluate_tagger(best, splits["test"])
    test.update(
        mode=args.mode,
        scheme=args.scheme,
        split=args.split,
        best_epoch=meta.get("epoch"),
        removed_test_duplicates=removed,
        sizes={k: len(v) for k, v in splits.items()},
        minutes=round((time.time() - start) / 60, 2),
    )
    (out / "metrics.json").write_text(json.dumps(test, indent=2, ensure_ascii=False))
    write_errors(out / "errors.jsonl", splits["test"], pred)
    e = test["entity"]["micro"]
    log.log({"test/entity_f1": e["f1"], "test/token_f1": test["token_v1_metric"]["f1"]})
    log.finish()
    logger.info("test entity F1 %.4f (P %.4f R %.4f)", e["f1"], e["precision"], e["recall"])
    return test


if __name__ == "__main__":
    main()
