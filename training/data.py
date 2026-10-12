"""VLSP2016 loading and leakage-aware splits.

``data/raw_data.csv`` is the HF dump ``datnth1709/VLSP2016-NER-data`` (train + valid),
with tokens/tags stored as numpy array reprs.

Splits
------
``v1`` reproduces the exact train/val/test partition used to train the published
checkpoint. v1 dropped every sentence whose ``@@``-regrouped PhoBERT tokens did not
match its tag count, then called ``train_test_split(random_state=42)`` twice
(70/15/15). Dropped rows depend only on the tokenizer, so the partition can be rebuilt
without the original embeddings. This lets the old and new models be scored on the
same held-out test set.

On top of that, sentences that also occur in the test set are removed from train/val
(VLSP contains exact duplicates). v1 did not do this.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from vietner.labels import LEGACY_TAGS, VLSP_TAGS

_QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


@dataclass(frozen=True)
class Sentence:
    tokens: tuple[str, ...]
    tags: tuple[str, ...]


def _parse_tokens(raw: str) -> list[str]:
    out = []
    for single, double in _QUOTED.findall(raw):
        value = single if single or not double else double
        out.append(ast.literal_eval(f"'{value}'") if "\\" in value else value)
    return out


def load_vlsp_csv(path: str | Path, scheme: str = "legacy") -> list[Sentence]:
    """``scheme='legacy'`` maps MISC to O (7 tags, like v1); ``'vlsp'`` keeps all 9."""

    if scheme not in ("legacy", "vlsp"):
        raise ValueError("scheme must be 'legacy' or 'vlsp'")
    df = pd.read_csv(path)
    sentences = []
    for raw_tokens, raw_tags in zip(df["tokens"], df["ner_tags"], strict=True):
        tokens = _parse_tokens(raw_tokens)
        ids = [int(x) for x in re.findall(r"\d+", raw_tags)]
        if len(tokens) != len(ids):
            raise ValueError(f"token/tag length mismatch: {raw_tokens[:60]}")
        tags = [VLSP_TAGS[i] for i in ids]
        if scheme == "legacy":
            tags = [t if t in LEGACY_TAGS else "O" for t in tags]
        sentences.append(Sentence(tuple(tokens), tuple(tags)))
    return sentences


def v1_kept(sentence: Sentence, tokenizer) -> bool:
    """Whether v1's ``group_embeddings`` produced one vector per tag for this sentence."""

    # Round-trip through ids like v1 did: an out-of-vocabulary "xx@@" piece comes back
    # as "<unk>", which no longer ends in "@@" and so closed a word in v1's grouping.
    pieces = tokenizer.convert_ids_to_tokens(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(sentence.tokens)))
    )
    words = sum(1 for p in pieces if not p.endswith("@@"))
    if pieces and pieces[-1].endswith("@@"):
        words += 1
    return words == len(sentence.tags)


def split_v1(
    sentences: list[Sentence], tokenizer, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> dict[str, list[Sentence]]:
    kept = [s for s in sentences if v1_kept(s, tokenizer)]
    idx = list(range(len(kept)))
    train_val, test = train_test_split(idx, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    return {k: [kept[i] for i in v] for k, v in {"train": train, "val": val, "test": test}.items()}


def split_random(
    sentences: list[Sentence], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> dict[str, list[Sentence]]:
    idx = list(range(len(sentences)))
    train_val, test = train_test_split(idx, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    return {k: [sentences[i] for i in v] for k, v in {"train": train, "val": val, "test": test}.items()}


def remove_test_overlap(splits: dict[str, list[Sentence]]) -> tuple[dict[str, list[Sentence]], dict[str, int]]:
    """Drop train/val sentences whose token sequence also appears in test (and train
    sentences that appear in val). Returns the cleaned splits and removal counts."""

    test_keys = {s.tokens for s in splits["test"]}
    val = [s for s in splits["val"] if s.tokens not in test_keys]
    blocked = test_keys | {s.tokens for s in val}
    train = [s for s in splits["train"] if s.tokens not in blocked]
    removed = {"train": len(splits["train"]) - len(train), "val": len(splits["val"]) - len(val)}
    return {"train": train, "val": val, "test": splits["test"]}, removed


def tag_ids(sentences: Iterable[Sentence], tags: tuple[str, ...]) -> list[list[int]]:
    index = {t: i for i, t in enumerate(tags)}
    return [[index[t] for t in s.tags] for s in sentences]
