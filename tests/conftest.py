from __future__ import annotations

import zlib
from pathlib import Path

import pytest
import torch

from vietner.encoder import WordEncoder

ROOT = Path(__file__).resolve().parent.parent
V1_CHECKPOINT = ROOT / "models" / "best_epoch_16.pt"
VOCAB = 500


class FakePhoBERTTokenizer:
    """Mimics PhoBERT's BPE output format: non-final pieces end with ``@@``.
    Pieces containing "Ω" are out of vocabulary and come back as ``<unk>``."""

    cls_token_id, pad_token_id, sep_token_id, unk_token_id = 0, 1, 2, 3

    def __init__(self):
        self.inverse = {0: "<s>", 1: "<pad>", 2: "</s>", 3: "<unk>"}

    def tokenize(self, text: str) -> list[str]:
        out = []
        for word in text.split():
            pieces = [word[i : i + 3] for i in range(0, len(word), 3)]
            out.extend([p + "@@" for p in pieces[:-1]] + pieces[-1:])
        return out

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        ids = []
        for t in tokens:
            # Hash into the vocab, using id parity to keep the "@@" property stable under
            # collisions (v1's grouping only looks at that suffix after the id round trip).
            i = self.unk_token_id if "Ω" in t else 4 + 2 * (zlib.crc32(t.encode()) % 247) + t.endswith("@@")
            self.inverse.setdefault(i, t)
            ids.append(i)
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.inverse[i] for i in ids]


def tiny_encoder(hidden: int = 32, max_subwords: int = 254, seed: int = 0) -> WordEncoder:
    from transformers import RobertaConfig, RobertaModel

    torch.manual_seed(seed)
    config = RobertaConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=258,
        pad_token_id=1,
    )
    return WordEncoder(RobertaModel(config).eval(), FakePhoBERTTokenizer(), max_subwords=max_subwords)


@pytest.fixture(scope="session")
def encoder() -> WordEncoder:
    return tiny_encoder()


@pytest.fixture(scope="session")
def encoder768() -> WordEncoder:
    """Same width as PhoBERT-base, so the real v1 head can be loaded on top."""

    return tiny_encoder(hidden=768)


class StubTagger:
    """Tags words from a fixed dictionary; everything else is O."""

    tags = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")

    def __init__(self, lexicon: dict[str, str] | None = None):
        self.lexicon = lexicon or {"Nguyễn_Văn_Nam": "PER", "Hà_Nội": "LOC", "FPT": "ORG", "Vietcombank": "ORG"}
        self.calls: list[list[list[str]]] = []

    def predict(self, words_batch, batch_size=16):
        self.calls.append(words_batch)
        return [[f"B-{self.lexicon[w]}" if w in self.lexicon else "O" for w in ws] for ws in words_batch]
