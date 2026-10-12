"""Word-level PhoBERT encoder.

Each word is BPE-tokenised on its own (PhoBERT's tokenizer splits on whitespace, so
this matches tokenising the whole sentence) and its subword states are mean-pooled,
exactly like v1's ``group_embeddings``. Doing it per word guarantees one vector per
word. v1 instead re-grouped ``@@`` pieces and dropped sentences whose counts did not
match.

PhoBERT has 258 positions, i.e. 256 tokens with ``<s>``/``</s>``. v1 crashed on longer
inputs; :meth:`WordEncoder.encode` splits word sequences into chunks that fit.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import torch
import torch.nn as nn

DEFAULT_ENCODER = "vinai/phobert-base"
MAX_SUBWORDS = 254  # 256 positions minus <s> and </s>


class SubwordTokenizer(Protocol):
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int
    unk_token_id: int

    def tokenize(self, text: str) -> list[str]: ...

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...


class WordEncoder(nn.Module):
    def __init__(self, transformer: nn.Module, tokenizer: SubwordTokenizer, max_subwords: int = MAX_SUBWORDS):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_subwords = max_subwords
        self._subword_ids = lru_cache(maxsize=200_000)(self._tokenize_word)

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_ENCODER, **kwargs) -> WordEncoder:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        return cls(AutoModel.from_pretrained(name), tokenizer, **kwargs)

    @property
    def hidden_size(self) -> int:
        return int(self.transformer.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    # ------------------------------------------------------------------ tokenisation

    def _tokenize_word(self, word: str) -> tuple[int, ...]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
        return tuple(ids) or (self.tokenizer.unk_token_id,)

    def subword_ids(self, word: str) -> tuple[int, ...]:
        return self._subword_ids(word)

    def chunk(self, words: list[str]) -> list[list[str]]:
        """Split ``words`` into consecutive chunks of at most ``max_subwords`` subwords.
        A single word longer than the limit is truncated to its first pieces."""

        chunks: list[list[str]] = [[]]
        used = 0
        for w in words:
            n = min(len(self.subword_ids(w)), self.max_subwords)
            if used + n > self.max_subwords and chunks[-1]:
                chunks.append([])
                used = 0
            chunks[-1].append(w)
            used += n
        return [c for c in chunks if c] or [[]]

    def batch_inputs(self, words_batch: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``input_ids``, ``attention_mask`` and ``word_index`` (subword -> word, -1
        for special/pad positions), all ``(B, L)``."""

        tok = self.tokenizer
        rows, owners = [], []
        for words in words_batch:
            ids, own = [tok.cls_token_id], [-1]
            for wi, w in enumerate(words):
                piece = self.subword_ids(w)[: self.max_subwords]
                ids.extend(piece)
                own.extend([wi] * len(piece))
            if len(ids) - 1 > self.max_subwords:
                raise ValueError(
                    f"sequence of {len(ids) - 1} subwords exceeds {self.max_subwords}; use encode() to chunk"
                )
            ids.append(tok.sep_token_id)
            own.append(-1)
            rows.append(ids)
            owners.append(own)

        length = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), length), tok.pad_token_id, dtype=torch.long)
        word_index = torch.full((len(rows), length), -1, dtype=torch.long)
        for i, (r, o) in enumerate(zip(rows, owners, strict=True)):
            input_ids[i, : len(r)] = torch.tensor(r)
            word_index[i, : len(o)] = torch.tensor(o)
        attention_mask = torch.zeros_like(input_ids)
        for i, r in enumerate(rows):
            attention_mask[i, : len(r)] = 1
        return input_ids, attention_mask, word_index

    # ------------------------------------------------------------------ forward

    def forward(self, words_batch: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return word embeddings ``(B, W, H)`` and a word mask ``(B, W)``.

        Every sequence must fit in ``max_subwords``. Differentiable, for fine-tuning.
        """

        input_ids, attention_mask, word_index = (t.to(self.device) for t in self.batch_inputs(words_batch))
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return mean_pool(hidden, word_index, max(1, max(len(w) for w in words_batch)))

    @torch.no_grad()
    def encode(self, words_batch: list[list[str]], batch_size: int = 16) -> list[torch.Tensor]:
        """Inference helper: one ``(n_words, H)`` CPU tensor per input, chunking long inputs."""

        # (input index, chunk position, words); sorted by length so batches pad less.
        pieces = [(i, k, c) for i, words in enumerate(words_batch) for k, c in enumerate(self.chunk(words)) if c]
        pieces.sort(key=lambda p: len(p[2]))
        parts: list[list[tuple[int, torch.Tensor]]] = [[] for _ in words_batch]
        for b in range(0, len(pieces), batch_size):
            batch = pieces[b : b + batch_size]
            emb, _ = self.forward([p[2] for p in batch])
            for (i, k, words), e in zip(batch, emb, strict=True):
                parts[i].append((k, e[: len(words)].float().cpu()))
        return [
            torch.cat([t for _, t in sorted(p, key=lambda x: x[0])]) if p else torch.zeros(0, self.hidden_size)
            for p in parts
        ]


def mean_pool(hidden: torch.Tensor, word_index: torch.Tensor, num_words: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average subword states per word. ``word_index`` is ``-1`` for positions to ignore."""

    b, length, h = hidden.shape
    slots = num_words + 1  # slot 0 collects ignored positions
    flat_index = (word_index + 1) + torch.arange(b, device=hidden.device).unsqueeze(1) * slots
    sums = hidden.new_zeros(b * slots, h).index_add_(0, flat_index.reshape(-1), hidden.reshape(-1, h))
    counts = hidden.new_zeros(b * slots).index_add_(0, flat_index.reshape(-1), hidden.new_ones(b * length))
    sums, counts = sums.view(b, slots, h)[:, 1:], counts.view(b, slots)[:, 1:]
    mask = counts > 0
    return sums / counts.clamp_min(1).unsqueeze(-1), mask
