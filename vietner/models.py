"""PhoBERT word encoder + Linear-CRF tagging head.

Two training regimes share this module:

* ``probe``    - frozen PhoBERT, only the head is trained (v1, ``models/best_epoch_16.pt``).
* ``finetune`` - PhoBERT is trained end to end together with the head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vietner.crf import CRF
from vietner.encoder import DEFAULT_ENCODER, WordEncoder
from vietner.labels import LEGACY_TAGS

CHECKPOINT_FORMAT = "vietner/v2"


class CRFHead(nn.Module):
    """Emission layer + CRF. Parameter names (``embed2tag``, ``crf``) match v1."""

    def __init__(self, input_dim: int, num_tags: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed2tag = nn.Linear(input_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def loss(self, x: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood. Padded tag positions may hold any value."""

        tags = tags.clamp_min(0)
        return -self.crf(self.embed2tag(self.dropout(x)), tags, mask=mask, reduction="mean")

    def decode(self, x: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        return self.crf.decode(self.embed2tag(x), mask=mask)


@dataclass
class TaggerConfig:
    encoder_name: str = DEFAULT_ENCODER
    tags: tuple[str, ...] = LEGACY_TAGS
    mode: str = "probe"  # probe | finetune
    dropout: float = 0.1

    def __post_init__(self):
        self.tags = tuple(self.tags)
        if self.mode not in ("probe", "finetune"):
            raise ValueError(f"unknown mode {self.mode!r}")


class NERTagger(nn.Module):
    def __init__(self, encoder: WordEncoder, config: TaggerConfig | None = None):
        super().__init__()
        self.config = config or TaggerConfig()
        self.encoder = encoder
        self.head = CRFHead(encoder.hidden_size, len(self.config.tags), self.config.dropout)
        if self.config.mode == "probe":
            self.encoder.requires_grad_(False)

    @property
    def tags(self) -> tuple[str, ...]:
        return self.config.tags

    def loss(self, words_batch: list[list[str]], tag_ids: list[list[int]]) -> torch.Tensor:
        """End-to-end loss (fine-tuning). Sequences must fit the encoder's subword limit."""

        feats, mask = self.encoder(words_batch)
        return self.head.loss(feats, pad_tags(tag_ids, feats.shape[1], feats.device), mask)

    @torch.no_grad()
    def predict(self, words_batch: list[list[str]], batch_size: int = 16) -> list[list[str]]:
        """Tag each word sequence. Long sequences are chunked by the encoder."""

        was_training = self.training
        self.eval()
        try:
            feats = self.encoder.encode(words_batch, batch_size=batch_size)
            ids = decode_features(self.head, feats, self.encoder.device)
        finally:
            self.train(was_training)
        return [[self.tags[i] for i in seq] for seq in ids]

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "config": asdict(self.config),
                "head": self.head.state_dict(),
                # A probe leaves PhoBERT untouched, so only the 24 KB head is stored.
                "encoder": self.encoder.state_dict() if self.config.mode == "finetune" else None,
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu", encoder: WordEncoder | None = None
    ) -> tuple[NERTagger, dict[str, Any]]:
        """Load a v2 checkpoint or v1's raw head ``state_dict``.

        ``encoder`` overrides the PhoBERT download (tests, offline use).
        """

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
            config = TaggerConfig(**payload["config"])
            head_state, encoder_state = payload["head"], payload.get("encoder")
            metadata = dict(payload.get("metadata") or {})
        else:
            config = TaggerConfig(tags=LEGACY_TAGS[: payload["embed2tag.weight"].shape[0]], mode="probe")
            head_state, encoder_state, metadata = payload, None, {"source": "v1-state-dict"}

        encoder = encoder or WordEncoder.from_pretrained(config.encoder_name)
        if encoder_state is not None:
            encoder.load_state_dict(encoder_state)
        model = cls(encoder, config)
        model.head.load_state_dict(head_state)
        return model.to(device).eval(), metadata


def pad_tags(tag_ids: list[list[int]], length: int, device) -> torch.Tensor:
    out = torch.full((len(tag_ids), length), -1, dtype=torch.long, device=device)
    for i, seq in enumerate(tag_ids):
        out[i, : len(seq)] = torch.tensor(seq[:length], dtype=torch.long, device=device)
    return out


def pad_features(feats: list[torch.Tensor], device) -> tuple[torch.Tensor, torch.Tensor]:
    """List of ``(n_i, H)`` -> ``(B, max_n, H)`` and a bool mask."""

    length = max(1, max(f.shape[0] for f in feats))
    x = torch.zeros(len(feats), length, feats[0].shape[1], device=device)
    mask = torch.zeros(len(feats), length, dtype=torch.bool, device=device)
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = f.to(device)
        mask[i, : f.shape[0]] = True
    return x, mask


def decode_features(head: CRFHead, feats: list[torch.Tensor], device, batch_size: int = 64) -> list[list[int]]:
    """Viterbi-decode precomputed word features; empty sequences decode to ``[]``."""

    out: list[list[int]] = [[] for _ in feats]
    nonempty = [i for i, f in enumerate(feats) if f.shape[0]]
    for b in range(0, len(nonempty), batch_size):
        idx = nonempty[b : b + batch_size]
        x, mask = pad_features([feats[i] for i in idx], device)
        for i, seq in zip(idx, head.decode(x, mask), strict=True):
            out[i] = seq
    return out
