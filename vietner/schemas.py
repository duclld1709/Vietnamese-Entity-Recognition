"""Data models shared by the library, REST API and UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str = Field(description="PER, ORG, LOC, MISC (model) or EMAIL, PHONE, ID_NUMBER, ... (rules)")
    text: str
    start: int = Field(description="Character offset in the (NFC-normalised) text, inclusive.")
    end: int = Field(description="Character offset, exclusive.")
    score: float = 1.0
    source: Literal["model", "rule"]


class AnalyzeResult(BaseModel):
    text: str = Field(description="NFC-normalised input; offsets refer to this string.")
    entities: list[Entity]
    warnings: list[str] = []


class Replacement(BaseModel):
    label: str
    original: str
    replacement: str
    start: int
    end: int


class AnonymizeResult(BaseModel):
    text: str
    replacements: list[Replacement]
    mapping: dict[str, str] = Field(
        default_factory=dict, description="placeholder -> original; only filled by the reversible 'tag' strategy."
    )
    warnings: list[str] = []
