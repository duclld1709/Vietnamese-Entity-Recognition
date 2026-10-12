"""REST API: ``uvicorn vietner.api:app`` or ``vietner serve-api``."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from vietner import __version__
from vietner.anonymize import DEFAULT_LABELS, STRATEGIES, Strategy, deanonymize
from vietner.pipeline import MAX_CHARS, VietNER
from vietner.schemas import AnalyzeResult, AnonymizeResult

logger = logging.getLogger(__name__)
MAX_TEXTS = 64


class EntitiesRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)
    ner: bool = Field(True, description="Run the PhoBERT-CRF model (PER/ORG/LOC).")
    rules: bool = Field(True, description="Run rule-based PII recognisers (phone, ID, email, ...).")


class AnonymizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    strategy: Strategy = "tag"
    labels: list[str] | None = Field(None, description=f"Labels to replace (default: {sorted(DEFAULT_LABELS)}).")


class DeanonymizeRequest(BaseModel):
    text: str
    mapping: dict[str, str]


class DeanonymizeResponse(BaseModel):
    text: str


def create_app(factory: Callable[[], VietNER] | None = None) -> FastAPI:
    factory = factory or VietNER.from_checkpoint
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["ner"] = factory()  # loaded once, not per request as in v1
        yield

    app = FastAPI(
        title="Vietnamese NER & PII API",
        version=__version__,
        description="Named entities (PhoBERT-CRF) and rule-based personal data detection for Vietnamese, "
        "with reversible anonymisation for sending documents to external LLMs.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    @app.get("/health")
    def health():
        ner: VietNER = state["ner"]
        return {
            "status": "ok" if ner.ner_available else "degraded",
            "ner_model": ner.ner_available,
            "detail": ner.unavailable_reason,
            "strategies": list(STRATEGIES),
        }

    @app.post("/v1/entities", response_model=list[AnalyzeResult])
    def entities(req: EntitiesRequest) -> list[AnalyzeResult]:
        try:
            return state["ner"].analyze(req.texts, ner=req.ner, rules=req.rules)
        except ValueError as exc:
            raise HTTPException(413, str(exc)) from exc

    @app.post("/v1/anonymize", response_model=AnonymizeResult)
    def anonymize(req: AnonymizeRequest) -> AnonymizeResult:
        return state["ner"].anonymize(req.text, req.strategy, req.labels)

    @app.post("/v1/deanonymize", response_model=DeanonymizeResponse)
    def restore(req: DeanonymizeRequest) -> DeanonymizeResponse:
        return DeanonymizeResponse(text=deanonymize(req.text, req.mapping))

    return app


def __getattr__(name: str):
    if name == "app":  # built lazily so importing the module has no side effects
        globals()["app"] = application = create_app()
        return application
    raise AttributeError(name)
