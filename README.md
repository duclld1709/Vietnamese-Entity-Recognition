---
title: Vietnamese NER Demo
emoji: 🛡️
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.46.1
app_file: app.py
pinned: false
---

# 🛡️ Vietnamese NER & Personal-Data Anonymiser

[![CI](https://github.com/duclld1709/Vietnamese-Entity-Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/duclld1709/Vietnamese-Entity-Recognition/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PhoBERT](https://img.shields.io/badge/PhoBERT-CRF-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

**Find and anonymise personal data in Vietnamese text, so documents can go to an LLM without exposing
customers.** A PhoBERT-CRF model finds people, organisations and places. Validated rules catch phone
numbers, citizen ID (CCCD/CMND) numbers, emails, bank accounts, tax codes, card numbers, licence plates
and dates. The reversible anonymiser swaps each entity for a placeholder. You can send the text to any
LLM and then put the real names back into its answer.

🚀 **Live demo:** [huggingface.co/spaces/DucLai/Vietnamese_NER](https://huggingface.co/spaces/DucLai/Vietnamese_NER) (the free Space can take about a minute to wake up)

```text
Input   Ông Nguyễn Văn Nam (FPT, Hà Nội), SĐT 0912.345.678, CCCD 001090012345, STK 0071 0012 34567.
Tag     Ông [PER_1] ([ORG_1], [LOC_1]), SĐT [PHONE_1], CCCD [ID_NUMBER_1], STK [BANK_ACCOUNT_1].
Mask    Ông N***** V** N** (F**, H* N**), SĐT 09**.***.*78, CCCD *********345, STK **** **** **567.
```

## Why

Decree 13/2023/NĐ-CP on personal data protection makes Vietnamese companies responsible for personal
data they pass to third parties, including LLM APIs. Off-the-shelf PII tools (Presidio, cloud DLP)
handle English names and US formats. They miss "Nguyễn Văn Nam", `0912.345.678` and 12-digit CCCD
numbers. This project combines a Vietnamese NER model with rules for Vietnamese identifier formats.

## Features

- **Hybrid detection.** PhoBERT-CRF handles PER/ORG/LOC (MISC optional). Rules handle structured PII, with validation (mobile prefixes, CCCD province codes, Luhn check for cards, date ranges) and context keywords for ambiguous digit strings (`STK`, `MST`, `CMND`, …).
- **Exact character offsets.** pyvi word segmentation is aligned back to the original string, so entities can be replaced in place. Input is NFC-normalised (macOS and PDF text often arrives decomposed).
- **Three anonymisation strategies.** `tag` gives consistent, reversible placeholders (the same person always maps to `[PER_1]`). `mask` keeps text readable. `redact` is irreversible.
- **Long documents.** Text is split into sentences, word sequences are chunked to fit PhoBERT's 256-token window, and sentences from all inputs are batched with length bucketing.
- **Graceful degradation.** If the model can't load, the service keeps running in rules-only mode, reports `degraded` on `/health`, and attaches a warning to every response.
- **Interfaces.** Python library, CLI, FastAPI service, Streamlit demo, Docker image.
- **Honest evaluation.** Entity-level metrics on the v1 held-out test split, test duplicates removed from training, and an error taxonomy. Details below.

## Quickstart

```bash
pip install -e ".[api,app]"

vietner entities "Ông Nguyễn Văn Nam làm việc tại Hà Nội, SĐT 0912345678"
vietner anonymize contract.txt --strategy tag --mapping mapping.json > contract.anon.txt
vietner serve-api                       # http://localhost:8000/docs
streamlit run app.py                    # demo UI
```

```python
from vietner import VietNER, deanonymize

ner = VietNER.from_checkpoint()  # models/best_epoch_16.pt + vinai/phobert-base
safe = ner.anonymize("Chị Lê Thị Lan (0987654321) ở Đà Nẵng.", strategy="tag")
reply = call_your_llm(safe.text)  # the LLM only ever sees [PER_1], [PHONE_1], [LOC_1]
print(deanonymize(reply, safe.mapping))
```

Docker (CPU, PhoBERT baked into the image):

```bash
docker build -t vietner . && docker run -p 8000:8000 vietner
```

### REST API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/v1/entities` | `{"texts": [...], "ner": true, "rules": true}` (≤ 64 texts) | entities with `label`, `text`, `start`, `end`, `score`, `source` |
| `POST` | `/v1/anonymize` | `{"text": "...", "strategy": "tag", "labels": null}` | anonymised text, replacements, `mapping` |
| `POST` | `/v1/deanonymize` | `{"text": "...", "mapping": {...}}` | restored text |
| `GET` | `/health` | | `ok` / `degraded` (rules-only) |

## Labels

| Source | Labels |
|---|---|
| Model (VLSP 2016) | `PER`, `ORG`, `LOC` (and `MISC` with the 9-tag scheme) |
| Rules | `PHONE`, `EMAIL`, `ID_NUMBER` (CCCD/CMND), `TAX_CODE`, `BANK_ACCOUNT`, `CREDIT_CARD`, `LICENSE_PLATE`, `DATE`, `URL` |

When spans overlap, validated rule matches (score 0.85–0.99) win over model spans (0.8), then longer
spans win. `TAX_CODE`, `BANK_ACCOUNT` and 9-digit `ID_NUMBER` require a context keyword before the
number. Without one, a random 10-digit string would be flagged.

## Model and evaluation

The model is PhoBERT-base with a Linear-CRF head. Subword states are mean-pooled into one vector per
word. Two training modes are provided:

| Mode | What is trained | Checkpoint |
|---|---|---|
| `probe` (v1 recipe) | Linear-CRF head on frozen PhoBERT features | 24 KB head; `models/best_epoch_16.pt` |
| `finetune` | PhoBERT + head end to end (AdamW, warmup, AMP) | full model |

```bash
python -m training.evaluate --checkpoint models/best_epoch_16.pt --output-dir reports/v1_probe
python -m training.train --mode probe    --output-dir runs/probe
python -m training.train --mode finetune --output-dir runs/finetune --amp
```

**Evaluation protocol.**
- **Same test set as v1.** `--split v1` rebuilds v1's exact train/val/test partition. v1's dropped rows depend only on the tokenizer. This lets the published checkpoint and new models be scored on the same held-out sentences.
- **No duplicate leakage.** VLSP contains repeated sentences, so sentences that also occur in test are removed from train and val.
- **Entity-level metric.** An entity counts only if span and type both match (CoNLL/VLSP standard). It is cross-checked against `seqeval` in the tests.
- **Error taxonomy.** Every mistake is classified as wrong type, wrong boundary, missed or spurious, and every wrong sentence is written to `errors.jsonl`.

**About the v1 numbers.** v1 reported F1 = 0.91 as **token-level macro F1 including the `O` tag**.
About 94% of tokens are `O`, which inflates that score, and the metric does not check whether entity
boundaries are right. The v1 dashboard also hard-coded a per-class report that does not match the
training log (kept in [`reports/v1_reported.json`](reports/v1_reported.json)). Entity-level results
for the v1 checkpoint and the fine-tuned model will be published here after the evaluation runs. The
Streamlit *Evaluation* tab reads them straight from `reports/*/metrics.json`, so nothing is typed in by
hand.

## Engineering audit of v1 → v2

| v1 issue | Impact | v2 |
|---|---|---|
| PhoBERT and the head reloaded on **every** prediction | Several seconds per request | Model loaded once; batched inference |
| No length handling; PhoBERT has 256 positions | Crash on long inputs | Sentence splitting + subword-aware chunking |
| Sentences whose `@@` regrouping mismatched were silently dropped | Lost training data; `zip` truncation at inference | Per-word tokenisation guarantees one vector per word |
| Token-level macro F1 incl. `O` | Inflated, boundary-blind metric | Entity-level P/R/F1 + error taxonomy |
| Random split with duplicate sentences across splits | Test leakage | Same v1 test set, duplicates removed from train/val |
| Hard-coded metrics in the dashboard (`results/output.py`) | Numbers not traceable to a run | UI reads generated `metrics.json` |
| User text injected into HTML unescaped (`unsafe_allow_html`) | HTML/script injection in the demo | Escaped rendering, tested |
| Deploy on every push to `main`, no tests | Broken code could ship | CI (ruff + 68 tests) gates the Space deploy |
| `.pyc` files committed, README describing files that don't exist | Confusing repo | Cleaned |

## Project structure

```
vietner/      text (segmentation + offsets) · encoder (PhoBERT word encoder) · models (CRF head)
              pii (rules) · anonymize · pipeline · metrics · api · cli · render
training/     data (VLSP loading, v1 split) · train (probe / finetune) · evaluate
app.py        Streamlit demo / Hugging Face Space
models/       best_epoch_16.pt (v1 probe head)
reports/      v1 training log and reported numbers; evaluation outputs land here
notebooks/    v1 experiments (CRF, softmax, random forest, rule-based)
tests/        68 tests: offsets, PII rules, anonymiser, encoder parity with v1, real checkpoint, API, UI, training smoke
```

## Development

```bash
pip install -e ".[dev]"
pytest              # ~10 s on CPU, no model download (tiny random PhoBERT stand-in)
ruff check . && ruff format --check .
```

## Roadmap

- [ ] Publish entity-level results: v1 probe vs fine-tuned PhoBERT vs zero-shot LLM (F1 · latency · cost per 1k sentences)
- [ ] Coreference-aware placeholders ("ông Nam" → same `[PER_1]` as "Nguyễn Văn Nam")
- [ ] Address detection (house number + street + ward), which VLSP's `LOC` does not cover
- [ ] Out-of-domain test set (contracts, CVs, chat logs); VLSP is news text
- [ ] ONNX export and quantisation for CPU serving

## Acknowledgements

VLSP 2016 NER dataset · [PhoBERT](https://github.com/VinAIResearch/PhoBERT) (VinAI) ·
[pyvi](https://github.com/trungtv/pyvi) · CRF layer vendored from
[pytorch-crf](https://github.com/kmkurn/pytorch-crf) (MIT). v1 report: [`docs/report.pdf`](docs/report.pdf).
See the [model card](docs/MODEL_CARD.md) for intended use and limitations.
