# Model card: PhoBERT-CRF (`models/best_epoch_16.pt`)

| | |
|---|---|
| Task | Named entity recognition for Vietnamese: PER, ORG, LOC (BIO, 7 tags) |
| Architecture | `vinai/phobert-base` (frozen) → mean-pooled subwords per word → Linear (768→7) → CRF |
| Training data | VLSP 2016 NER (news), train+valid dump from `datnth1709/VLSP2016-NER-data`; MISC mapped to `O` |
| Input | Word-segmented text (pyvi at inference, gold segmentation in training) |
| Size | 24 KB head on top of PhoBERT-base (135M parameters, downloaded from the Hub) |

## Intended use

- Detect person, organisation and location names in Vietnamese text, combined with the rule-based
  recognisers, to pseudonymise documents before they reach third-party services such as LLM APIs.
- Treat it as a first-pass filter with human review for high-stakes data. It does not guarantee that
  all personal data is removed.

## Evaluation

v1 reported a token-level macro F1 of 0.91 that includes the `O` tag. That is not comparable with
standard NER results. Run `python -m training.evaluate` to get entity-level P/R/F1 on v1's own
held-out test split. The README and the demo's *Evaluation* tab show those numbers once generated.

## Known limitations

- **Domain shift.** Trained on news. Contracts, CVs, chat messages, all-lowercase or unaccented text
  ("nguyen van nam") will have lower recall.
- **Segmentation mismatch.** Training used gold word segmentation, inference uses pyvi. pyvi errors
  (e.g. "Lê Thị_Hoà" instead of "Lê_Thị_Hoà") can split names.
- **Frozen encoder.** The head is a linear probe. A fine-tuned PhoBERT (`--mode finetune`) is
  expected to be markedly better, especially on ORG.
- **No confidence.** CRF outputs are not calibrated. Model spans get a fixed score of 0.8.
- **Rules are format-based.** Rules check the format of a number but cannot tell whose number it is.
  A CCCD-shaped number that is really an order ID will be masked (a safe failure), and a phone number
  written in words will be missed.
- **Coreference.** "Nguyễn Văn Nam" and a later "anh Nam" get different placeholders.

## Ethical considerations

A missed entity leaks personal data, while a spurious one only hides harmless text. The defaults
therefore favour recall for structured identifiers. Keep the placeholder→original mapping as
sensitive as the original document. Anyone who has both can re-identify the people in it.
