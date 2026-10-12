import json
from collections import Counter

import pandas as pd
import pytest

from tests.conftest import ROOT, FakePhoBERTTokenizer, tiny_encoder
from training import train
from training.data import Sentence, load_vlsp_csv, remove_test_overlap, split_v1, v1_kept


@pytest.fixture(scope="module")
def vlsp():
    return load_vlsp_csv(ROOT / "data" / "raw_data.csv", scheme="vlsp")


def test_load_real_dataset(vlsp):
    assert len(vlsp) == 16858
    counts = Counter(t for s in vlsp for t in s.tags)
    assert counts["B-PER"] == 7479 and counts["B-MISC"] == 282
    legacy = load_vlsp_csv(ROOT / "data" / "raw_data.csv", scheme="legacy")
    assert not any("MISC" in t for s in legacy for t in s.tags)
    assert any('"' in s.tokens for s in vlsp)  # quote tokens survive the numpy-repr parsing


def test_v1_split_reproduces_v1_filtering_and_ratios(vlsp):
    tok = FakePhoBERTTokenizer()
    assert v1_kept(Sentence(("Hà_Nội", "đẹp"), ("B-LOC", "O")), tok)
    # an out-of-vocabulary non-final piece becomes <unk>, which split the word in v1
    assert not v1_kept(Sentence(("ΩΩΩΩ",), ("O",)), tok)
    splits = split_v1(vlsp[:2000], tok)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes == {"train": 1400, "val": 300, "test": 300}
    assert split_v1(vlsp[:2000], tok)["test"] == splits["test"]  # deterministic


def test_remove_test_overlap():
    a, b, c = (Sentence((w,), ("O",)) for w in "abc")
    cleaned, removed = remove_test_overlap({"train": [a, b, c], "val": [a, b], "test": [a]})
    assert cleaned == {"train": [c], "val": [b], "test": [a]} and removed == {"train": 2, "val": 1}


@pytest.fixture
def tiny_csv(tmp_path, vlsp):
    path = tmp_path / "vlsp.csv"
    rows = vlsp[:60]
    id_of = {
        t: i for i, t in enumerate(("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"))
    }
    pd.DataFrame(
        {
            "tokens": ["[" + " ".join(repr(t) for t in s.tokens) + "]" for s in rows],
            "ner_tags": ["[" + " ".join(str(id_of[t]) for t in s.tags) + "]" for s in rows],
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("mode", ["probe", "finetune"])
def test_train_smoke(tmp_path, tiny_csv, mode):
    out = tmp_path / mode
    metrics = train.main(
        ["--data", str(tiny_csv), "--output-dir", str(out), "--mode", mode, "--epochs", "2", "--batch-size", "8",
         "--device", "cpu"],
        encoder=tiny_encoder(),
    )  # fmt: skip
    assert set(metrics["entity"]) == {"micro", "macro", "per_type"}
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["mode"] == mode and saved["best_epoch"] in (1, 2)
    assert (out / "best.pt").exists() and (out / "errors.jsonl").exists()
    assert len((out / "history.jsonl").read_text().splitlines()) == 2
