"""Headless run of the Streamlit demo (rules-only, so no PhoBERT download)."""

import pytest

from tests.conftest import ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("VIETNER_CHECKPOINT", str(ROOT / "missing-checkpoint.pt"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_starts_in_rules_only_mode(app):
    assert any("rules only" in w.value for w in app.warning)


def test_entities_tab(app):
    app.button[0].click().run()
    assert not app.exception
    html = "".join(m.value for m in app.markdown)
    assert "PHONE" in html and "0912.345.678" in html


def test_anonymise_tab(app):
    app.button[1].click().run()
    assert not app.exception
    anonymised = app.code[0].value
    assert "[PHONE_1]" in anonymised and "0912.345.678" not in anonymised
