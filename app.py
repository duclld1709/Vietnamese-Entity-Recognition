"""Streamlit demo (also the Hugging Face Space entry point): ``streamlit run app.py``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from vietner import VietNER, deanonymize
from vietner.anonymize import DEFAULT_LABELS, STRATEGIES
from vietner.render import render_html

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"
# The Space deploy strips PNGs (no Git LFS), so fall back to the copies on GitHub.
FIGURE_URL = "https://raw.githubusercontent.com/duclld1709/Vietnamese-Entity-Recognition/main/docs/figures/{}.png"
FIGURE_NAMES = ["ner_freq", "ner_freq_add", "ent_dis", "sent_len", "token_len"]

EXAMPLE = (
    "Ông Nguyễn Văn Nam, giám đốc Công ty Cổ phần FPT, đang công tác tại Hà Nội. "
    "Liên hệ SĐT 0912.345.678 hoặc email nam.nguyen@fpt.com.vn. "
    "Số CCCD: 001090012345, STK 0071 0012 34567 tại Vietcombank, sinh ngày 12/05/1990."
)

st.set_page_config(page_title="Vietnamese NER & PII", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading PhoBERT-CRF…")
def load_model() -> VietNER:
    checkpoint = os.environ.get("VIETNER_CHECKPOINT")
    return VietNER.from_checkpoint(checkpoint) if checkpoint else VietNER.from_checkpoint()


ner = load_model()
st.title("🛡️ Vietnamese NER & Personal-Data Anonymiser")
st.caption(
    "PhoBERT-CRF finds people, organisations and places. Validated rules find phone numbers, ID numbers, emails, "
    "bank accounts and more. Anonymise documents before sending them to an LLM, then restore the names in its answer."
)
if not ner.ner_available:
    st.warning(f"NER model unavailable, running rules only: {ner.unavailable_reason}")

tab_ner, tab_anon, tab_eval, tab_data = st.tabs(["🔍 Entities", "🛡️ Anonymise", "📊 Evaluation", "📈 Dataset"])

with tab_ner:
    text = st.text_area("Vietnamese text", EXAMPLE, height=140, key="ner_text")
    c1, c2 = st.columns(2)
    use_model = c1.checkbox("Model entities (PER/ORG/LOC)", value=True)
    use_rules = c2.checkbox("Rule-based PII (phone, ID, email, …)", value=True)
    if st.button("Analyse", type="primary") and text.strip():
        result = ner.analyze([text], ner=use_model, rules=use_rules)[0]
        st.markdown(render_html(result.text, result.entities), unsafe_allow_html=True)
        if result.entities:
            st.dataframe(
                pd.DataFrame([e.model_dump() for e in result.entities]), use_container_width=True, hide_index=True
            )
        else:
            st.info("No entities found.")

with tab_anon:
    text = st.text_area("Document", EXAMPLE, height=140, key="anon_text")
    c1, c2 = st.columns([1, 2])
    strategy = c1.radio(
        "Strategy",
        STRATEGIES,
        format_func={"tag": "Placeholders (reversible)", "mask": "Partial mask", "redact": "Redact"}.get,
    )
    labels = c2.multiselect("Labels to anonymise", sorted(DEFAULT_LABELS | {"MISC"}), sorted(DEFAULT_LABELS))
    if st.button("Anonymise", type="primary") and text.strip():
        st.session_state["anon"] = ner.anonymize(text, strategy, labels)
    result = st.session_state.get("anon")
    if result is not None:
        for w in result.warnings:
            if not w.startswith("NER model unavailable"):  # already shown in the banner
                st.warning(w)
        st.code(result.text, language=None, wrap_lines=True)
        st.download_button("Download anonymised text", result.text, "anonymised.txt")
        if result.mapping:
            with st.expander(f"Mapping ({len(result.mapping)} placeholders) - keep it private"):
                st.json(result.mapping)
            st.download_button(
                "Download mapping (JSON)", json.dumps(result.mapping, ensure_ascii=False, indent=2), "mapping.json"
            )
            st.subheader("Restore an LLM answer")
            reply = st.text_area("Paste text that contains the placeholders", key="llm_reply")
            if reply.strip():
                st.code(deanonymize(reply, result.mapping), language=None, wrap_lines=True)

with tab_eval:
    st.markdown(
        "**Entity-level** F1 counts an entity as correct only if its span *and* type match exactly "
        "(CoNLL/VLSP standard). v1 reported **token-level macro F1 including `O`**, which is inflated by "
        "the ~94% of tokens that are `O`. Results come from `python -m training.evaluate`."
    )
    runs = sorted(REPORTS.glob("*/metrics.json"))
    if not runs:
        st.info("No evaluation run yet. Run `python -m training.evaluate --checkpoint models/best_epoch_16.pt "
                "--output-dir reports/v1_probe` to populate this tab.")  # fmt: skip
    for path in runs:
        m = json.loads(path.read_text())
        st.subheader(path.parent.name)
        micro = m["entity"]["micro"]
        c = st.columns(4)
        c[0].metric("Entity F1 (micro)", f"{micro['f1']:.3f}")
        c[1].metric("Precision", f"{micro['precision']:.3f}")
        c[2].metric("Recall", f"{micro['recall']:.3f}")
        c[3].metric("v1 token metric", f"{m['token_v1_metric']['f1']:.3f}")
        st.dataframe(pd.DataFrame(m["entity"]["per_type"]).T.round(3), use_container_width=True)
        st.caption(f"Errors: {m.get('errors')}")
    v1 = REPORTS / "v1_reported.json"
    if v1.exists():
        with st.expander("Numbers reported by v1 (token-level, for reference)"):
            st.json(json.loads(v1.read_text()))

with tab_data:
    st.markdown("VLSP 2016 NER (news), 16,858 sentences. MISC is mapped to `O` in the 7-tag scheme.")
    cols = st.columns(2)
    for i, name in enumerate(FIGURE_NAMES):
        local = FIGURES / f"{name}.png"
        cols[i % 2].image(str(local) if local.exists() else FIGURE_URL.format(name), caption=name.replace("_", " "))
