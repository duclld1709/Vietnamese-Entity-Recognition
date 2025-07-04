import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.predict import predict_demo
from src.front import render_html
from results.output import training_log, report_dict, report_dict_2, model_compare, data_compare

st.set_page_config(page_title="Vietnamese NER", layout="wide")

# ===== Main Title =====
st.title("🔍 Vietnamese Named Entity Recognition (NER) Application")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "📈 Training Results", "🧪 Model Demo"])

# --- Tab 1: DATA ANALYSIS ---
with tab1:
    col1, col2 = st.columns(2)

    # ==== Distribution of NER Label Frequency ====
    with col1:
        st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/ner_freq.png",
                 caption="NER Label Frequency Distribution")

    # ==== Distribution of NER Label Frequency (Add crawled data) ====
    with col2:
        st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/ner_freq_add.png",
                 caption="NER Label Frequency (Extended with Crawled Data)")

    # ==== Distribution of the Number of Entities per Sentence (0 to 15+) ====
    with col1:
        st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/ent_dis.png",
                 caption="Number of Entities per Sentence")

    # ==== Distribution of Sentence Lengths ====
    with col2:
        st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/sent_len.png",
                 caption="Sentence Length Distribution")

    # ==== Distribution of Token Lengths ====
    with col1:
        st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/token_len.png",
                 caption="Token Length Distribution")

# --- Tab 2: TRAINING RESULTS ---
with tab2:
    st.set_page_config(
        page_title="Vietnamese NER",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ==== CREATE FIGURES ====

    # 1️⃣ Loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["train_loss"],
                                  mode='lines+markers', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["val_loss"],
                                  mode='lines+markers', name='Validation Loss'))
    fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")

    # 2️⃣ F1-Score
    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["train_f1"],
                                mode='lines+markers', name='Train F1'))
    fig_f1.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["val_f1"],
                                mode='lines+markers', name='Validation F1'))
    fig_f1.update_layout(title="F1-Score Curve", xaxis_title="Epoch", yaxis_title="F1-Score")

    # 3️⃣ Classification Report Table & Bar
    labels = [k for k in report_dict.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    report_data = [[lbl,
                    report_dict[lbl]["precision"],
                    report_dict[lbl]["recall"],
                    report_dict[lbl]["f1-score"]]
                   for lbl in labels]
    df_report = pd.DataFrame(report_data,
                             columns=["Label", "Precision", "Recall", "F1-Score"])

    fig_report = go.Figure()
    for col in ["Precision", "Recall", "F1-Score"]:
        fig_report.add_trace(go.Bar(x=df_report["Label"], y=df_report[col], name=col))
    fig_report.update_layout(barmode='group',
                             title="Class Metrics: PhoBERT + CRF",
                             xaxis_title="Label", yaxis_title="Score",
                             yaxis=dict(range=[0, 1.0]))

    labels2 = [k for k in report_dict_2.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    report_data2 = [[lbl,
                     report_dict_2[lbl]["precision"],
                     report_dict_2[lbl]["recall"],
                     report_dict_2[lbl]["f1-score"]]
                    for lbl in labels2]
    df_report2 = pd.DataFrame(report_data2,
                              columns=["Label", "Precision", "Recall", "F1-Score"])

    fig_report2 = go.Figure()
    for col in ["Precision", "Recall", "F1-Score"]:
        fig_report2.add_trace(go.Bar(x=df_report2["Label"], y=df_report2[col], name=col))
    fig_report2.update_layout(barmode='group',
                              title="Class Metrics: PhoBERT + Softmax",
                              xaxis_title="Label", yaxis_title="Score",
                              yaxis=dict(range=[0, 1.0]))

    # 4️⃣ Model & Data Comparison Tables
    df_model = pd.DataFrame(
        [[m, v["F1"], v["Accuracy"]] for m, v in model_compare["Data"].items()],
        columns=["Model", "F1-Score", "Accuracy"]
    )
    df_data = pd.DataFrame(
        [[s, f1] for s, f1 in data_compare["Data"].items()],
        columns=["Preprocessing", "F1-Score"]
    )

    # ==== CLEAN LAYOUT WITH COLUMNS ====

    # Row 1: Loss | F1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_loss, use_container_width=True)
    with col2:
        st.plotly_chart(fig_f1, use_container_width=True)

    # Row 2: Class Report Table | Bar Chart
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_report2, use_container_width=True)
    with col4:
        st.plotly_chart(fig_report, use_container_width=True)

    # Row 3: Model Compare | Data Compare
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Model Comparison**")
        st.dataframe(df_model, use_container_width=True)
    with col6:
        st.markdown("**Data Preprocessing Comparison**")
        st.dataframe(df_data, use_container_width=True)

# --- Tab 3: MODEL DEMO ---
with tab3:
    st.header("🧪 Vietnamese Named Entity Recognition Demo")

    text = st.text_input("Enter Vietnamese text:", "Nguyễn Văn A đang làm việc tại Hà Nội")

    if st.button("Analyze"):
        if not text.strip():
            return jsonify({'error': 'No text provided.'}), 400
        tokens, labels = predict_demo(text)
        print("Tokens:", tokens)
        print("Labels:", labels)
        html_result = render_html(tokens, labels)
        print("HTML Result:", html_result)
        return jsonify({'tokens': tokens, 'labels': labels, 'html_result': html_result})
    except Exception as e:
        print("Exception:", e)
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
