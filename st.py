import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from results.output import training_log, report_dict, report_dict_2, model_compare, data_compare

st.set_page_config(
    page_title="My NER App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== TẠO FIGURES ====

# 1️⃣ Loss
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["train_loss"],
                              mode='lines+markers', name='Train Loss'))
fig_loss.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["val_loss"],
                              mode='lines+markers', name='Val Loss'))
fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")

# 2️⃣ F1-Score
fig_f1 = go.Figure()
fig_f1.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["train_f1"],
                            mode='lines+markers', name='Train F1'))
fig_f1.add_trace(go.Scatter(x=training_log["epoch"], y=training_log["val_f1"],
                            mode='lines+markers', name='Val F1'))
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
                         title="Class Report Metrics of PhoBert + CRF",
                         xaxis_title="Label", yaxis_title="Score",
                         yaxis=dict(range=[0,1.0]))

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
                          title="Class Report Metrics of PhoBert + Softmax",
                          xaxis_title="Label", yaxis_title="Score",
                          yaxis=dict(range=[0,1.0]))

# 4️⃣ Model & Data Comparison Tables
df_model = pd.DataFrame(
    [[m, v["F1"], v["Accuracy"]] for m, v in model_compare["Data"].items()],
    columns=["Model", "F1-Score", "Accuracy"]
)
df_data = pd.DataFrame(
    [[s, f1] for s, f1 in data_compare["Data"].items()],
    columns=["Preprocessing", "F1-Score"]
)

# ==== LAYOUT RAO GỌN VỚI COLUMNS ====

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
