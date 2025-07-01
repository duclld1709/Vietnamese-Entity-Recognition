import streamlit as st
import pandas as pd
from src.predict import predict_demo
from src.front import render_html

st.set_page_config(page_title="Vietnamese NER", layout="wide")

# ===== Tiêu đề chính =====
st.title("🔍 Ứng dụng nhận diện thực thể có tên (NER) cho tiếng Việt")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Phân tích dữ liệu", "📈 Kết quả huấn luyện", "🧪 Demo mô hình"])

# --- Tab 1: PHÂN TÍCH DỮ LIỆU ---
with tab1:
    st.header("📊 Phân tích dữ liệu")

    df = pd.DataFrame({
        "Loại thực thể": ["PER", "LOC", "ORG", "MISC"],
        "Số lượng": [3200, 2500, 1800, 900]
    })
    
    st.bar_chart(df.set_index("Loại thực thể"))

# --- Tab 2: KẾT QUẢ HUẤN LUYỆN ---
with tab2:
    st.header("📈 Kết quả huấn luyện")

    loss = [0.9, 0.7, 0.5, 0.35, 0.28]
    epoch = [1, 2, 3, 4, 5]
    df_loss = pd.DataFrame({"Epoch": epoch, "Loss": loss})
    st.line_chart(df_loss.set_index("Epoch"))

    st.subheader("Đánh giá mô hình")
    df_eval = pd.DataFrame({
        "Phiên bản": ["v1", "v2", "v3"],
        "F1-score": [0.78, 0.83, 0.86],
        "Accuracy": [0.81, 0.85, 0.88]
    })
    st.dataframe(df_eval)

# --- Tab 3: DEMO MÔ HÌNH ---
with tab3:
    st.header("🧪 Vietnamese Named Entity Recognition")

    text = st.text_input("Nhập văn bản tiếng Việt:", "Nguyễn Văn A đang làm việc tại Hà Nội")

    if st.button("Phân tích"):
        if not text.strip():
            st.warning("Vui lòng nhập văn bản!")
        else:
            tokens, labels = predict_demo(text)

            st.subheader("Thực thể được phát hiện")
            entities = [(tok, lab) for tok, lab in zip(tokens, labels) if lab != "O"]

            if entities:
                for tok, lab in entities:
                    st.markdown(f"🔹 **{tok}** — *{lab}*")
            else:
                st.info("Không phát hiện thực thể.")

        st.subheader("Highlight trong văn bản:")
        st.markdown(render_html(tokens, labels), unsafe_allow_html=True)
