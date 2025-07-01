import streamlit as st
from src.predict import predict_demo
from src.front import render_html

st.title("Vietnamese Named Entity Recognition")

text = st.text_input("Nhập văn bản tiếng Việt:", "Nguyễn Văn A đang làm việc tại Hà Nội")

if st.button("Phân tích"):
    if not text.strip():
        st.warning("Vui lòng nhập văn bản!")
    else:
        tokens, labels = predict_demo(text)

        st.subheader(" Thực thể được phát hiện")
        entities = [ (tok, lab) for tok, lab in zip(tokens, labels) if lab != "O"]

        if entities:
            for tok, lab in entities:
                st.markdown(f"🔹 **{tok}** — *{lab}*")
        
        else:
            st.info("Không phát hiện thực thể.")
    
    st.subheader("Highlight trong văn bản:")
    st.markdown(render_html(tokens, labels), unsafe_allow_html=True)