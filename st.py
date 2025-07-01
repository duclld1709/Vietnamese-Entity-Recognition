import streamlit as st

col1, col2 = st.columns(2)

# ==== Distribution of NER Label Frequency ====
with col1:
    st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/ner_freq.png")

# ==== Distribution of NER Label Frequency (Add crawled data) ====
with col2:
    st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/ner_freq_add.png")

# ==== Distribution of the Number of Entities per Sentence (0 to 15+) ====
with col1:
    st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/ent_dis.png")

# ==== Distribution of Sentence Lengths ====
with col2:
    st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/sent_len.png")

# ==== Distribution of Token Lengths ====
with col1:
    st.image("https://raw.githubusercontent.com/duclld1709/vietnamese-ner/refs/heads/main/results/token_len.png")
