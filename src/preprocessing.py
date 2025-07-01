import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.configs import configs
from pyvi import ViTokenizer

def join_tokens(tokens):
    text = ' '.join(tokens)
    return text

def reform_raw_text(tokens):
    text = ' '.join(tokens)
    return text.replace("_", " ")

def label(x, ):
    id_tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}
    return [id_tag[int(i)] for i in x]

def replace_7_8(lst):
    return [0 if x in (7, 8) else x for x in lst]

# Hàm gộp các embedding vectors của token bị tách ra khi qua SentencePiece
def group_embeddings(tokens, embeddings):
    word_embeddings = []
    current_vecs = []

    for token, emb in zip(tokens, embeddings):
        if token in ["<s>", "</s>"]:
            continue

        if token.endswith("@@"):
            current_vecs.append(emb)
        else:
            current_vecs.append(emb)
            word_emb = torch.mean(torch.stack(current_vecs), dim=0)
            word_embeddings.append(word_emb)
            current_vecs = []

    if current_vecs:  # Trong trường hợp sót lại cuối câu
        word_emb = torch.mean(torch.stack(current_vecs), dim=0)
        word_embeddings.append(word_emb)

    return word_embeddings


# Download the dataset form Hugging Face
def download_raw_data():
    splits = {'train': 'data/train-00000-of-00001-b0417886a268b83a.parquet', 'valid': 'data/valid-00000-of-00001-846411c236133ba3.parquet'}
    df_train = pd.read_parquet("hf://datasets/datnth1709/VLSP2016-NER-data/" + splits["train"])
    df_valid = pd.read_parquet("hf://datasets/datnth1709/VLSP2016-NER-data/" + splits["valid"])
    df = pd.concat([df_train, df_valid]).reset_index(drop=True)

    return df

# Process dataframe for EDA
def preprocess_data_for_EDA(df):
    # Define tag - id
    tag_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
    id_tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

    # Add columns and remove inappropriate tags
    df['ner_tags'] = df['ner_tags'].apply(replace_7_8)
    df['text_withseg'] = df['tokens'].apply(join_tokens)
    df['text_raw'] = df['tokens'].apply(reform_raw_text)
    df["ner_labels"] = df.ner_tags.apply(label)
    df.columns = ['tokens', 'id_labels', 'seg_text', 'raw_text', 'labels']

    return df




def load_phoBERT_model_and_tokenizer():
    # Load PhoBERT tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model = AutoModel.from_pretrained("vinai/phobert-base")
    model.eval()
    return model, tokenizer


# Embedding text
def create_embeddings(df, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings = []  # list of [seq_len_i, 768] tensors
    all_labels = [] # list of [seq_len_i,] tensors
    remove_index = []

    for i, row in tqdm(df.iterrows(), total=len(df)):

        # Truy cập phần tử từng dòng
        sentence = row['seg_text']
        gold_labels = row["id_labels"]

        # Cho sentence đi qua SentencePiece
        input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        # Encode tạo embeddings
        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_state = outputs.last_hidden_state.squeeze(0).cpu()

        # Gộp các embeddings đã bị tách khi đi qua SentencePiece
        word_embeds = group_embeddings(tokens, last_hidden_state)

        # Kiểm tra số lượng embeddings và số lượng labels, nếu conflict -> xóa dòng đó
        if len(word_embeds) != len(gold_labels):
            # print(f"Warning: Skip row {i} - length mismatch")
            remove_index.append(i)
            continue

        # Thêm vào list tổng & Tới đây là data đã sẵn sàng cho training
        all_embeddings.append(torch.stack(word_embeds))
        all_labels.append(torch.tensor(gold_labels))
    
        # Create Dict
        processed_data = {
          "embeddings": all_embeddings,
          "labels": all_labels
        }

    return processed_data


def split_dataset(data):

    # Train_Val / Test Split
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(data["embeddings"], data["labels"], test_size=configs["test_ratio"], random_state=42)

    # Train / Val Split
    val_rest_ratio = configs["val_ratio"] / (configs["val_ratio"] + configs["train_ratio"])
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = val_rest_ratio, random_state=42)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# TODO: Refactor hàm process_demo_sentence, và hàm predict demo, warning nếu độ dài tokens_word không bằng độ dài sau group_embeddings

def process_demo_sentence(text):
    """
    Trả về tensor shape 1 x Seq_length x 768
    """
    segmented_text = ViTokenizer.tokenize(text)
    tokens_word = segmented_text.strip().split(" ")

    model, tokenizer = load_phoBERT_model_and_tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(segmented_text, return_tensors="pt").to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_state = outputs.last_hidden_state.squeeze(0).cpu()
    
    word_embeds = group_embeddings(tokens, last_hidden_state)

    all_embeddings = torch.stack(word_embeds) # seq_length x 768

    all_embeddings = all_embeddings.unsqueeze(0) # Thêm chiều batch size là 1 -> 1 x seq_length x 768

    return all_embeddings, tokens_word

