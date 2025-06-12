from src.preprocessing import *

def main():

    # Download VLSP2016 from hgface
    df = download_raw_data()

    # Save raw data
    df.to_csv(r".\data\raw_data.csv", index=False)

    # Process data for EDA
    df = preprocess_data_for_EDA(df)
    df.to_csv(r".\data\processed_data_EDA.csv", index=False)

    # Init PhoBERT Tokenizer and PhoBERT Model
    model, tokenizer = load_phoBERT_model_and_tokenizer()

    # Add feature embedding vectors
    df = add_embeddings_feature(df, model, tokenizer)
    df.to_csv(r".\data\processed_data_full.csv", index=False)


if __name__ == "__main__":
    main()
