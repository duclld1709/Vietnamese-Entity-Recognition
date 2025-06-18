from src.preprocessing import download_raw_data, preprocess_data_for_EDA, load_phoBERT_model_and_tokenizer, create_embeddings, split_dataset
from src.data_set import NerDataset, collate_fn
from src.configs import configs
from src.model import CRF_Tagger
from src.train import train_model

import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")


def main():

    # Download VLSP2016 from hgface
    print("Download raw data ...")
    df = download_raw_data()

    # Save raw data
    df.to_csv(r".\data\raw_data.csv", index=False)
    print("Save at data\raw_data.csv \n")

    # Process data for EDA
    print("Process data for EDA ...")
    df = preprocess_data_for_EDA(df)
    df.to_csv(r".\data\processed_data_EDA.csv", index=False)
    print("Save at data\processed_data_EDA.csv \n")
    
    # Init PhoBERT Tokenizer and PhoBERT Model
    print("Embedding data ...")
    model, tokenizer = load_phoBERT_model_and_tokenizer()

    # Embeddings data
    processed_data = create_embeddings(df, model, tokenizer)
    torch.save(processed_data, r".\data\processed_data_full.pt")
    print("Save at data\processed_data_full.pt \n")

    # Split data into train/valid/test
    print("Train/Valid/Test Split ...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(processed_data)
    print("Done \n")

    # Data Agumentation for training set
    # Pass

    # Init DataLoader 
    print("Init DataLoader ...")
    datasets = {
        'train': NerDataset(X_train, Y_train),
        'val': NerDataset(X_val, Y_val),
        'test': NerDataset(X_test, Y_test)
    }

    loaders = {
        split: DataLoader(dataset, batch_size=configs["batch_size"], shuffle=(split=='train'), collate_fn=collate_fn)
        for split, dataset in datasets.items()
    }
    print("Done \n")

    # Init sequence label model
    print("Init Model ...")
    NUM_TAGS = 7
    model = CRF_Tagger(input_dim=X_train[0].size(1), num_tags=NUM_TAGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["learning_rate"])
    print("Done \n")

    # Training Model
    print("Start training ...")
    train_model(model, optimizer, configs, loaders)

if __name__ == "__main__":
    main()
