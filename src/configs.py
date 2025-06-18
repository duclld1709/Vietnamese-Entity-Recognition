configs = {   
    # Init
    "project": "NER",
    "name": "CRF_VLSP2016_Ultra",
    "model": "Linear/CRF",

    # Hyperparameters
    "optim": "Adam",
    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 20,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15
}