# Model Results
training_log = {
    "epoch": list(range(1, 21)),
    "train_loss": [
        2.6912, 0.8061, 0.5842, 0.4782, 0.4056,
        0.3599, 0.3218, 0.2942, 0.2699, 0.2517,
        0.2383, 0.2223, 0.2127, 0.2026, 0.1925,
        0.1863, 0.1795, 0.1728, 0.1673, 0.1640
    ],
    "val_loss": [
        1.0848, 0.7191, 0.5643, 0.4838, 0.4281,
        0.3934, 0.3751, 0.3560, 0.3521, 0.3413,
        0.3292, 0.3305, 0.3244, 0.3213, 0.3392,
        0.3169, 0.3187, 0.3219, 0.3261, 0.3230
    ],
    "train_f1": [
        0.8224, 0.8674, 0.8996, 0.9122, 0.9254,
        0.9343, 0.9383, 0.9424, 0.9429, 0.9493,
        0.9551, 0.9543, 0.9593, 0.9609, 0.9574,
        0.9654, 0.9677, 0.9692, 0.9681, 0.9715
    ],
    "val_f1": [
        0.8273, 0.8613, 0.8895, 0.8994, 0.9101,
        0.9190, 0.9192, 0.9189, 0.9177, 0.9222,
        0.9232, 0.9207, 0.9221, 0.9224, 0.9117,
        0.9250, 0.9237, 0.9173, 0.9195, 0.9185
    ]
}

report_dict = {
    'O': {"precision": 1.00, "recall": 1.00, "f1-score": 1.00, "support": 51036},
    'B-PER': {"precision": 0.99, "recall": 0.98, "f1-score": 0.98, "support": 1112},
    'I-PER': {"precision": 0.97, "recall": 0.99, "f1-score": 0.98, "support": 506},
    'B-ORG': {"precision": 0.83, "recall": 0.84, "f1-score": 0.84, "support": 180},
    'I-ORG': {"precision": 0.88, "recall": 0.84, "f1-score": 0.86, "support": 291},
    'B-LOC': {"precision": 0.93, "recall": 0.95, "f1-score": 0.94, "support": 939},
    'I-LOC': {"precision": 0.93, "recall": 0.91, "f1-score": 0.92, "support": 428},
    "accuracy": 0.99,
    "macro avg": {"precision": 0.93, "recall": 0.93, "f1-score": 0.93, "support": 54492},
    "weighted avg": {"precision": 0.99, "recall": 0.99, "f1-score": 0.99, "support": 54492}
}


report_dict_2 = {
    'O': {"precision": 1.00, "recall": 1.00, "f1-score": 1.00, "support": 68476},
    'B-PER': {"precision": 0.99, "recall": 0.98, "f1-score": 0.98, "support": 1464},
    'I-PER': {"precision": 0.98, "recall": 0.98, "f1-score": 0.98, "support": 686},
    'B-ORG': {"precision": 0.77, "recall": 0.82, "f1-score": 0.80, "support": 257},
    'I-ORG': {"precision": 0.80, "recall": 0.77, "f1-score": 0.78, "support": 430},
    'B-LOC': {"precision": 0.88, "recall": 0.90, "f1-score": 0.89, "support": 1241},
    'I-LOC': {"precision": 0.83, "recall": 0.82, "f1-score": 0.82, "support": 554},
    "accuracy": 0.99,
    "macro avg": {"precision": 0.89, "recall": 0.89, "f1-score": 0.89, "support": 73108},
    "weighted avg": {"precision": 0.99, "recall": 0.99, "f1-score": 0.99, "support": 73108}
}


model_compare = {
    "Header": ["Model", "F1", "Accuracy"],
    "Data": {
        "PhoBERT + CRF": {"F1": 0.93, "Accuracy": 0.99},
        "CRF": {"F1": 0.91, "Accuracy": 0.99},
        "Softmax": {"F1": 0.89, "Accuracy": 0.99},
        "Random Forest": {"F1": 0.78, "Accuracy": 0.98}
    }
}

data_compare = {
    "Header": ["Data Preprocessing Strategy", "F1"],
    "Data": {
        "Raw": 0.93,
        "Crawl for Balance": 0.91,
        "Remove Sentences with Only 'O' Tags": 0.91
    }
}



# EDA 
data_aug_count_sorted = {
    'B-PER': 474,
    'I-PER': 121,
    'B-LOC': 874,
    'I-LOC': 289,
    'B-ORG': 1110,
    'I-ORG': 761
}

raw_data_count_sorted = {
    'B-PER': 7479,
    'I-PER': 3522,
    'B-LOC': 6244,
    'I-LOC': 2783,
    'B-ORG': 1212,
    'I-ORG': 2055,
    'B-NAT': 282,
    'I-NAT': 279
}

raw_data_count_withoutNAT_sorted = {
    'B-PER': 7479,
    'I-PER': 3522,
    'B-LOC': 6244,
    'I-LOC': 2783,
    'B-ORG': 1212,
    'I-ORG': 2055
}

combined_count_sorted = {
    'B-PER': 7953,
    'I-PER': 3643,
    'B-LOC': 7118,
    'I-LOC': 3072,
    'B-ORG': 2322,
    'I-ORG': 2816
}
