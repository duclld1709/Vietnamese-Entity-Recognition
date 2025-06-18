from torch.utils.data import Dataset
import torch

class NerDataset(Dataset):
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def collate_fn(batch): # Batch_size x Seq_length x 768
    embeddings, labels = zip(*batch)
    lengths = [e.size(0) for e in embeddings]
    max_len = max(lengths)

    padded_embs = torch.stack([
        torch.cat([e, torch.zeros(max_len - e.size(0), e.size(1))]) for e in embeddings
    ])

    padded_labels = torch.stack([
        torch.cat([l, torch.full((max_len - l.size(0),), -1, dtype=torch.long)]) for l in labels
    ])
    
    return padded_embs, padded_labels, lengths
    
    