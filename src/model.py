from torchcrf import CRF
import torch.nn as nn

class CRF_Tagger(nn.Module):
    def __init__(self, input_dim, num_tags):
        super().__init__()
        self.embed2tag = nn.Linear(input_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, x, labels, mask):
        emissions = self.embed2tag(x)
        return -self.crf(emissions, labels, mask=mask, reduction="mean")
    
    def decode(self, x, mask=None):
        emissions = self.embed2tag(x)
        return self.crf.decode(emissions, mask)