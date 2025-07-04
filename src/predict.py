import torch
from model import CRF_Tagger
from preprocessing import process_demo_sentence
def predict(model, loader, count_loss=True):
    
    model.eval() # Evaluation Mode, Ignore Dropout, BatchNorm, ...
    all_preds, all_true = [], []
    loss = 0.0

    with torch.no_grad(): # Stop track gradient
        for x, y, _ in loader:
            mask = (y != -1)

            # Get loss
            if count_loss:
                loss += model(x, y, mask).item()
            
            # Get prediction
            preds = model.decode(x, mask)

            # Loop for each sentence in mini-batch
            for pred_seq, true_seq, m in zip(preds, y, mask):
                true_labels = true_seq[m].tolist() # tensor[mask tensor boolean]
                all_preds.extend(pred_seq)
                all_true.extend(true_labels)
    
    return all_preds, all_true, loss/len(loader)

def predict_demo(text):


    id_tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

    x, tokens = process_demo_sentence(text) # 1 x seq_length x 768
    NUM_TAGS = 7

    model = CRF_Tagger(input_dim=x.size(2), num_tags=NUM_TAGS)
    model.load_state_dict(torch.load("models/best_epoch_16.pt"))
    model.eval()
    with torch.no_grad():
        preds = model.decode(x)
    
    labels = [id_tag[lab] for lab in preds[0]] # preds[0] vì sẽ trả về nhiều batch nhưng chúng ta chỉ có 1

    return tokens, labels
