import torch

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
