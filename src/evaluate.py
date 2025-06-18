from src.predict import predict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

def evaluate(model, loader, count_loss=True, report=False):

    # Model Preidction (Inference)
    all_preds, all_true, loss = predict(model, loader, count_loss)
    class_report = None
    
    # Get evaluation metric
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_true, all_preds)

    # Get classification report
    if report:
        class_report = classification_report(all_true, all_preds)

    return precision, recall, f1, acc, loss, class_report

def evaluate_ignore_O(model, loader):
    pass