from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score
import numpy as np

def compute_metrics(labels, preds, scores):
    """
    labels: (T, N)
    preds: (T, N)
    scores: (T, N)
    """
    y_true = labels.flatten()
    y_pred = preds.flatten()
    y_score = scores.flatten()
    
    # Ignore perfectly clean data if computing AUC and only 1 class is present
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)
    else:
        auc_roc = float('nan')
        auc_pr = float('nan')
        
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # For binary classification where 1 is attack and 0 is benign
    dr = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'Macro-F1': macro_f1,
        'Detection Rate (DR)': dr,
        'False Alarm Rate (FAR)': far,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr
    }

def print_metrics(metrics_dict):
    print("-" * 30)
    print("Evaluation Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)
