import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import label_binarize

def calculate_eer(y_true, y_pred):
    n_classes = y_true.shape[1]
    class_eer = []
    
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
        fnr = 1 - tpr
        if len(fpr) > 0 and not np.all(np.isnan(fpr)) and not np.all(np.isnan(fnr)):
             idx = np.nanargmin(np.abs(fnr - fpr))
             eer = (fpr[idx] + fnr[idx]) / 2.0
        else:
             eer = np.nan 
        class_eer.append(eer)
    
    avg_eer = np.nanmean(class_eer)
    return avg_eer, class_eer

def calculate_metrics(y_true, y_pred, y_logits):
    accuracy = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    bca = balanced_accuracy_score(y_true, y_pred)

    unique_labels = np.unique(y_true)
    y_true = label_binarize(y_true, classes=unique_labels)
    if len(unique_labels) == 2 and y_true.ndim == 1:
        y_true = np.column_stack([1 - y_true, y_true])
    y_pred = torch.softmax(torch.from_numpy(y_logits), dim=1).numpy()
    
    try:
        eer, _ = calculate_eer(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Could not calculate EER due to: {e}. Setting EER to NaN.")
        eer = float('nan')

    return accuracy, f1, bca, eer

def evaluate(model, dataloader, args=None, device=None):
    model.eval()
    total_loss = 0.0
    total_samples = len(dataloader.dataset)

    if total_samples == 0:
        print("Warning: Evaluation dataloader is empty.")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    if device is None:
        if args is not None:
            device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
        else:
            device = next(model.parameters()).device

    clf_loss_func = nn.CrossEntropyLoss().to(device) 

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = clf_loss_func(logits, y.long())

            total_loss += loss.item() 
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / len(dataloader)

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)

    y_true = all_labels_cat.numpy()
    y_logits = all_logits_cat.numpy()
    y_pred = all_logits_cat.argmax(axis=1).numpy()

    accuracy, f1, bca, eer = calculate_metrics(y_true, y_pred, y_logits)

    return avg_loss, accuracy, f1, bca, eer