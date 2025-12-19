import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize


def calculate_eer(y_true, y_pred):
    n_classes = y_true.shape[1]
    class_eer = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        fnr = 1 - tpr
        if len(fpr) > 0 and not np.all(np.isnan(fpr)) and not np.all(np.isnan(fnr)):
            idx = np.nanargmin(np.abs(fnr - fpr))
            eer = (fpr[idx] + fnr[idx]) / 2.0
        else:
            eer = np.nan
        class_eer.append(eer)

    avg_eer = np.nanmean(class_eer)
    return avg_eer, class_eer


def calculate_metrics(labels: torch.Tensor, logits: torch.Tensor):
    if labels.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().float()
    accuracy = correct / labels.numel()

    num_classes = logits.shape[1]
    labels_one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(logits.dtype)
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).to(logits.dtype)

    true_positives = (labels_one_hot * preds_one_hot).sum(dim=0)
    support = labels_one_hot.sum(dim=0)
    predicted_support = preds_one_hot.sum(dim=0)
    false_positives = predicted_support - true_positives
    false_negatives = support - true_positives

    precision_denom = true_positives + false_positives
    recall_denom = true_positives + false_negatives

    precision = torch.where(
        precision_denom > 0, true_positives / precision_denom, torch.zeros_like(true_positives)
    )
    recall = torch.where(
        recall_denom > 0, true_positives / recall_denom, torch.zeros_like(true_positives)
    )

    f1_denom = precision + recall
    f1_per_class = torch.where(
        f1_denom > 0, 2 * precision * recall / f1_denom, torch.zeros_like(f1_denom)
    )

    total_support = support.sum()
    if total_support.item() > 0:
        f1_weighted = (f1_per_class * support).sum() / total_support
    else:
        f1_weighted = torch.tensor(float("nan"), device=logits.device)

    valid_recalls = recall[support > 0]
    if valid_recalls.numel() > 0:
        bca = valid_recalls.mean()
    else:
        bca = torch.tensor(float("nan"), device=logits.device)

    probabilities = torch.softmax(logits, dim=1)
    labels_cpu = labels.detach().cpu().numpy()
    probabilities_cpu = probabilities.detach().cpu().numpy()

    classes = np.arange(logits.shape[1])
    y_true_binarized = label_binarize(labels_cpu, classes=classes)
    if y_true_binarized.ndim == 1:
        y_true_binarized = np.column_stack([1 - y_true_binarized, y_true_binarized])

    if y_true_binarized.shape[1] < 2 or probabilities_cpu.shape[1] < 2:
        eer = float("nan")
    else:
        try:
            eer, _ = calculate_eer(y_true_binarized, probabilities_cpu)
        except Exception as e:
            print(f"Warning: Could not calculate EER due to: {e}. Setting EER to NaN.")
            eer = float("nan")

    return accuracy.item(), f1_weighted.item(), bca.item(), eer


def evaluate(model, dataloader, args=None, device=None):
    model.eval()
    total_loss = 0.0
    total_samples = len(dataloader.dataset)

    if total_samples == 0:
        print("Warning: Evaluation dataloader is empty.")
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    if device is None:
        if args is not None:
            device = torch.device(
                "cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu"
            )
        else:
            device = next(model.parameters()).device

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            y = y.long()
            logits = model(x)
            loss = clf_loss_func(logits, y)

            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    avg_loss = total_loss / len(dataloader)

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)

    accuracy, f1, bca, eer = calculate_metrics(all_labels_cat, all_logits_cat)

    return avg_loss, accuracy, f1, bca, eer
