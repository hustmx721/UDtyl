import torch
import torch.nn as nn


def _compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute a confusion matrix on the current device."""

    indices = y_true * num_classes + y_pred
    conf_flat = torch.bincount(indices, minlength=num_classes * num_classes)
    return conf_flat.reshape(num_classes, num_classes)


def _calculate_eer(y_true_onehot: torch.Tensor, y_pred_probs: torch.Tensor) -> torch.Tensor:
    """Calculate mean EER across classes using GPU tensors."""

    n_classes = y_pred_probs.shape[1]
    class_eers = []

    for i in range(n_classes):
        targets = y_true_onehot[:, i]
        scores = y_pred_probs[:, i]

        pos_total = targets.sum()
        neg_total = targets.numel() - pos_total

        if pos_total == 0 or neg_total == 0:
            class_eers.append(torch.tensor(float("nan"), device=targets.device))
            continue

        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_targets = targets[indices]

        tps = torch.cumsum(sorted_targets, dim=0)
        fps = torch.cumsum(1 - sorted_targets, dim=0)

        tpr = tps / pos_total
        fpr = fps / neg_total
        fnr = 1 - tpr

        diff = torch.abs(fpr - fnr)
        diff = torch.nan_to_num(diff, nan=float("inf"))
        idx = torch.argmin(diff)
        eer = (fpr[idx] + fnr[idx]) / 2
        class_eers.append(eer)

    eers_tensor = torch.stack(class_eers)
    return torch.nanmean(eers_tensor)


def calculate_metrics(y_true: torch.Tensor, y_logits: torch.Tensor):
    """Calculate accuracy, weighted F1, balanced accuracy, and EER on GPU tensors."""

    y_true = y_true.to(torch.int64)
    y_logits = y_logits.to(torch.float32)
    y_pred = y_logits.argmax(dim=1)

    num_classes = y_logits.shape[1]
    total_samples = y_true.numel()

    confusion = _compute_confusion_matrix(y_true, y_pred, num_classes)
    true_positives = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    accuracy = true_positives.sum() / max(total_samples, 1)

    precision = true_positives / predicted.clamp(min=1)
    recall = true_positives / support.clamp(min=1)
    f1_per_class = 2 * precision * recall / (precision + recall).clamp(min=torch.finfo(torch.float32).eps)

    weights = support / support.sum().clamp(min=1)
    f1_weighted = (weights * f1_per_class).sum()

    valid_recall = recall[support > 0]
    bca = valid_recall.mean() if valid_recall.numel() > 0 else torch.tensor(float("nan"), device=y_true.device)

    y_true_onehot = torch.zeros((total_samples, num_classes), device=y_logits.device, dtype=torch.float32)
    y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1.0)
    y_pred_probs = torch.softmax(y_logits, dim=1)

    eer = _calculate_eer(y_true_onehot, y_pred_probs)

    return accuracy, f1_weighted, bca, eer


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
            logits = model(x)
            loss = clf_loss_func(logits, y.long())

            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    avg_loss = total_loss / len(dataloader)

    all_logits_cat = torch.cat(all_logits, dim=0).to(torch.float32)
    all_labels_cat = torch.cat(all_labels, dim=0)

    accuracy, f1, bca, eer = calculate_metrics(all_labels_cat, all_logits_cat)

    return avg_loss, accuracy.item(), f1.item(), bca.item(), eer.item()
