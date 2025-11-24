import torch
import warnings
from evaluate import calculate_metrics

warnings.filterwarnings("ignore")


def train_one_epoch(model, dataloader, device, optimizer, clf_loss_func):
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_labels = []

    for batch_idx, (b_x, b_y) in enumerate(dataloader):
        b_x, b_y = b_x.to(device), b_y.to(device)

        output = model(b_x)
        pred_y = torch.argmax(output, dim=1)

        clf_loss = clf_loss_func(output, b_y.long())
        loss = clf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * b_x.size(0)
        total_samples += b_x.size(0)

        all_logits.append(output.detach().cpu())
        all_labels.append(b_y.cpu())

    avg_loss = total_loss / total_samples

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)

    y_true = all_labels_cat.numpy()
    y_logits = all_logits_cat.numpy()
    y_pred = all_logits_cat.argmax(axis=1).numpy()

    accuracy, f1, bca, eer = calculate_metrics(y_true, y_logits)

    return avg_loss, accuracy, f1, bca, eer
