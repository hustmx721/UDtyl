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

        clf_loss = clf_loss_func(output, b_y.long())
        loss = clf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * b_x.size(0)
        total_samples += b_x.size(0)

        all_logits.append(output.detach())
        all_labels.append(b_y.detach())

    avg_loss = total_loss / total_samples

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)

    accuracy, f1, bca, eer = calculate_metrics(all_labels_cat, all_logits_cat)

    return avg_loss, accuracy, f1, bca, eer
