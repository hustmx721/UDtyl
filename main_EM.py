import os
import sys
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import set_seed
from utils.init_all import init_args, set_args, load_all, load_data
from utils.Logging import Logger

from evaluate import evaluate


def compute_soft_labels(logits: torch.Tensor, temperature: float, smoothing: float) -> torch.Tensor:
    scaled_logits = logits / max(temperature, 1e-6)
    soft_targets = torch.softmax(scaled_logits, dim=1)
    if smoothing > 0:
        num_classes = soft_targets.size(1)
        soft_targets = (1 - smoothing) * soft_targets + smoothing / num_classes
    return soft_targets


def refine_soft_labels(logits: torch.Tensor, initial_soft: torch.Tensor, steps: int, lr: float, entropy_reg: float) -> torch.Tensor:
    """
    Inner minimization (min over q): optimize soft labels w.r.t. negative log-likelihood
    with an optional entropy regularizer. The model logits are treated as constants
    so this realizes the min-min structure from EM_Huang (q-update then theta-update).
    """
    if steps <= 0:
        return initial_soft.detach()

    soft = initial_soft.clone().detach().requires_grad_(True)
    log_probs = torch.log_softmax(logits.detach(), dim=1)

    for _ in range(steps):
        loss = -(soft * log_probs).sum(dim=1).mean()
        if entropy_reg > 0:
            entropy = -(soft * torch.log(soft.clamp_min(1e-12))).sum(dim=1).mean()
            loss = loss + entropy_reg * entropy

        grad = torch.autograd.grad(loss, soft, retain_graph=False)[0]
        with torch.no_grad():
            soft -= lr * grad
            soft.clamp_(min=1e-8)
            soft /= soft.sum(dim=1, keepdim=True)

        soft.requires_grad_(True)

    return soft.detach()


def e_step(model, dataloader, device, temperature, smoothing, inner_steps, inner_lr, entropy_reg):
    model.eval()

    cached_soft_labels = []
    total_samples = 0
    total_log_likelihood = 0.0

    with torch.no_grad():
        for b_x, _ in dataloader:
            b_x = b_x.to(device)
            logits = model(b_x)
            base_soft = compute_soft_labels(logits, temperature, smoothing)

            if inner_steps > 0:
                refined_soft = refine_soft_labels(logits, base_soft, inner_steps, inner_lr, entropy_reg)
            else:
                refined_soft = base_soft.detach()

            log_probs = torch.log_softmax(logits.detach() / max(temperature, 1e-6), dim=1)
            batch_ll = (refined_soft * log_probs).sum(dim=1)

            cached_soft_labels.append(refined_soft.cpu())
            total_log_likelihood += batch_ll.sum().item()
            total_samples += b_x.size(0)

    avg_log_likelihood = total_log_likelihood / max(total_samples, 1)
    cached_soft_labels = torch.cat(cached_soft_labels, dim=0) if cached_soft_labels else torch.empty(0)
    return avg_log_likelihood, cached_soft_labels


def m_step(model, dataloader, device, optimizer, soft_labels):
    return soft_targets.detach()


def em_one_epoch(model, dataloader, device, optimizer, temperature, smoothing):
    model.train()

    total_loss = 0.0
    total_samples = 0
    offset = 0

    for b_x, _ in dataloader:
        bsz = b_x.size(0)
        targets = soft_labels[offset:offset + bsz].to(device)
        offset += bsz

        b_x = b_x.to(device)
        logits = model(b_x)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
    total_log_likelihood = 0.0
    cached_soft_labels = []

    for batch_idx, (b_x, _) in enumerate(dataloader):
        b_x = b_x.to(device)

        logits = model(b_x)
        soft_targets = compute_soft_labels(logits, temperature, smoothing)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bsz
        total_samples += bsz

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def build_consistent_loader(trainloader):
    return DataLoader(
        dataset=trainloader.dataset,
        batch_size=trainloader.batch_size,
        shuffle=False,
        num_workers=trainloader.num_workers,
        pin_memory=trainloader.pin_memory,
        drop_last=False,
    )
        batch_size = b_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_log_likelihood += (-loss.item()) * batch_size
        cached_soft_labels.append(soft_targets.cpu())

    avg_loss = total_loss / max(total_samples, 1)
    avg_log_likelihood = total_log_likelihood / max(total_samples, 1)
    cached_soft_labels = torch.cat(cached_soft_labels, dim=0) if cached_soft_labels else torch.empty(0)

    return avg_loss, avg_log_likelihood, cached_soft_labels


def EMTrain(trainloader, valloader, savepath, args):
    print("-" * 20 + "开始 EM 训练!" + "-" * 20)

    model, optimizer, device = load_all(args)
    if args.em_init_model:
        init_path = os.path.expanduser(args.em_init_model)
        if os.path.isfile(init_path):
            state_dict = torch.load(init_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded initial model weights from {init_path}")
        else:
            print(f"Warning: init model path {init_path} not found, training from scratch.")

    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    best_epoch = 0
    best_acc = 0
    prev_log_likelihood = None

    val_acc_all = []

    em_loader = build_consistent_loader(trainloader)

    for epoch in tqdm(range(args.em_iters), desc="EM Training:"):
        # E-step: min over q (soft labels)
        em_ll, cached_soft_labels = e_step(
            model=model,
            dataloader=em_loader,
            device=device,
            temperature=args.em_temperature,
            smoothing=args.em_label_smoothing,
            inner_steps=args.em_inner_steps,
            inner_lr=args.em_inner_lr,
            entropy_reg=args.em_entropy_reg,
        )

        # M-step: min over theta (model params)
        em_loss = m_step(
            model=model,
            dataloader=em_loader,
            device=device,
            optimizer=optimizer,
            soft_labels=cached_soft_labels,
        )

    train_acc_all = []
    train_f1_all = []
    train_bca_all = []
    train_eer_all = []
    val_acc_all = []
    val_f1_all = []
    val_bca_all = []
    val_eer_all = []
    loss_item_train = []
    loss_item_val = []

    prev_log_likelihood = None

    for epoch in tqdm(range(args.em_iters), desc="EM Training:"):
        em_loss, em_ll, _ = em_one_epoch(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            temperature=args.em_temperature,
            smoothing=args.em_label_smoothing,
        )

        loss_item_train.append(em_loss)

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model,
            dataloader=valloader,
            args=args,
        )

        val_acc_all.append(val_acc)
        val_f1_all.append(val_f1)
        val_bca_all.append(val_bca)
        val_eer_all.append(val_eer)
        loss_item_val.append(val_loss)

        # Early stopping based on validation accuracy
        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(savepath, f"EM_{args.model}_{args.seed}.pth"))

        # Convergence check based on training log-likelihood
        if prev_log_likelihood is not None:
            delta_ll = abs(em_ll - prev_log_likelihood)
            if delta_ll < args.em_threshold:
                print(f"EM converged at epoch {epoch+1} with ΔLL={delta_ll:.6f} < {args.em_threshold}")
                break
        prev_log_likelihood = em_ll

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch:{epoch+1}\tTrain_negLL:{em_loss:.6f}\tVal_acc:{val_acc:.4f}\tVal_loss:{val_loss:.6f}\tAvg_LL:{em_ll:.6f}")
            print(f"  Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}")

    # Reload best checkpoint if it was saved
    best_ckpt = os.path.join(savepath, f"EM_{args.model}_{args.seed}.pth")
    if os.path.isfile(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        # Estimate training metrics using cached soft labels
        if len(trainloader.dataset) > 0:
            train_acc_all.append(val_acc)
            train_f1_all.append(val_f1)
            train_bca_all.append(val_bca)
            train_eer_all.append(val_eer)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch:{epoch+1}\tTrain_negLL:{em_loss:.6f}\tVal_acc:{val_acc:.4f}\tVal_loss:{val_loss:.6f}")
            print(f"  Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}")

    print("-" * 20 + "EM 训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model


def main():
    args = init_args()
    # Extend EM-specific defaults after parsing
    for field, default in [
        ("em_iters", 100),
        ("em_threshold", 1e-4),
        ("em_temperature", 1.0),
        ("em_label_smoothing", 0.0),
        ("em_init_model", None),
        ("em_inner_steps", 5),
        ("em_inner_lr", 0.1),
        ("em_entropy_reg", 0.0),
    ]:
        if not hasattr(args, field):
            setattr(args, field, default)
    if not hasattr(args, "em_iters"):
        args.em_iters = 100
    if not hasattr(args, "em_threshold"):
        args.em_threshold = 1e-4
    if not hasattr(args, "em_temperature"):
        args.em_temperature = 1.0
    if not hasattr(args, "em_label_smoothing"):
        args.em_label_smoothing = 0.0
    if not hasattr(args, "em_init_model"):
        args.em_init_model = None

    args = set_args(args)
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    results = np.zeros((5, 4))

    log_path = args.log_root / f"{args.dataset}_EM_{args.model}.log"
    sys.stdout = Logger(log_path)

    for idx, seed in enumerate(range(args.seed, args.seed + args.repeats)):
        args.seed = seed
        args.is_task = True
        start_time = time.time()
        print("=" * 30)
        print(f"dataset: {args.dataset}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        print(f"is_task: {args.is_task}")
        print(f"em_iters: {args.em_iters}, em_threshold: {args.em_threshold}")
        print(f"temperature: {args.em_temperature}, smoothing: {args.em_label_smoothing}")
        print(f"inner_steps: {args.em_inner_steps}, inner_lr: {args.em_inner_lr}, entropy_reg: {args.em_entropy_reg}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        model_path = args.model_root / f"{args.dataset}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model = EMTrain(trainloader, valloader, model_path, args)
        print("=====================model are trained===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate(model, testloader, args, device)

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        print(
            f"测试集平均指标为  Acc:{test_acc * 100:.2f}%;  F1:{test_f1 * 100:.2f}%;  BCA:{test_bca * 100:.2f}%; EER:{test_eer * 100:.2f}%;")
        print("=====================test are done===================")

        row_labels = ['2024', '2025', '2026', '2027', '2028', "Avg", "Std"]
        col_labels = ['Acc', 'F1', 'BCA', 'EER']
        print(
            f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
        for i, row in enumerate(results):
            print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
        print(f"{row_labels[-2]:<10} {np.mean(results[:idx + 1, 0]):<10.4f} {np.mean(results[:idx + 1, 1]):<10.4f} {np.mean(results[:idx + 1, 2]):<10.4f} {np.mean(results[:idx + 1, 3]):<10.4f}")
        print(f"{row_labels[-1]:<10} {np.std(results[:idx + 1, 0]):<10.4f} {np.std(results[:idx + 1, 1]):<10.4f} {np.std(results[:idx + 1, 2]):<10.4f} {np.std(results[:idx + 1, 3]):<10.4f}")
        gc.collect()

    print("-" * 50)
    print(model)

    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(final_results,
                      columns=['Acc', 'F1', 'BCA', 'EER'],
                      index=['2024', '2025', '2026', '2027', '2028', "Avg", "Std"])
    df = df.round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"EM_{args.model}.csv")


if __name__ == "__main__":
    main()
    main()
