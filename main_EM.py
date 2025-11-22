import os
import sys
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy

from utils.dataset import set_seed
from utils.init_all import init_args, set_args, load_all, load_data
from utils.Logging import Logger

from evaluate import evaluate


def clamp_tensor(x: torch.Tensor, clip_min, clip_max):
    if clip_min is not None and clip_max is not None:
        return x.clamp(clip_min, clip_max)
    if clip_min is not None:
        return x.clamp(min=clip_min)
    if clip_max is not None:
        return x.clamp(max=clip_max)
    return x


def init_classwise_noise(trainloader, n_classes: int, eps: float, device: torch.device):
    first_batch = next(iter(trainloader))
    x = first_batch[0]
    x_shape = x.shape[1:]
    delta = torch.zeros((n_classes,) + x_shape, device=device)
    delta = clamp_tensor(delta, -eps, eps)
    return delta


def train_one_epoch_with_noise(model, trainloader, delta, optimizer, device, eps, clip_min=None, clip_max=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in trainloader:
        b_x, b_y = batch[:2]
        b_x = b_x.to(device)
        b_y = b_y.to(device).long()

        pert = delta[b_y]
        x_adv = clamp_tensor(b_x + pert, clip_min, clip_max)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * b_x.size(0)
        total_samples += b_x.size(0)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == b_y).sum().item()

    train_err = 1.0 - correct / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    return train_err, avg_loss


def update_classwise_noise(model, trainloader, delta, eps, alpha, pgd_steps, device, clip_min=None, clip_max=None):
    model.eval()
    for batch in trainloader:
        b_x, b_y = batch[:2]
        b_x = b_x.to(device)
        b_y = b_y.to(device).long()

        for cls in b_y.unique():
            cls_mask = b_y == cls
            x_cls = b_x[cls_mask]
            y_cls = b_y[cls_mask]
            if x_cls.numel() == 0:
                continue

            x_adv = clamp_tensor((x_cls + delta[cls]).detach(), clip_min, clip_max)
            x_adv.requires_grad_(True)

            for _ in range(pgd_steps):
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y_cls)
                grad = torch.autograd.grad(loss, x_adv)[0]

                with torch.no_grad():
                    x_adv = x_adv - alpha * grad.sign()
                    perturb = torch.clamp(x_adv - x_cls, -eps, eps)
                    x_adv = clamp_tensor(x_cls + perturb, clip_min, clip_max)
                    x_adv.requires_grad_(True)

            with torch.no_grad():
                perturb = torch.clamp(x_adv - x_cls, -eps, eps)
                delta[cls] = clamp_tensor(perturb.mean(dim=0), -eps, eps)


def em_error_min_train(model, optimizer, trainloader, valloader, savepath, args, device, mode_tag: str, pretrained_delta=None):
    eps = args.em_eps
    alpha = args.em_alpha if args.em_alpha is not None else eps / 10.0
    outer_rounds = args.em_iters if args.em_iters is not None else args.em_outer

    if args.em_init_model:
        init_path = os.path.expanduser(args.em_init_model)
        if os.path.isfile(init_path):
            state_dict = torch.load(init_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded initial model weights from {init_path}")
        else:
            print(f"Warning: init model path {init_path} not found, training from scratch.")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_device(device)

    if pretrained_delta is not None:
        delta = clamp_tensor(pretrained_delta.to(device), -eps, eps)
        print(f"Loaded pretrained class-wise perturbations for {mode_tag} mode")
    else:
        delta = init_classwise_noise(trainloader, args.nclass, eps, device)
    best_acc = 0.0
    best_outer = -1

    for outer in tqdm(range(outer_rounds), desc="Error-Minimization"):
        for _ in range(args.em_theta_epochs):
            train_err, train_loss = train_one_epoch_with_noise(
                model=model,
                trainloader=trainloader,
                delta=delta,
                optimizer=optimizer,
                device=device,
                eps=eps,
                clip_min=args.em_clip_min,
                clip_max=args.em_clip_max,
            )

        update_classwise_noise(
            model=model,
            trainloader=trainloader,
            delta=delta,
            eps=eps,
            alpha=alpha,
            pgd_steps=args.em_pgd_steps,
            device=device,
            clip_min=args.em_clip_min,
            clip_max=args.em_clip_max,
        )

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model,
            dataloader=valloader,
            args=args,
        )

        print(
            f"[Outer {outer+1}] train_err={train_err:.4f}, train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
        )
        print(f"  Val_F1={val_f1:.4f}, BCA={val_bca:.4f}, EER={val_eer:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_outer = outer
            torch.save(model.state_dict(), os.path.join(savepath, f"EM_{mode_tag}_{args.model}_{args.seed}.pth"))
            torch.save(delta.cpu(), os.path.join(savepath, f"EM_delta_{mode_tag}_{args.model}_{args.seed}.pt"))

        if train_err < args.em_lambda:
            print(f"Stop since train_err={train_err:.4f} < λ={args.em_lambda}")
            break
        if (outer - best_outer) > args.earlystop:
            print(f"Early stopping triggered at outer round {outer+1}.")
            break

    best_ckpt = os.path.join(savepath, f"EM_{mode_tag}_{args.model}_{args.seed}.pth")
    if os.path.isfile(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    return model, delta



def run_mode(base_args, is_task: bool, results: np.ndarray, log_prefix: str, pretrained_deltas=None):
    collected_deltas = []
    for idx, seed in enumerate(range(base_args.seed, base_args.seed + base_args.repeats)):
        args = copy.deepcopy(base_args)
        args.seed = seed
        args.is_task = is_task
        args = set_args(args)
        start_time = time.time()
        mode_tag = "Task" if args.is_task else "UID"
        print("=" * 30)
        print(f"[{mode_tag}] dataset: {args.dataset}")
        print(f"[{mode_tag}] model  : {args.model}")
        print(f"[{mode_tag}] seed   : {args.seed}")
        print(f"[{mode_tag}] gpu    : {args.gpuid}")
        print(f"[{mode_tag}] nclass : {args.nclass}")
        print(
            f"[{mode_tag}] em_outer: {args.em_outer}, em_theta_epochs: {args.em_theta_epochs}, em_pgd_steps: {args.em_pgd_steps}, "
            f"em_eps: {args.em_eps}, em_alpha: {args.em_alpha if args.em_alpha is not None else args.em_eps/10.0}, em_lambda: {args.em_lambda}"
        )

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args, include_index=False)

        print(f"[{mode_tag}] =====================data are prepared===============")
        print(f"[{mode_tag}] 累计用时{time.time() - start_time:.4f}s!")

        model_path = args.model_root / f"{args.dataset}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model, optimizer, device = load_all(args)
        init_delta = None
        if args.is_task and pretrained_deltas is not None and idx < len(pretrained_deltas):
            init_delta = pretrained_deltas[idx]

        model, delta = em_error_min_train(
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            savepath=model_path,
            args=args,
            device=device,
            mode_tag=mode_tag,
            pretrained_delta=init_delta,
        )
        if not args.is_task:
            collected_deltas.append(delta.detach().cpu())
        print(f"[{mode_tag}] =====================model are trained===============")
        print(f"[{mode_tag}] 累计用时{time.time() - start_time:.4f}s!")

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate(model, testloader, args, device)

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        print(
            f"[{mode_tag}] 测试集平均指标为  Acc:{test_acc * 100:.2f}%;  F1:{test_f1 * 100:.2f}%;  BCA:{test_bca * 100:.2f}%; EER:{test_eer * 100:.2f}%;",
        )
        print(f"[{mode_tag}] =====================test are done===================")

        row_labels = ['2024', '2025', '2026', '2027', '2028', "Avg", "Std"]
        col_labels = ['Acc', 'F1', 'BCA', 'EER']
        print(f"[{mode_tag}] 训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
        for i, row in enumerate(results):
            print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
        print(f"{row_labels[-2]:<10} {np.mean(results[:idx + 1, 0]):<10.4f} {np.mean(results[:idx + 1, 1]):<10.4f} {np.mean(results[:idx + 1, 2]):<10.4f} {np.mean(results[:idx + 1, 3]):<10.4f}")
        print(f"{row_labels[-1]:<10} {np.std(results[:idx + 1, 0]):<10.4f} {np.std(results[:idx + 1, 1]):<10.4f} {np.std(results[:idx + 1, 2]):<10.4f} {np.std(results[:idx + 1, 3]):<10.4f}")
        gc.collect()

    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(final_results,
                      columns=['Acc', 'F1', 'BCA', 'EER'],
                      index=['2024', '2025', '2026', '2027', '2028', "Avg", "Std"])
    df = df.round(4)
    csv_path = base_args.csv_root / f"{base_args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"EM_{log_prefix}_{base_args.model}.csv")

    return collected_deltas


def main():
    args = init_args()
    args = set_args(args)
    log_path = args.log_root / f"EM_{args.dataset}_{args.model}.log"
    sys.stdout = Logger(log_path)

    task_results = np.zeros((5, 4))
    uid_results = np.zeros((5, 4))

    print("运行 UID EM 训练")
    uid_deltas = run_mode(args, False, uid_results, "UID")

    print("运行 Task EM 训练 (使用 UID 的 class-wise 噪声初始化)")
    run_mode(args, True, task_results, "Task", pretrained_deltas=uid_deltas)


if __name__ == '__main__':
    main()
