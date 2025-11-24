import os
import sys
import time
import gc
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baselines.LLock_Chen.LearnabilityLock import LinearLock, iResLock
from utils.dataset import set_seed
from utils.init_all import init_args, set_args, load_all, load_data
from utils.Logging import Logger
from evaluate import calculate_metrics

warnings.filterwarnings("ignore")


def build_lock_params(trainloader, args) -> dict:
    sample_batch = next(iter(trainloader))
    sample_x = sample_batch[0]
    if not torch.is_tensor(sample_x):
        sample_x = torch.tensor(sample_x)

    sample_shape = sample_x.shape
    if len(sample_shape) < 3:
        raise ValueError(
            f"LLock requires at least 3D input (batch, channel, ...). Got shape {sample_shape}."
        )

    n_channel = sample_shape[1]
    input_shape = sample_shape[2:]
    in_shape = input_shape[-1] if len(input_shape) > 0 else args.fs

    lock_params = {
        "n_class": args.nclass,
        "n_channel": n_channel,
        "input_shape": input_shape,
        "in_shape": in_shape,
    }

    if args.lock_type == "ires":
        lock_params["mid_planes"] = args.lock_mid_planes

    return lock_params


def create_lock(args, lock_params, device: torch.device):
    if args.lock_type == "linear":
        return LinearLock(
            epsilon=args.lock_epsilon, lock_params=lock_params, device=device
        )
    return iResLock(
        epsilon=args.lock_epsilon,
        lock_params=lock_params,
        device=device,
        sname=f"LLock_{args.model}_{args.dataset}",
    )


def train_lock_and_model(trainloader, lock, model, args, device: torch.device):
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initlr)
    lock.train(
        base_model=model,
        loader=trainloader,
        opt_base=optimizer,
        loss_base=clf_loss_func,
        learning_rate=args.initlr,
    )


def evaluate_with_lock(model, lock, dataloader, device: torch.device):
    model.eval()
    clf_loss_func = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)

            locked_x = lock.lock(x, y).to(device)
            logits = model(locked_x)
            loss = clf_loss_func(logits, y.long())

            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    avg_loss = total_loss / len(dataloader)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    acc, f1, bca, eer = calculate_metrics(labels_cat, logits_cat)
    return avg_loss, acc, f1, bca, eer


def summarize_results(results: np.ndarray, seeds: List[int], idx: int, prefix: str):
    row_labels = [str(seed) for seed in seeds] + ["Avg", "Std"]
    col_labels = ["Acc", "F1", "BCA", "EER"]
    print(f"{prefix}结果汇总（前{idx + 1}轮）")
    print(
        f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}"
    )

    for i in range(idx + 1):
        row = results[i]
        print(
            f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}"
        )

    print(
        f"{row_labels[-2]:<10} {np.mean(results[:idx + 1, 0]):<10.4f} {np.mean(results[:idx + 1, 1]):<10.4f} {np.mean(results[:idx + 1, 2]):<10.4f} {np.mean(results[:idx + 1, 3]):<10.4f}"
    )
    print(
        f"{row_labels[-1]:<10} {np.std(results[:idx + 1, 0]):<10.4f} {np.std(results[:idx + 1, 1]):<10.4f} {np.std(results[:idx + 1, 2]):<10.4f} {np.std(results[:idx + 1, 3]):<10.4f}"
    )


def save_results_csv(results: np.ndarray, args, prefix: str, seeds: List[int]):
    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(
        final_results,
        columns=["Acc", "F1", "BCA", "EER"],
        index=[*(str(seed) for seed in seeds), "Avg", "Std"],
    ).round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"{prefix}_{args.model}.csv")


def main():
    args = init_args()
    args = set_args(args)
    device = torch.device(
        "cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu"
    )

    seeds = list(range(args.seed, args.seed + args.repeats))
    results = np.zeros((len(seeds), 4))

    log_path = args.log_root / f"{args.dataset}_LLock_{args.model}.log"
    sys.stdout = Logger(log_path)

    for idx, seed in enumerate(seeds):
        args.seed = seed
        start_time = time.time()
        print("=" * 30)
        print(f"dataset: {args.dataset}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        print(f"is_task: {args.is_task}")
        print(f"lock   : {args.lock_type}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        lock_params = build_lock_params(trainloader, args)
        print(f"锁参数: {lock_params}")

        model, optimizer, device = load_all(args)
        torch.cuda.empty_cache()
        if device.type == "cuda":
            torch.cuda.set_device(device)

        lock = create_lock(args, lock_params, device)
        train_lock_and_model(trainloader, lock, model, args, device)
        print("=====================LLock training done===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate_with_lock(
            model, lock, valloader, device
        )
        print(
            f"验证集平均指标为  Acc:{val_acc * 100:.2f}%;  F1:{val_f1 * 100:.2f}%;  BCA:{val_bca * 100:.2f}%; EER:{val_eer * 100:.2f}%;"
        )

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate_with_lock(
            model, lock, testloader, device
        )
        print(
            f"测试集平均指标为  Acc:{test_acc * 100:.2f}%;  F1:{test_f1 * 100:.2f}%;  BCA:{test_bca * 100:.2f}%; EER:{test_eer * 100:.2f}%;"
        )
        print("=====================test are done===================")

        model_path = args.model_root / f"{args.dataset}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(
            model.state_dict(),
            os.path.join(model_path, f"LLock_{args.model}_{args.seed}.pth"),
        )
        lock.save(sname=f"LLock_{args.model}_{args.seed}", path=model_path)

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        summarize_results(results, seeds, idx, "LLock")
        print(
            f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 50)
    print(model)

    save_results_csv(results, args, "LLock", seeds)


if __name__ == "__main__":
    main()
