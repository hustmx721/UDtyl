import os
import sys
import time
import gc
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import *  # noqa: F401,F403
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, set_args, load_all, load_data
from utils.Logging import Logger

from evaluate import evaluate
from train import train_one_epoch

warnings.filterwarnings("ignore")


def run_classification(
    trainloader,
    valloader,
    savepath: str,
    args,
    prefix: str,
) -> torch.nn.Module:
    """Shared training loop for both task and UID classification."""
    print("-" * 20 + "开始训练!" + "-" * 20)

    model, optimizer, device = load_all(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    best_epoch = 0
    best_acc = 0
    train_acc_all = []
    train_f1_all = []
    train_bca_all = []
    train_eer_all = []
    val_acc_all = []
    val_f1_all = []
    val_bca_all = []
    val_eer_all = []

    for epoch in tqdm(range(args.epoch), desc="Training:"):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
        )

        train_acc_all.append(train_acc)
        train_f1_all.append(train_f1)
        train_bca_all.append(train_bca)
        train_eer_all.append(train_eer)

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model,
            dataloader=valloader,
            args=args,
        )

        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(savepath, f"{prefix}_{args.model}_{args.seed}.pth"),
            )

        val_acc_all.append(val_acc)
        val_f1_all.append(val_f1)
        val_bca_all.append(val_bca)
        val_eer_all.append(val_eer)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{train_loss:.6f}\tVal_loss:{val_loss:.6f}"
            )
            print(
                f"  Train_F1:{train_f1:.4f}, BCA:{train_bca:.4f}, EER:{train_eer:.4f} | Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}"
            )

    print("-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model


def log_and_eval(
    model: torch.nn.Module,
    testloader,
    args,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    test_loss, test_acc, test_f1, test_bca, test_eer = evaluate(
        model, testloader, args, device
    )
    print(
        f"测试集平均指标为  Acc:{test_acc * 100:.2f}%;  F1:{test_f1 * 100:.2f}%;  BCA:{test_bca * 100:.2f}%; EER:{test_eer * 100:.2f}%;"
    )
    print("=====================test are done===================")
    return test_loss, test_acc, test_f1, test_bca, test_eer


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


def run_experiment(args, device: torch.device, is_task: bool):
    prefix = "Task" if is_task else "UID"
    args.is_task = is_task
    args = set_args(args)
    seeds = list(range(args.seed, args.seed + args.repeats))
    results = np.zeros((len(seeds), 4))

    print(f"========== 开始{prefix}分类 ==========")

    for idx, seed in enumerate(seeds):
        args.seed = seed
        args.is_task = is_task
        args = set_args(args)
        start_time = time.time()
        print("=" * 30)
        print(f"dataset: {args.dataset}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        print(f"is_task: {args.is_task}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        model_path = args.model_root / f"{args.dataset}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model = run_classification(trainloader, valloader, model_path, args, prefix)
        print("=====================model are trained===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        _, test_acc, test_f1, test_bca, test_eer = log_and_eval(
            model, testloader, args, device
        )

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        summarize_results(results, seeds, idx, prefix)
        print(
            f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(f"========== {prefix}分类完成 ==========")
    print(model)
    save_results_csv(results, args, prefix, seeds)


def main():
    args = init_args()
    args = set_args(args)
    apply_thread_limits(getattr(args, "torch_threads", 4))
    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")

    log_path = args.log_root / f"{args.dataset}_joint_{args.model}.log"
    sys.stdout = Logger(log_path)

    base_seed = args.seed
    for is_task in (True, False):
        args.seed = base_seed
        run_experiment(args, device, is_task)


if __name__ == "__main__":
    main()
