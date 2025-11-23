import os
import sys
import time
import gc
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from gene_handi_UD import RandTemplate, SNTemplate, STFTRandTemplate
from utils.dataset import set_seed
from utils.init_all import init_args, set_args, load_all, load_data
from utils.Logging import Logger

warnings.filterwarnings("ignore")


def build_template(trainloader, args, device: torch.device):
    """Create a handcrafted UD template using the first batch statistics."""
    first_batch = next(iter(trainloader))
    sample = first_batch[0].to(device)
    bath_size, _, channels, timesteps = sample.shape
    channel_std = sample.std(dim=(0, 2, 3))

    if args.handi_method == "rand":
        return RandTemplate(channels, timesteps, args.handi_alpha, channel_std, str(device))
    if args.handi_method == "sn":
        return SNTemplate(channels, timesteps, args.handi_alpha, channel_std, str(device))
    return STFTRandTemplate(channels, n_fft=256, alpha=args.handi_alpha, per_channel_std=channel_std, device=str(device))


def apply_template(x: torch.Tensor, user_ids: torch.Tensor, template) -> torch.Tensor:
    if template is None:
        return x
    return template.apply(x, user_ids)


def train_one_epoch_with_template(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    optimizer,
    clf_loss_func,
    template,
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_labels = []

    for batch in dataloader:
        if len(batch) == 3:
            b_x, b_y, user_ids = batch
        else:
            b_x, b_y = batch
            user_ids = b_y
        b_x = apply_template(b_x.to(device), user_ids.to(device), template)
        b_y = b_y.to(device)

        output = model(b_x)
        clf_loss = clf_loss_func(output, b_y.long())

        optimizer.zero_grad()
        clf_loss.backward()
        optimizer.step()

        total_loss += clf_loss.item() * b_x.size(0)
        total_samples += b_x.size(0)

        all_logits.append(output.detach().cpu())
        all_labels.append(b_y.cpu())

    avg_loss = total_loss / max(total_samples, 1)
    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)

    y_true = all_labels_cat.numpy()
    y_logits = all_logits_cat.numpy()
    y_pred = all_logits_cat.argmax(axis=1).numpy()

    from evaluate import calculate_metrics  # Local import to reuse existing metrics

    accuracy, f1, bca, eer = calculate_metrics(y_true, y_pred, y_logits)
    return avg_loss, accuracy, f1, bca, eer


def evaluate_with_template(model, dataloader, device, template):
    model.eval()
    clf_loss_func = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, user_ids = batch
            else:
                x, y = batch
                user_ids = y

            x = apply_template(x.to(device), user_ids.to(device), template)
            y = y.to(device)

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

    from evaluate import calculate_metrics

    accuracy, f1, bca, eer = calculate_metrics(y_true, y_pred, y_logits)
    return avg_loss, accuracy, f1, bca, eer


def run_classification(
    trainloader,
    valloader,
    savepath: str,
    args,
    prefix: str,
) -> torch.nn.Module:
    """Training loop using handcrafted UD templates."""
    print("-" * 20 + "开始训练!" + "-" * 20)

    model, optimizer, device = load_all(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)
    template = build_template(trainloader, args, device)

    best_epoch = 0
    best_acc = 0
    train_acc_all: List[float] = []
    train_f1_all: List[float] = []
    train_bca_all: List[float] = []
    train_eer_all: List[float] = []
    val_acc_all: List[float] = []
    val_f1_all: List[float] = []
    val_bca_all: List[float] = []
    val_eer_all: List[float] = []

    for epoch in tqdm(range(args.epoch), desc="Training:"):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch_with_template(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
            template=template,
        )

        train_acc_all.append(train_acc)
        train_f1_all.append(train_f1)
        train_bca_all.append(train_bca)
        train_eer_all.append(train_eer)

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate_with_template(
            model=model,
            dataloader=valloader,
            device=device,
            template=template,
        )

        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(savepath, f"Handi_{prefix}_{args.handi_method}_{args.model}_{args.seed}.pth"),
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
    template,
) -> Tuple[float, float, float, float]:
    test_loss, test_acc, test_f1, test_bca, test_eer = evaluate_with_template(
        model, testloader, device, template
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
    df.to_csv(csv_path / f"Handi_{prefix}_{args.handi_method}_{args.model}.csv")


def run_experiment(args, device: torch.device, is_task: bool):
    prefix = "Task" if is_task else "UID"
    seeds = list(range(args.seed, args.seed + args.repeats))
    results = np.zeros((len(seeds), 4))

    print(f"========== 开始{prefix}分类（handcrafted UD） ==========")

    for idx, seed in enumerate(seeds):
        args.seed = seed
        args.is_task = is_task
        start_time = time.time()
        print("=" * 30)
        print(f"dataset: {args.dataset}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        print(f"is_task: {args.is_task}")
        print(f"handi_method: {args.handi_method}, handi_alpha: {args.handi_alpha}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args, include_index=True)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        model_path = args.model_root / f"{args.dataset}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model = run_classification(trainloader, valloader, model_path, args, prefix)
        print("=====================model are trained===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        # Rebuild template to keep dataloader iterator alignment deterministic during testing
        template = build_template(trainloader, args, device)
        _, test_acc, test_f1, test_bca, test_eer = log_and_eval(
            model, testloader, args, device, template
        )

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        summarize_results(results, seeds, idx, prefix)
        print(
            f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}"
        )
        gc.collect()

    print(f"========== {prefix}分类完成 ==========")
    print(model)
    save_results_csv(results, args, prefix, seeds)


def main():
    args = init_args()
    args = set_args(args)
    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")

    log_path = args.log_root / f"Handi_{args.handi_method}_{args.dataset}_{args.model}.log"
    sys.stdout = Logger(log_path)

    base_seed = args.seed
    for is_task in (True, False):
        args.seed = base_seed
        run_experiment(args, device, is_task)


if __name__ == "__main__":
    main()
