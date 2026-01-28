import os
import sys
import time
import gc
import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
warnings.filterwarnings("ignore")

import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from utils.data_loader import *
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, set_args, load_all, load_data
from utils.Logging import Logger

from Distill import EOTDistribution, IdentityTransform, STFTDeltaPerturber
from evaluate import calculate_metrics, evaluate
from train import train_one_epoch 

def format_lambda_tag(lambda_task: float, lambda_uid: float, lambda_reg: float) -> str:
    return f"lt{lambda_task}_lu{lambda_uid}_lr{lambda_reg}"


def build_eot_distribution(args) -> Optional[EOTDistribution]:
    if args.disable_eot:
        return None

    enabled = any(
        [
            args.eot_shift_prob > 0 and args.eot_shift > 0,
            args.eot_scale_prob > 0 and args.eot_scale,
            args.eot_channel_dropout_prob > 0 and args.eot_channel_dropout > 0,
            args.eot_resample_prob > 0 and args.eot_resample > 0,
        ]
    )
    if not enabled:
        return None

    return EOTDistribution(
        max_shift=args.eot_shift,
        shift_prob=args.eot_shift_prob,
        scale_low=args.eot_scale_min,
        scale_high=args.eot_scale_max,
        scale_prob=args.eot_scale_prob,
        channel_dropout=args.eot_channel_dropout,
        channel_dropout_prob=args.eot_channel_dropout_prob,
        resample_max_rate_delta=args.eot_resample,
        resample_prob=args.eot_resample_prob,
    )


def describe_eot(args) -> str:
    if args.disable_eot:
        return "eot_disabled"

    parts = []
    if args.eot_shift_prob > 0 and args.eot_shift > 0:
        parts.append(f"shift{args.eot_shift}_p{args.eot_shift_prob}")
    if args.eot_scale_prob > 0 and args.eot_scale:
        parts.append(
            f"scale{args.eot_scale_min}-{args.eot_scale_max}_p{args.eot_scale_prob}"
        )
    if args.eot_channel_dropout_prob > 0 and args.eot_channel_dropout > 0:
        parts.append(
            f"chdrop{args.eot_channel_dropout}_p{args.eot_channel_dropout_prob}"
        )
    if args.eot_resample_prob > 0 and args.eot_resample > 0:
        parts.append(f"resample{args.eot_resample}_p{args.eot_resample_prob}")

    if not parts:
        return "noeot"
    return "eot_" + "_".join(parts)


def _candidate_delta_paths(
    delta_root: Path,
    dataset: str,
    src_model: str,
    seed: int,
    tags: Sequence[str],
) -> Sequence[Path]:
    base_dir = delta_root / dataset / src_model
    return [base_dir / f"{tag}_seed{seed}.pth" for tag in tags]


def resolve_delta_path(args, seed: int) -> Path:
    if args.perturbation_path:
        return Path(args.perturbation_path)
    if not args.src_model:
        raise ValueError("Either --perturbation_path or --src_model must be provided for UD evaluation.")

    lambda_tag = format_lambda_tag(args.lambda_task, args.lambda_uid, args.lambda_reg)
    tag = f"{describe_eot(args)}_{lambda_tag}"
    candidates = _candidate_delta_paths(args.delta_root, args.dataset, args.src_model, seed, [tag])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError("Could not find perturbation checkpoint. Tried:\n" + joined)


def _load_perturber(path: Path, device: torch.device) -> STFTDeltaPerturber:
    if not path.is_file():
        raise FileNotFoundError(f"Perturbation checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "delta" in state:
        perturber = state["delta"]
    else:
        perturber = state

    if not isinstance(perturber, STFTDeltaPerturber):
        raise TypeError("Loaded object is not an STFTDeltaPerturber")

    perturber = perturber.to(device)
    perturber.eval()
    for p in perturber.parameters():
        p.requires_grad_(False)
    return perturber


def _sample_transform(
    eot_distribution: Optional[EOTDistribution], device: torch.device
) -> IdentityTransform:
    if eot_distribution is None:
        return IdentityTransform()
    return eot_distribution.sample(device=device)


@torch.no_grad()
def _apply_ud(
    x: torch.Tensor,
    perturber: STFTDeltaPerturber,
    transform: IdentityTransform,
) -> torch.Tensor:
    x_t_bar = transform.apply(x.squeeze(1))
    x_prime = perturber(x_t_bar.unsqueeze(1))
    x_ud = transform.apply(x_prime.squeeze(1)).unsqueeze(1)
    return x_ud


@torch.no_grad()
def evaluate_on_ud(
    model: nn.Module,
    dataloader,
    device: torch.device,
    perturber: STFTDeltaPerturber,
    eot_distribution: Optional[EOTDistribution],
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        transform = _sample_transform(eot_distribution, device)
        x_ud = _apply_ud(x, perturber, transform)

        logits = model(x_ud)
        loss = clf_loss_func(logits, y.long())

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        all_logits.append(logits.detach())
        all_labels.append(y.detach())

    avg_loss = total_loss / max(1, total_samples)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, f1, bca, eer = calculate_metrics(labels_cat, logits_cat)
    return avg_loss, acc, f1, bca, eer


def UIDClassify(trainloader, valloader, savepath, args):
    print("-" * 20 + "开始训练!" + "-" * 20)
    
    model, optimizer, device = load_all(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    # Loss functions
    clf_loss_func = nn.CrossEntropyLoss().to(device)

    # Training metrics
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
    loss_item_train = []
    loss_item_val = []

    for epoch in tqdm(range(args.epoch), desc="Training:"):
        # Run one training epoch using the imported function
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func
        )

        train_acc_all.append(train_acc)
        train_f1_all.append(train_f1)
        train_bca_all.append(train_bca)
        train_eer_all.append(train_eer)
        loss_item_train.append(train_loss)

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model,
            dataloader=valloader,
            args=args
        )

        # Early stopping logic based on validation accuracy
        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(savepath, f"UID_{args.model}_{args.seed}.pth"))

        val_acc_all.append(val_acc)
        val_f1_all.append(val_f1)
        val_bca_all.append(val_bca)
        val_eer_all.append(val_eer)
        loss_item_val.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{train_loss:.6f}\tVal_loss:{val_loss:.6f}")
            print(f"  Train_F1:{train_f1:.4f}, BCA:{train_bca:.4f}, EER:{train_eer:.4f} | Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}")

    print("-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model


def main():
    args = init_args()
    args = set_args(args)
    apply_thread_limits(getattr(args, "torch_threads", 4))
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    results = np.zeros((5, 4))
    ud_results = np.zeros((5, 4))

    log_path = args.log_root / f"{args.dataset}_UID_{args.model}.log"
    sys.stdout = Logger(log_path)

    for idx, seed in enumerate(range(args.seed, args.seed + args.repeats)):
        args.seed = seed
        args.is_task = False
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

        model = UIDClassify(trainloader, valloader, model_path, args)
        print("=====================model are trained===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")
        
        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate(model, testloader, args, device)

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        print(
            f"测试集平均指标为  Acc:{test_acc * 100:.2f}%;  F1:{test_f1 * 100:.2f}%;  BCA:{test_bca * 100:.2f}%; EER:{test_eer * 100:.2f}%;")
        print("=====================test are done===================")

        if args.perturbation_path or args.src_model:
            eot_distribution = build_eot_distribution(args)
            perturber_path = resolve_delta_path(args, seed)
            print(f"delta path : {perturber_path}")
            perturber = _load_perturber(perturber_path, device)
            ud_loss, ud_acc, ud_f1, ud_bca, ud_eer = evaluate_on_ud(
                model=model,
                dataloader=testloader,
                device=device,
                perturber=perturber,
                eot_distribution=eot_distribution,
            )
            ud_results[idx] = [ud_acc, ud_f1, ud_bca, ud_eer]
            print(
                f"扰动测试集平均指标为  Acc:{ud_acc * 100:.2f}%;  F1:{ud_f1 * 100:.2f}%;  "
                f"BCA:{ud_bca * 100:.2f}%; EER:{ud_eer * 100:.2f}%;"
            )

        row_labels = ['2024', '2025', '2026', '2027', '2028', "Avg", "Std"]
        col_labels = ['Acc', 'F1', 'BCA', 'EER'] 
        print(
            f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        # 打印列标签
        print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
        # 打印每一行数据，包括行标签
        for i, row in enumerate(results):
            print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
        print(f"{row_labels[-2]:<10} {np.mean(results[:idx + 1, 0]):<10.4f} {np.mean(results[:idx + 1, 1]):<10.4f} {np.mean(results[:idx + 1, 2]):<10.4f} {np.mean(results[:idx + 1, 3]):<10.4f}")
        print(f"{row_labels[-1]:<10} {np.std(results[:idx + 1, 0]):<10.4f} {np.std(results[:idx + 1, 1]):<10.4f} {np.std(results[:idx + 1, 2]):<10.4f} {np.std(results[:idx + 1, 3]):<10.4f}")
        if args.perturbation_path or args.src_model:
            print("扰动测试集汇总")
            print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
            for i, row in enumerate(ud_results):
                print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
            print(
                f"{row_labels[-2]:<10} {np.mean(ud_results[:idx + 1, 0]):<10.4f} {np.mean(ud_results[:idx + 1, 1]):<10.4f} "
                f"{np.mean(ud_results[:idx + 1, 2]):<10.4f} {np.mean(ud_results[:idx + 1, 3]):<10.4f}"
            )
            print(
                f"{row_labels[-1]:<10} {np.std(ud_results[:idx + 1, 0]):<10.4f} {np.std(ud_results[:idx + 1, 1]):<10.4f} "
                f"{np.std(ud_results[:idx + 1, 2]):<10.4f} {np.std(ud_results[:idx + 1, 3]):<10.4f}"
            )
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 50)
    print(model)

    # Update final results array to hold 4 metrics
    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(final_results,
                      columns=['Acc', 'F1', 'BCA', 'EER'],
                      index=['2024', '2025', '2026', '2027', '2028', "Avg", "Std"])
    df = df.round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"UID_{args.model}.csv")

    if args.perturbation_path or args.src_model:
        ud_final_results = np.vstack([ud_results, np.mean(ud_results, axis=0), np.std(ud_results, axis=0)])
        ud_df = pd.DataFrame(ud_final_results,
                             columns=['Acc', 'F1', 'BCA', 'EER'],
                             index=['2024', '2025', '2026', '2027', '2028', "Avg", "Std"])
        ud_df = ud_df.round(4)
        ud_df.to_csv(csv_path / f"UID_{args.model}_UD.csv")


if __name__ == "__main__":
    main()
