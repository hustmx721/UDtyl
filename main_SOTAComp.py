import os
import sys
import time
import copy
import gc
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, set_args, load_all, load_data
from utils.Logging import Logger
from evaluate import evaluate
from main_EM import em_error_min_train
from main_LLock import build_lock_params, create_lock, train_lock_and_model, evaluate_with_lock
from main_Distill import build_argument_parser as build_distill_parser, train_distillation
from main_Handi import (
    build_template,
    train_one_epoch_with_template,
    evaluate_with_template,
)

warnings.filterwarnings("ignore")


def _ensure_dir(path: str | os.PathLike) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _save_method_csv(results: np.ndarray, seeds: List[int], args, method: str) -> None:
    metrics = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(
        metrics,
        columns=["Acc", "F1", "BCA", "EER", "Time"],
        index=[*(str(seed) for seed in seeds), "Avg", "Std"],
    ).round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    _ensure_dir(csv_path)
    df.to_csv(csv_path / f"SOTAComp_{method}_{args.model}.csv")


def _build_distill_args(base_args):
    parser = build_distill_parser()
    distill_args = parser.parse_args([])
    distill_args.dataset = base_args.dataset
    distill_args.gpuid = base_args.gpuid
    distill_args.seed = base_args.seed
    distill_args.task_model = base_args.model
    distill_args.uid_model = base_args.model
    distill_args.initlr = base_args.initlr
    distill_args.bs = base_args.bs
    distill_args.model_root = base_args.model_root
    distill_args.csv_root = base_args.csv_root
    distill_args.log_root = base_args.log_root
    distill_args.repeats = 3
    return distill_args


def _train_handi_method(trainloader, valloader, args, device, template):
    model, optimizer, device = load_all(args)
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    best_epoch = 0
    best_acc = 0.0
    best_state = None

    for epoch in range(args.epoch):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch_with_template(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
            template=template,
        )

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate_with_template(
            model=model,
            dataloader=valloader,
            device=device,
            template=template,
        )

        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch:{epoch + 1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}"
                f"\tTrain_loss:{train_loss:.6f}\tVal_loss:{val_loss:.6f}"
            )
            print(
                f"  Train_F1:{train_f1:.4f}, BCA:{train_bca:.4f}, EER:{train_eer:.4f}"
                f" | Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"最佳验证准确率为 {best_acc * 100:.2f}% (epoch {best_epoch + 1})")
    return model


def run_em(args, seeds: List[int]) -> np.ndarray:
    results = np.zeros((len(seeds), 5))

    for idx, seed in enumerate(seeds):
        start_time = time.time()
        args.seed = seed
        args = set_args(args)

        print("=" * 30)
        print(f"[EM] dataset: {args.dataset}")
        print(f"[EM] model  : {args.model}")
        print(f"[EM] seed   : {args.seed}")
        print(f"[EM] gpu    : {args.gpuid}")
        print(f"[EM] is_task: {args.is_task}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args, include_index=False)

        model_path = args.model_root / f"{args.dataset}"
        _ensure_dir(model_path)

        model, optimizer, device = load_all(args)
        model, _ = em_error_min_train(
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            savepath=model_path,
            args=args,
            device=device,
            mode_tag="Task" if args.is_task else "UID",
        )

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate(
            model, testloader, args, device
        )
        elapsed = time.time() - start_time

        results[idx] = [test_acc, test_f1, test_bca, test_eer, elapsed]
        print(
            f"[EM] 测试集 Acc:{test_acc * 100:.2f}% F1:{test_f1 * 100:.2f}% "
            f"BCA:{test_bca * 100:.2f}% EER:{test_eer * 100:.2f}%"
        )
        print(f"[EM] 总耗时: {elapsed:.2f}s")

        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_llock_linear(args, seeds: List[int]) -> np.ndarray:
    results = np.zeros((len(seeds), 5))

    for idx, seed in enumerate(seeds):
        start_time = time.time()
        args.seed = seed
        args.lock_type = "linear"
        args = set_args(args)

        print("=" * 30)
        print(f"[LLock-Linear] dataset: {args.dataset}")
        print(f"[LLock-Linear] model  : {args.model}")
        print(f"[LLock-Linear] seed   : {args.seed}")
        print(f"[LLock-Linear] gpu    : {args.gpuid}")
        print(f"[LLock-Linear] is_task: {args.is_task}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args)

        lock_params = build_lock_params(trainloader, args)
        model, optimizer, device = load_all(args)
        torch.cuda.empty_cache()
        if device.type == "cuda":
            torch.cuda.set_device(device)

        lock = create_lock(args, lock_params, device)
        train_lock_and_model(trainloader, lock, model, args, device)

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate_with_lock(
            model, lock, testloader, device
        )

        elapsed = time.time() - start_time
        results[idx] = [test_acc, test_f1, test_bca, test_eer, elapsed]
        print(
            f"[LLock-Linear] 测试集 Acc:{test_acc * 100:.2f}% F1:{test_f1 * 100:.2f}% "
            f"BCA:{test_bca * 100:.2f}% EER:{test_eer * 100:.2f}%"
        )
        print(f"[LLock-Linear] 总耗时: {elapsed:.2f}s")

        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_handi_method(args, seeds: List[int], method: str) -> np.ndarray:
    results = np.zeros((len(seeds), 5))

    for idx, seed in enumerate(seeds):
        start_time = time.time()
        args.seed = seed
        args.handi_method = method
        args = set_args(args)

        print("=" * 30)
        print(f"[Handi-{method.upper()}] dataset: {args.dataset}")
        print(f"[Handi-{method.upper()}] model  : {args.model}")
        print(f"[Handi-{method.upper()}] seed   : {args.seed}")
        print(f"[Handi-{method.upper()}] gpu    : {args.gpuid}")
        print(f"[Handi-{method.upper()}] is_task: {args.is_task}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args, include_index=True)

        device = torch.device(
            "cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu"
        )
        template = build_template(trainloader, args, device)
        model = _train_handi_method(trainloader, valloader, args, device, template)

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate_with_template(
            model, testloader, device, template
        )

        elapsed = time.time() - start_time
        results[idx] = [test_acc, test_f1, test_bca, test_eer, elapsed]
        print(
            f"[Handi-{method.upper()}] 测试集 Acc:{test_acc * 100:.2f}% "
            f"F1:{test_f1 * 100:.2f}% BCA:{test_bca * 100:.2f}% EER:{test_eer * 100:.2f}%"
        )
        print(f"[Handi-{method.upper()}] 总耗时: {elapsed:.2f}s")

        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_distill(args, seeds: List[int]) -> np.ndarray:
    results = np.zeros((len(seeds), 5))

    for idx, seed in enumerate(seeds):
        start_time = time.time()
        distill_args = _build_distill_args(args)
        distill_args.seed = seed

        print("=" * 30)
        print(f"[Distill] dataset: {distill_args.dataset}")
        print(f"[Distill] task model: {distill_args.task_model}")
        print(f"[Distill] uid model : {distill_args.uid_model}")
        print(f"[Distill] seed      : {distill_args.seed}")
        print(f"[Distill] gpu       : {distill_args.gpuid}")

        metrics = train_distillation(distill_args)
        pert_task = metrics["perturbed_task"]
        pert_uid = metrics["perturbed_uid"]

        elapsed = time.time() - start_time
        results[idx] = [pert_uid[0], pert_uid[1], pert_uid[2], pert_uid[3], elapsed]

        print(
            f"[Distill] Task Acc:{pert_task[0] * 100:.2f}% F1:{pert_task[1] * 100:.2f}% "
            f"BCA:{pert_task[2] * 100:.2f}% EER:{pert_task[3] * 100:.2f}%"
        )
        print(
            f"[Distill] UID  Acc:{pert_uid[0] * 100:.2f}% F1:{pert_uid[1] * 100:.2f}% "
            f"BCA:{pert_uid[2] * 100:.2f}% EER:{pert_uid[3] * 100:.2f}%"
        )
        print(f"[Distill] 总耗时: {elapsed:.2f}s")

        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    args = init_args()
    args = set_args(args)
    apply_thread_limits(getattr(args, "torch_threads", 5))

    log_path = args.log_root / f"SOTAComp_{args.dataset}_{args.model}.log"
    _ensure_dir(args.log_root)
    sys.stdout = Logger(log_path)

    seeds = list(range(args.seed, args.seed + args.repeats))

    method_results: Dict[str, np.ndarray] = {}

    print("开始进行 SOTA baseline 对比")
    method_results["EM"] = run_em(copy.deepcopy(args), seeds)
    method_results["LLockLinear"] = run_llock_linear(copy.deepcopy(args), seeds)
    method_results["SN"] = run_handi_method(copy.deepcopy(args), seeds, "sn")
    method_results["RAND"] = run_handi_method(copy.deepcopy(args), seeds, "rand")
    method_results["Distill"] = run_distill(copy.deepcopy(args), seeds)

    summary_rows = []
    for method, result in method_results.items():
        _save_method_csv(result, seeds, args, method)
        summary_rows.append(
            {
                "Method": method,
                "Acc": np.mean(result[:, 0]),
                "F1": np.mean(result[:, 1]),
                "BCA": np.mean(result[:, 2]),
                "EER": np.mean(result[:, 3]),
                "Time": np.mean(result[:, 4]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("Method").round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    _ensure_dir(csv_path)
    summary_df.to_csv(csv_path / f"SOTAComp_summary_{args.model}.csv")
    print("SOTA baseline 对比完成。")


if __name__ == "__main__":
    main()
