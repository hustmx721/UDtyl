import sys
from pathlib import Path

import torch
import argparse
from .data_loader import *
from .models import LoadModel


def init_args():
    project_root = Path(__file__).resolve().parent.parent
    default_log_root = project_root / "logs"
    default_model_root = project_root / "ModelSave"
    default_csv_root = project_root / "csv"
    default_sys_path = project_root

    parser = argparse.ArgumentParser(description="Model Train Hyperparameter")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=9)
    parser.add_argument("--nclass", type=int, default=9)  # 用户数量
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=int, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--is_task", type=bool, default=True)
    # logs path
    parser.add_argument("--log_root", type=Path, default=default_log_root,
                        help="Directory to store training logs")
    parser.add_argument("--model_root", type=Path, default=default_model_root,
                        help="Directory to store trained model checkpoints")
    parser.add_argument("--csv_root", type=Path, default=default_csv_root,
                        help="Directory to store exported CSV results")
    parser.add_argument("--extra_sys_path", type=Path, default=default_sys_path,
                        help="Additional path to append to sys.path for imports")
    parser.add_argument("--em_outer", type=int, default=100,
                        help="Maximum number of outer error-minimization rounds (Algorithm 1)")
    parser.add_argument("--em_iters", type=int, default=None,
                        help="Deprecated alias for em_outer; if set, overrides em_outer")
    parser.add_argument("--em_theta_epochs", type=int, default=10,
                        help="Number of parameter-update epochs (M in Algorithm 1) before refreshing noise")
    parser.add_argument("--em_pgd_steps", type=int, default=20,
                        help="Number of PGD steps to update error-minimizing noise δ")
    parser.add_argument("--em_eps", type=float, default=8/255.,
                        help="L_inf radius for error-minimizing noise")
    parser.add_argument("--em_alpha", type=float, default=None,
                        help="Step size for PGD noise update; defaults to em_eps/10 when unset")
    parser.add_argument("--em_lambda", type=float, default=0.1,
                        help="Training error threshold λ to stop error-minimization early")
    parser.add_argument("--em_clip_min", type=float, default=None,
                        help="Optional lower clamp for perturbed inputs (e.g., 0.0 for images)")
    parser.add_argument("--em_clip_max", type=float, default=None,
                        help="Optional upper clamp for perturbed inputs (e.g., 1.0 for images)")
    parser.add_argument("--em_init_model", type=str, default=None,
                        help="Optional checkpoint path to initialize EM model weights")

    args = parser.parse_args()

    # Append additional sys.path if provided
    if args.extra_sys_path:
        extra_path = args.extra_sys_path
        if not extra_path.is_absolute():
            extra_path = project_root / extra_path
        resolved_path = extra_path.resolve()
        if str(resolved_path) not in sys.path:
            sys.path.append(str(resolved_path))

    return args


def set_args(args: argparse.ArgumentParser):
    OpenBMI = ["MI", "SSVEP", "ERP"]
    M3CV = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]
    if args.dataset in OpenBMI:
        args.channel = 62
        args.fs = 250
        if args.dataset == "ERP":
            args.nclass = 2
            args.timepoint = 0.8
        elif args.dataset == "MI":
            args.nclass = 2
            args.timepoint = 4
        elif args.dataset == "SSVEP":
            args.nclass = 4
            args.timepoint = 4
        # UID分类
        if not args.is_task:
            args.nclass = 54
    elif args.dataset in M3CV:
        args.channel = 64
        args.fs = 250
        args.timepoint = 4
        match args.dataset:
            case "Rest":
                args.nclass = 2
            case "Transient":
                args.nclass = 3
            case "Steady":
                args.nclass = 3
            case "Motor":
                args.nclass = 3
        # UID分类
        if not args.is_task:
            args.nclass = 20
    return args


def load_data(args: argparse.ArgumentParser, include_index: bool = False):
    OpenBMI = ["MI", "SSVEP", "ERP"]
    M3CV = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]
    if args.dataset in OpenBMI:
        trainloader, valloader, testloader = GetLoaderOpenBMI(args.seed, Task=args.dataset, is_task=args.is_task, include_index=include_index)
    elif args.dataset in M3CV:
        trainloader, valloader, testloader = GetLoaderM3CV(args.seed, Task=args.dataset, is_task=args.is_task, include_index=include_index)
    else:
        raise ValueError("Invalid dataset name")
    return trainloader, valloader, testloader


def load_all(args: argparse.ArgumentParser):
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    model = LoadModel(model_name=args.model, Chans=args.channel, Samples=int(args.fs*args.timepoint), n_classes=args.nclass).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initlr)

    return model, optimizer, device