import sys
import os
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
    parser.add_argument(
        "--gpu_prefetch",
        action="store_true",
        help="Enable GPU-first dataloader path (preload tensors to GPU or pinned memory for LLock).",
    )
    parser.add_argument(
        "--gpu_prefetch_pin_memory",
        action="store_true",
        help="Pin host memory when GPU-first prefetch is enabled and tensors remain on CPU.",
    )
    parser.add_argument(
        "--gpu_prefetch_non_blocking",
        action="store_true",
        help="Use non_blocking transfers when moving prefetched tensors to GPU.",
    )
    parser.add_argument("--torch_threads", type=int, default=4,
                        help="Number of threads to use for torch operations")
    parser.add_argument(
        "--handi_method",
        type=str,
        default="rand",
        choices=["rand", "sn", "stft"],
        help="Handcrafted UD template type used in main_handi.py",
    )
    parser.add_argument(
        "--handi_alpha",
        type=float,
        default=0.05,
        help="Scaling factor for handcrafted UD templates",
    )
    # logs path
    parser.add_argument("--log_root", type=Path, default=default_log_root,
                        help="Directory to store training logs")
    parser.add_argument("--model_root", type=Path, default=default_model_root,
                        help="Directory to store trained model checkpoints")
    parser.add_argument("--csv_root", type=Path, default=default_csv_root,
                        help="Directory to store exported CSV results")
    parser.add_argument("--extra_sys_path", type=Path, default=default_sys_path,
                        help="Additional path to append to sys.path for imports")
    # LLock args
    parser.add_argument(
        "--lock_type",
        type=str,
        default="linear",
        choices=["linear", "ires"],
        help="Type of learnability lock to apply during training",
    )
    parser.add_argument(
        "--lock_epsilon",
        type=float,
        default=1e-3,
        help="Perturbation budget (epsilon) used by learnability locks",
    )
    parser.add_argument(
        "--lock_mid_planes",
        type=int,
        default=4,
        help="Hidden channel size for iResLock transforms",
    )
    # em args
    """
     Error-Minimization (EM) 参数说明：
      - em_outer：外层EM迭代轮数（算法1中的循环次数）。
      - em_iters：em_outer的兼容别名；若设置则覆盖em_outer。
      - em_theta_epochs：每次刷新噪声前，模型参数更新的epoch数（算法1中的M）。
      - em_pgd_steps：用于更新误差最小化噪声δ的PGD步数。
      - em_eps：噪声的L_inf半径限制。
      - em_alpha：PGD更新步长，默认取em_eps/10。
      - em_lambda：训练误差阈值λ，低于该值则提前停止EM。
      - em_clip_min / em_clip_max：对扰动后输入的可选下/上界（如图像0~1范围）。
      - em_init_model：EM阶段初始化模型权重的可选检查点路径。
    """
    parser.add_argument("--em_outer", type=int, default=100,
                        help="Maximum number of outer error-minimization rounds (Algorithm 1)")
    parser.add_argument("--em_iters", type=int, default=None,
                        help="Deprecated alias for em_outer; if set, overrides em_outer")
    parser.add_argument("--em_theta_epochs", type=int, default=5,
                        help="Number of parameter-update epochs (M in Algorithm 1) before refreshing noise")
    parser.add_argument("--em_pgd_steps", type=int, default=10,
                        help="Number of PGD steps to update error-minimizing noise δ")
    parser.add_argument("--em_eps", type=float, default=1e-3,
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
    loader_kwargs = dict(
        include_index=include_index,
        llock_gpu=getattr(args, "gpu_prefetch", False),
        target_device=f"cuda:{args.gpuid}" if getattr(args, "gpu_prefetch", False) else None,
        pin_memory=getattr(args, "gpu_prefetch_pin_memory", False) or getattr(args, "gpu_prefetch", False),
        non_blocking=getattr(args, "gpu_prefetch_non_blocking", False) or getattr(args, "gpu_prefetch", False),
    )
    batch_size = getattr(args, "bs", 64)

    if args.dataset in OpenBMI:
        trainloader, valloader, testloader = GetLoaderOpenBMI(
            args.seed,
            Task=args.dataset,
            batchsize=batch_size,
            is_task=args.is_task,
            **loader_kwargs,
        )
    elif args.dataset in M3CV:
        trainloader, valloader, testloader = GetLoaderM3CV(
            args.seed,
            Task=args.dataset,
            batchsize=batch_size,
            is_task=args.is_task,
            **loader_kwargs,
        )
    else:
        raise ValueError("Invalid dataset name")
    return trainloader, valloader, testloader



def apply_thread_limits(thread_count: int | None):
    """Clamp CPU thread usage for torch and BLAS backends.

    Ensures a consistent ceiling across libraries and provides a single place to
    tighten thread counts when running multiple experiments in parallel.
    """
    thread_count = max(1, thread_count or 1)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(max(1, thread_count // 2))
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = str(thread_count)


def load_all(args: argparse.ArgumentParser):
    # thread_count = max(1, getattr(args, "torch_threads", 4))
    # apply_thread_limits(thread_count)
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    model = LoadModel(model_name=args.model, Chans=args.channel, Samples=int(args.fs*args.timepoint), n_classes=args.nclass).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initlr)

    return model, optimizer, device
