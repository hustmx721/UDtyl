import os, gc
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Cap low-level library threading before importing heavy deps to avoid oversubscribing
# Allow overriding via DATA_WORKERS env for easier debugging or tighter CPU limits.
DEFAULT_WORKERS = max(1, int(os.environ.get("DATA_WORKERS", min(4, (os.cpu_count() or 1)))))
os.environ.setdefault("OMP_NUM_THREADS", str(DEFAULT_WORKERS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(DEFAULT_WORKERS))
os.environ.setdefault("MKL_NUM_THREADS", str(DEFAULT_WORKERS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(DEFAULT_WORKERS))

import scipy, pickle
import numpy as np
import psutil
import torch
from einops import rearrange
from sklearn.model_selection import train_test_split
import scipy.linalg as la
from utils.dataset import ToDataLoader, ToDataLoaderLLock, set_seed
from utils.preprocess import preprocessing

# CPU kernel limitation
def LimitCpu():
    # 获取可用的CPU核心数
    num_cores = psutil.cpu_count(logical=False)
    available_memory = psutil.virtual_memory().available
    # 假设使用32位浮点数(4bytes),1/16减少内存开销占用
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16)  
    return max_chunk_size

global max_chunk_size
max_chunk_size = LimitCpu()


def _resolve_device(target_device):
    if target_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(target_device, torch.device):
        return target_device
    if isinstance(target_device, int):
        return torch.device(f"cuda:{target_device}")
    return torch.device(target_device)


def _to_device_tensor(array, device, *, dtype=torch.float32, pin_memory=False, non_blocking=True):
    tensor = torch.as_tensor(array, dtype=dtype)
    if pin_memory and device.type == "cpu":
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=non_blocking)


def _prepare_loaders(
    tx,
    vx,
    test_x,
    ts,
    vs,
    test_y,
    batchsize,
    include_index,
    llock_gpu,
    target_device,
    pin_memory,
    non_blocking,
):
    if llock_gpu:
        device = _resolve_device(target_device)
        pin_flag = pin_memory and device.type == "cpu"

        def _convert_data(arr):
            arr = np.expand_dims(arr, axis=1) if arr.ndim == 3 else arr
            return _to_device_tensor(
                arr,
                device,
                dtype=torch.float32,
                pin_memory=pin_flag,
                non_blocking=non_blocking,
            )

        data_tensors = [_convert_data(arr) for arr in [tx, vx, test_x]]
        label_tensors = [
            _to_device_tensor(lbl, device, dtype=torch.long, pin_memory=pin_flag, non_blocking=non_blocking)
            for lbl in [ts, vs, test_y]
        ]

        loaders = [
            ToDataLoaderLLock(
                x,
                y,
                mode,
                batch_size=batchsize,
                include_index=include_index,
                pin_memory=pin_flag,
            )
            for x, y, mode in zip(data_tensors, label_tensors, ["train", "test", "test"])
        ]
    else:
        tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]]
        loaders = [
            ToDataLoader(x, y, mode, batch_size=batchsize, include_index=include_index)
            for x, y, mode in zip([tx, vx, test_x], [ts, vs, test_y], ["train", "test", "test"])
        ]
    return loaders


def Task(x, y, ratio=0.8, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(idx)
    train_size = int(len(x) * ratio)

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]


def balance_split(x, y, num_class, ratio):
    lb_idx = []
    for c in range(num_class):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, int(np.ceil(len(idx) * ratio)), False)
        lb_idx.extend(idx)
    ulb_idx = np.array(sorted(list(set(range(len(x))) - set(lb_idx))))

    return x[lb_idx], y[lb_idx], x[ulb_idx], y[ulb_idx]


def standard_normalize(x, clip_range=None):
    mean, std = np.mean(x), np.std(x)
    x = (x - mean) / std
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def align(data):
    """data alignment"""
    data_align = []
    length = len(data)
    rf_matrix = np.dot(data[0], np.transpose(data[0]))
    for i in range(1, length):
        rf_matrix += np.dot(data[i], np.transpose(data[i]))
    rf_matrix /= length

    rf = la.inv(la.sqrtm(rf_matrix))
    if rf.dtype == complex:
        rf = rf.astype(np.float64)

    for i in range(length):
        data_align.append(np.dot(rf, data[i]))

    return np.asarray(data_align).squeeze(), rf


# Get Filter-bank EEG
def SubBandSplit(data: np.ndarray, freq_start: int = 4, freq_end: int = 40, bandwidth: int = 4, fs: int = 250):
    """
    优化后的子带切分函数
    data(batch,channel,time) --> sub_band_data(batch,(channel*nBands),time)
    """
    @lru_cache(maxsize=32)
    def get_sos_coeffs(freq_low, freq_high, fs):
        """缓存并返回 SOS 滤波器系数"""
        return scipy.signal.butter(6, [2.0 * freq_low / fs, 2.0 * freq_high / fs], 'bandpass', output='sos')

    def process_single_band(args):
        """处理单个频带的数据"""
        data, freq_low, freq_high = args
        sos = get_sos_coeffs(freq_low, freq_high, fs)
        return scipy.signal.sosfilt(sos, data, axis=-1)

    subbands = np.arange(freq_start, freq_end + 1, bandwidth)
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        band_args = [(data, low_freq, high_freq)
                     for low_freq, high_freq in zip(subbands[:-1], subbands[1:])]
        results = list(executor.map(process_single_band, band_args))

    sub_band_data = np.stack(results, axis=1).astype(np.float32)
    del results
    gc.collect()
    # return rearrange(sub_band_data, 'b c t n -> b (c n) t')
    return sub_band_data

# MI:(54,200,62,4000) --> downsample: (54,200,62,1000)
# SSVEP:(54,200,62,4000) --> downsample : (54,200,62,1000)
# ERP:(54,4140,62,800) --> downsample : (54,200,62,200)
def GetLoaderOpenBMI(
    seed,
    Task: str = "MI",
    batchsize: int = 64,
    is_task: bool = True,
    include_index: bool = False,
    llock_gpu: bool = False,
    target_device=None,
    pin_memory: bool = True,
    non_blocking: bool = True,
):  # Task = "ERP", "MI", "SSVEP"
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    def process_data(data):
        processed_data = np.empty_like(data)
        chunk_size = min(data.shape[0], max_chunk_size)
        with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
            futures = []
            for i in range(0, data.shape[0], chunk_size):
                chunk = data[i:i+chunk_size]
                future = executor.submit(processor, chunk)
                futures.append((i, future))
            for i, future in futures:
                processed_data[i:i+chunk_size] = future.result()
        return processed_data
    
    # 统一加载 EEG 数据
    data_train = load_data(f'/mnt/data1/tyl/data/OpenBMI/Task/{Task}/train.pkl')
    data_test  = load_data(f'/mnt/data1/tyl/data/OpenBMI/Task/{Task}/test.pkl')

    train_x = data_train['data'].astype(np.float32)
    test_x  = data_test['data'].astype(np.float32)
    train_x, test_x = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [train_x, test_x]]

    # 根据任务类型选择标签
    if is_task:
        train_y = data_train['label'].astype(np.int16)
        test_y  = data_test['label'].astype(np.int16)
    else:
        subj_train = load_data(f'/mnt/data1/tyl/data/OpenBMI/processed/{Task}/train.pkl')
        subj_test  = load_data(f'/mnt/data1/tyl/data/OpenBMI/processed/{Task}/test.pkl')
        train_y = (subj_train['ori_train_s'] - 1).astype(np.int16)
        test_y  = (subj_test['ori_test_s'] - 1).astype(np.int16)

    train_y, test_y = [s.reshape(-1) for s in [train_y, test_y]]
        
    fs = 250
    DataProcessor = preprocessing(fs=fs)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [process_data(x) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_y, test_size=0.2, random_state=seed, stratify=train_y)

    print("-----数据预处理完成-----")
    print(f"是否任务分类: {is_task}, 类别数量: {len(np.unique(train_y))}")
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")

    trainloader, validateloader, testloader = _prepare_loaders(
        tx,
        vx,
        test_x,
        ts,
        vs,
        test_y,
        batchsize,
        include_index,
        llock_gpu,
        target_device,
        pin_memory,
        non_blocking,
    )
    return trainloader, validateloader, testloader


"""
Clibration#---------------------------------------------- Test
Rest (600, 65, 1000) (600,) -- 20 * 30 (subs * trials)
Transient (1791, 65, 1000) (1791,) -- 20 * (88~90)
Steady (740, 65, 1000) (740,) -- 20 * 37
P300 (299, 65, 1000) (299,) -- 20 * (15~14)
Motor (2400, 65, 1000) (2400,) -- 20 * 120
SSVEP_SA (240, 65, 1000) (240,) -- 20 * 12

Partial Enrollment#---------------------------------------------- Real Train
Rest (1200, 65, 1000) (1200,) -- 20 * 60 (subs * trials)
Transient (3590, 65, 1000) (3590,) -- 20 * (178~180)
Steady (1530, 65, 1000) (1530,) -- 20 * (75/105)
P300 (599, 65, 1000) (599,) -- 20 * (29~30)
Motor (4828, 65, 1000) (4828,) -- 20* (239~243 / 265)
SSVEP_SA (480, 65, 1000) (480,) -- 20 * 24
! 注意: 去除EasyCap后是64通道
"""
# Task = "Rest"， "Transient", "Steady", "Motor"
def GetLoaderM3CV(
    seed,
    Task: str = "Rest",
    batchsize: int = 64,
    is_task: bool = True,
    include_index: bool = False,
    llock_gpu: bool = False,
    target_device=None,
    pin_memory: bool = True,
    non_blocking: bool = True,
):
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    def process_data(data):
        processed_data = np.empty_like(data)
        chunk_size = min(data.shape[0], max_chunk_size)
        with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
            futures = []
            for i in range(0, data.shape[0], chunk_size):
                chunk = data[i:i+chunk_size]
                future = executor.submit(processor, chunk)
                futures.append((i, future))
            for i, future in futures:
                processed_data[i:i+chunk_size] = future.result()
        return processed_data
    
    # different session for train and test
    data_train = load_data(f'/mnt/data1/tyl/data/M3CV/Task/Session1_{Task}.pkl')
    data_test  = load_data(f'/mnt/data1/tyl/data/M3CV/Task/Session2_{Task}.pkl')

    # 去除 EasyCap 通道
    train_x = data_train['data'][:, :-1, :].astype(np.float32)
    test_x  = data_test['data'][:, :-1, :].astype(np.float32)

    # === 根据任务类型选择标签 ===
    if is_task:
        train_y = data_train['label'].astype(np.int16)
        test_y  = data_test['label'].astype(np.int16)
    else:
        # 身份分类 → 使用身份标签
        subj_train = load_data(f'/mnt/data1/tyl/data/M3CV/Train/T_{Task}.pkl')
        subj_test  = load_data(f'/mnt/data1/tyl/data/M3CV/Test/{Task}.pkl')
        train_y = subj_train['label'].astype(np.int16)
        test_y  = subj_test['label'].astype(np.int16)


    DataProcessor = preprocessing(fs=250) 
    processor = DataProcessor.EEGpipline

    train_x, test_x = [process_data(x) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_y, test_size=0.2, random_state=seed, stratify=train_y)

    # [tx, vx, test_x] = [SubBandSplit(x,8,32,2) for x in [tx, vx, test_x]]
    print("-----数据预处理完成-----")
    print(f"是否任务分类: {is_task}, 类别数量: {len(np.unique(train_y))}")
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")

    trainloader, validateloader, testloader = _prepare_loaders(
        tx,
        vx,
        test_x,
        ts,
        vs,
        test_y,
        batchsize,
        include_index,
        llock_gpu,
        target_device,
        pin_memory,
        non_blocking,
    )
    del data_train, data_test, train_x, train_y, test_x, test_y, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader

def Load_Dataloader(
    seed,
    Task,
    batchsize,
    is_task=True,
    include_index: bool = False,
    llock_gpu: bool = False,
    target_device=None,
    pin_memory: bool = True,
    non_blocking: bool = True,
):
    OpenBMI = ["MI", "SSVEP", "ERP"]
    M3CV = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]
    set_seed(seed)

    if Task in OpenBMI:
        trainloader, valloader, testloader = GetLoaderOpenBMI(
            seed,
            Task=Task,
            batchsize=batchsize,
            is_task=is_task,
            include_index=include_index,
            llock_gpu=llock_gpu,
            target_device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
    elif Task in M3CV:
        trainloader, valloader, testloader = GetLoaderM3CV(
            seed,
            Task=Task,
            batchsize=batchsize,
            is_task=is_task,
            include_index=include_index,
            llock_gpu=llock_gpu,
            target_device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
    return trainloader, valloader, testloader
