"""
Cross-session brainprint recognition using handcrafted features and SVM.

Features:
    - WPD (WaveletPacket)
    - STFT
    - AR (Burg)
    - MFCC

Usage example:
    python -u handifea/fea_cross_session.py --dataset Rest --feature STFT
"""

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np
import pickle
import warnings

from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder

from utils.preprocess import preprocessing
from utils.Logging import Logger
from handifea.fea import WaveletPacket, STFT, AR_burg, trans_mfccs

warnings.filterwarnings("ignore")


def calculate_eer(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, list]:
    """Compute average EER and per-class EER."""
    classes = np.unique(np.concatenate((y_true)))
    n_classes = len(classes)

    encoder = OneHotEncoder(sparse_output=False)
    y_true_bin = encoder.fit_transform(y_true.reshape(-1, 1))
    y_pred_bin = encoder.transform(y_pred.reshape(-1, 1))

    class_eer = []
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        fnr = 1 - tpr
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = np.mean([fpr[idx], fnr[idx]])
        class_eer.append(eer)

    avg_eer = np.mean(class_eer)
    return avg_eer, class_eer


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_m3cv_cross_session(dataset: str, label_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Load M3CV session1/session2 data with task or UID labels."""
    data_train = _load_pickle(f"/mnt/data1/tyl/data/M3CV/Task/Session1_{dataset}.pkl")
    data_test = _load_pickle(f"/mnt/data1/tyl/data/M3CV/Task/Session2_{dataset}.pkl")

    x_train = data_train["data"][:, :-1, :].astype(np.float32)
    x_test = data_test["data"][:, :-1, :].astype(np.float32)

    if label_type == "task":
        y_train = data_train["label"].astype(np.int16)
        y_test = data_test["label"].astype(np.int16)
    else:
        subj_train = _load_pickle(f"/mnt/data1/tyl/data/M3CV/Train/T_{dataset}.pkl")
        subj_test = _load_pickle(f"/mnt/data1/tyl/data/M3CV/Test/{dataset}.pkl")
        y_train = subj_train["label"].astype(np.int16)
        y_test = subj_test["label"].astype(np.int16)

    fs = 250
    t = 4.0
    return x_train, y_train, x_test, y_test, fs, t


def _extract_stft_features(data: np.ndarray, fs: int, window_seconds: float) -> np.ndarray:
    """Extract STFT features for data shaped [trials, channels, samples]."""
    data_4d = data[None, ...]
    stft_features = STFT(data_4d, time_length=window_seconds, fs=fs)
    return stft_features.squeeze(0)


def _extract_mfcc_features(
    data: np.ndarray,
    fs: int,
    framesize: int,
    mel_band: int,
    hop_length: int,
) -> np.ndarray:
    trials, channels, _ = data.shape
    mfcc_features = []
    for trial in range(trials):
        channel_feats = []
        for channel in range(channels):
            mfcc = trans_mfccs(
                wav_data=data[trial, channel],
                sample_rate=fs,
                framesize=framesize,
                mel_band=mel_band,
                hop_length=hop_length,
            )
            channel_feats.append(mfcc.reshape(-1))
        mfcc_features.append(np.concatenate(channel_feats, axis=0))
    return np.array(mfcc_features, dtype=np.float32)


def extract_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    fs: int,
    feature: str,
    stft_window_seconds: float,
    ar_order: int,
    mfcc_framesize: int,
    mfcc_mel_band: int,
    mfcc_hop_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    start_time = time.time()

    DataProcessor = preprocessing(fs=fs)
    train_x = DataProcessor.EEGpipline(train_x)
    test_x = DataProcessor.EEGpipline(test_x)

    if feature == "WPD":
        train_f = WaveletPacket(train_x)
        test_f = WaveletPacket(test_x)
        train_f = train_f.reshape((train_f.shape[0], -1))
        test_f = test_f.reshape((test_f.shape[0], -1))
    elif feature == "STFT":
        train_f = _extract_stft_features(train_x, fs, stft_window_seconds)
        test_f = _extract_stft_features(test_x, fs, stft_window_seconds)
    elif feature == "AR":
        train_f = AR_burg(train_x, order=ar_order)
        test_f = AR_burg(test_x, order=ar_order)
        train_f = train_f.reshape((train_f.shape[0], -1))
        test_f = test_f.reshape((test_f.shape[0], -1))
    elif feature == "MFCC":
        train_f = _extract_mfcc_features(
            train_x,
            fs,
            framesize=mfcc_framesize,
            mel_band=mfcc_mel_band,
            hop_length=mfcc_hop_length,
        )
        test_f = _extract_mfcc_features(
            test_x,
            fs,
            framesize=mfcc_framesize,
            mel_band=mfcc_mel_band,
            hop_length=mfcc_hop_length,
        )
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    train_f = train_f.astype(np.float32)
    test_f = test_f.astype(np.float32)
    train_f[np.isnan(train_f)] = 0
    test_f[np.isnan(test_f)] = 0
    train_f[np.isinf(train_f)] = 1e6
    test_f[np.isinf(test_f)] = 1e6

    print(f"特征提取完毕,累计用时{time.time() - start_time:.2f}s!")
    return train_f, test_f


def clf_predict(train_f, test_f, train_y, test_y):
    start_time = time.time()
    clf = svm.SVC()
    clf.fit(train_f, train_y)
    print(f"分类器训练完毕,累计用时{time.time() - start_time:.2f}s!")

    pred = clf.predict(test_f)
    acc = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average="weighted")
    bca = balanced_accuracy_score(test_y, pred)
    recall = recall_score(test_y, pred, average="weighted")
    precision = precision_score(test_y, pred, average="weighted")
    eer, _ = calculate_eer(np.expand_dims(np.array(test_y), axis=0), np.expand_dims(np.array(pred), axis=0))
    return acc, f1, bca, recall, precision, eer


def main():
    parser = argparse.ArgumentParser(description="Cross-session handcrafted feature baseline (SVM)")
    parser.add_argument("--dataset", type=str, default="Rest")
    parser.add_argument("--feature", type=str, default="STFT", choices=["WPD", "STFT", "AR", "MFCC"])
    parser.add_argument("--label", type=str, default="uid", choices=["uid", "task"])
    parser.add_argument("--stft_window_seconds", type=float, default=1.0)
    parser.add_argument("--ar_order", type=int, default=5)
    parser.add_argument("--mfcc_framesize", type=int, default=256)
    parser.add_argument("--mfcc_mel_band", type=int, default=16)
    parser.add_argument("--mfcc_hop_length", type=int, default=128)
    parser.add_argument("--log_root", type=str, default="/mnt/data1/tyl/UnlearnableData/src/logs")
    args = parser.parse_args()

    sys.stdout = Logger(os.path.join(args.log_root, f"{args.dataset}_CrossSession_{args.feature}_SVM.log"))

    print("=" * 30)
    print(f"dataset: {args.dataset}")
    print(f"feature: {args.feature}")
    print("classifier: SVM")
    print(f"label: {args.label}")

    train_x, train_y, test_x, test_y, fs, _ = load_m3cv_cross_session(args.dataset, args.label)

    train_f, test_f = extract_features(
        train_x,
        test_x,
        fs,
        feature=args.feature,
        stft_window_seconds=args.stft_window_seconds,
        ar_order=args.ar_order,
        mfcc_framesize=args.mfcc_framesize,
        mfcc_mel_band=args.mfcc_mel_band,
        mfcc_hop_length=args.mfcc_hop_length,
    )

    acc, f1, bca, recall, precision, eer = clf_predict(train_f, test_f, train_y, test_y)
    print(
        f"用户分类准确率为{acc:.4f}, F1值为{f1:.4f}, BCA值为{bca:.4f}, "
        f"Recall为{recall:.4f}, Precision为{precision:.4f}, EER为{eer:.4f}"
    )
    print("=" * 30)


if __name__ == "__main__":
    main()
