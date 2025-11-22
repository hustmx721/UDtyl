import os
import sys
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from utils.dataset import set_seed
from utils.init_all import init_args, set_args, load_all, load_data
from utils.Logging import Logger

from evaluate import evaluate
from train import train_one_epoch


def classify(trainloader, valloader, savepath, args):
    mode_tag = "Task" if args.is_task else "UID"
    print("-" * 20 + f"开始{mode_tag}训练!" + "-" * 20)

    model, optimizer, device = load_all(args)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    best_epoch = 0
    best_acc = 0
    val_acc_all = []
    val_f1_all = []
    val_bca_all = []
    val_eer_all = []
    loss_item_train = []
    loss_item_val = []

    for epoch in tqdm(range(args.epoch), desc="Training:"):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func
        )

        loss_item_train.append(train_loss)

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model,
            dataloader=valloader,
            args=args
        )

        if (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(savepath, f"{mode_tag}_{args.model}_{args.seed}.pth"))

        val_acc_all.append(val_acc)
        val_f1_all.append(val_f1)
        val_bca_all.append(val_bca)
        val_eer_all.append(val_eer)
        loss_item_val.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{train_loss:.6f}\tVal_loss:{val_loss:.6f}")
            print(f"  Train_F1:{train_f1:.4f}, BCA:{train_bca:.4f}, EER:{train_eer:.4f} | Val_F1:{val_f1:.4f}, BCA:{val_bca:.4f}, EER:{val_eer:.4f}")

    print("-" * 20 + f"{mode_tag}训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model


def main():
    args = init_args()
    args = set_args(args)
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")

    results = np.zeros((5, 4))
    mode_tag = "Task" if args.is_task else "UID"
    log_path = args.log_root / f"{args.dataset}_{mode_tag}_{args.model}.log"
    sys.stdout = Logger(log_path)

    for idx, seed in enumerate(range(args.seed, args.seed + args.repeats)):
        args.seed = seed
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

        model = classify(trainloader, valloader, model_path, args)
        print("=====================model are trained===============")
        print(f"累计用时{time.time() - start_time:.4f}s!")

        test_loss, test_acc, test_f1, test_bca, test_eer = evaluate(model, testloader, args, device)

        results[idx] = [test_acc, test_f1, test_bca, test_eer]
        print(
            f"测试集平均指标为  Acc:{test_acc * 100:.2f}%;  F1:{test_f1 * 100:.2f}%;  BCA:{test_bca * 100:.2f}%; EER:{test_eer * 100:.2f}%;"
        )
        print("=====================test are done===================")

        row_labels = ['2024', '2025', '2026', '2027', '2028', "Avg", "Std"]
        col_labels = ['Acc', 'F1', 'BCA', 'EER']
        print(
            f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
        for i, row in enumerate(results):
            print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
        print(f"{row_labels[-2]:<10} {np.mean(results[:idx + 1, 0]):<10.4f} {np.mean(results[:idx + 1, 1]):<10.4f} {np.mean(results[:idx + 1, 2]):<10.4f} {np.mean(results[:idx + 1, 3]):<10.4f}")
        print(f"{row_labels[-1]:<10} {np.std(results[:idx + 1, 0]):<10.4f} {np.std(results[:idx + 1, 1]):<10.4f} {np.std(results[:idx + 1, 2]):<10.4f} {np.std(results[:idx + 1, 3]):<10.4f}")
        gc.collect()

    print("-" * 50)
    print(model)

    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(final_results,
                      columns=['Acc', 'F1', 'BCA', 'EER'],
                      index=['2024', '2025', '2026', '2027', '2028', "Avg", "Std"])
    df = df.round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path / f"{mode_tag}_{args.model}.csv")


if __name__ == "__main__":
    main()
