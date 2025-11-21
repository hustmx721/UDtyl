import os
import copy
import logging
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from models import LoadModel, Classifier, Discriminator, CalculateOutSize
from utils.data_loader import DataLoad
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score
from unlearnable_gen import unlearnable_optim


def trainer(feature_ext, model, x_train, y_train, x_test, y_test, phase, args):
    feature_ext.apply(init_weights)
    model.apply(init_weights)
    params = []
    for _, v in feature_ext.named_parameters():
            params += [{'params': v, 'lr': args.lr}]
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    sample_weights = weight_for_balanced_classes(y_train)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False)

    train_recorder, test_recorder = [], []
    for epoch in range(args.epochs):
        # model training
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        feature_ext.train()
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = model(feature_ext(batch_x))
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            feature_ext.MaxNormConstraint()
            if phase == 'c': model.MaxNormConstraint()

        if (epoch + 1) % 1 == 0:
            feature_ext.eval()
            model.eval()
            train_loss, train_acc, train_bca = eval(feature_ext, model,
                                                    criterion, train_loader)
            test_loss, test_acc, test_bca = eval(feature_ext, model, criterion,
                                                     test_loader)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train bca: {:.2f} | test loss: {:.4f} test bca: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_bca, test_loss, test_bca))

            if phase == 'c':
                train_recorder.append(train_bca)
                test_recorder.append(test_bca)
            else:
                train_recorder.append(train_acc)
                test_recorder.append(test_acc)
    return feature_ext, model, train_recorder, test_recorder


def train(x_train, y_train, x_pert, x_test, y_test, args):
    x_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor))
    y_train = Variable(torch.from_numpy(y_train).type(torch.LongTensor))
    x_pert = Variable(torch.from_numpy(x_pert).type(torch.FloatTensor))

    x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
    y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

    # initialize the model
    # model
    chans, samples = x_train.shape[2], x_train.shape[3]
    feat = LoadModel(model_name=args.model,
                            Chans=chans,
                            Samples=samples)

    cal = Classifier(input_dim=CalculateOutSize(feat, chans, samples),
                          n_classes=len(np.unique(y_train.numpy())))

    feat.to(args.device)
    cal.to(args.device)

    p_feat = copy.deepcopy(feat)
    p_cal = copy.deepcopy(cal)

    criterion = nn.CrossEntropyLoss().to(args.device)

    # gender privacy 
    logging.info('*' * 25 + ' train task classifier ' + '*' * 25)
    feat, cal, train_acc, test_acc = trainer(feat, cal, x_train, y_train, x_test, y_test, phase='c', args=args)

    feat.eval()
    cal.eval()

    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, acc, bca = eval(feat, cal, criterion, test_loader)

    logging.info(f'Original BCA: {bca}')

    # identity privacy
    logging.info('*' * 25 + ' train perturbed classifier ' + '*' * 25)
    p_feat, p_cal, p_train_acc, p_test_acc = trainer(p_feat,
                                                     p_cal,
                                                     x_pert,
                                                     y_train,
                                                     x_test,
                                                     y_test,
                                                     phase='c',
                                                     args=args)

    p_feat.eval()
    p_cal.eval()

    _, pacc, pbca = eval(p_feat, p_cal, criterion, test_loader)

    logging.info(f'Perturbed BCA: {pbca}')

    return bca, pbca


def eval(model1: nn.Module, model2: nn.Module, criterion: nn.Module,
         data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model2(model1(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='1')

    parser.add_argument('--batch_size', type=int, default=128) # 128
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--log', type=str, default='')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    log_path = f'results/log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(
        log_path, f'task_{args.model}.log')

    npz_path = f'results/npz/task_{args.model}'

    if not os.path.exists(npz_path):
        os.makedirs(npz_path)

    if len(args.log): log_name = log_name.replace('.log', f'_{args.log}.log')

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    # load data
    # MI_x, MI_y, MI_s, MI_g, MI_e = DataLoad(dataset='MI')
    # ERP_x, ERP_y, ERP_s, ERP_g, ERP_e = DataLoad(dataset='ERP')
    # SSVEP_x, SSVEP_y, SSVEP_s, SSVEP_g, SSVEP_e = DataLoad(dataset='SSVEP')
    MI_data = np.load('data/MI.npz')
    ERP_data = np.load('data/ERP.npz')
    SSVEP_data = np.load('data/SSVEP.npz')
    MI_x, MI_y, MI_s, MI_g, MI_e = MI_data['x'], MI_data['y'], MI_data['s'], MI_data['g'], MI_data['e']
    ERP_x, ERP_y, ERP_s, ERP_g, ERP_e = ERP_data['x'], ERP_data['y'], ERP_data['s'], ERP_data['g'], ERP_data['e']
    SSVEP_x, SSVEP_y, SSVEP_s, SSVEP_g, SSVEP_e = SSVEP_data['x'], SSVEP_data['y'], SSVEP_data['s'], SSVEP_data['g'],   SSVEP_data['e']
    # np.savez('MI.npz', x=MI_x, y=MI_y, s=MI_s, g=MI_g, e=MI_e)
    # np.savez('ERP.npz', x=ERP_x, y=ERP_y, s=ERP_s, g=ERP_g, e=ERP_e)
    # np.savez('SSVEP.npz', x=SSVEP_x, y=SSVEP_y, s=SSVEP_s, g=SSVEP_g, e=SSVEP_e)

    
    # model train
    r_bca, r_pbca = [], []
    for repeat in range(5):
        seed(repeat)
        perturbation = np.load(f'results/npz/protection_EEGNet_1e-5/repeat{repeat}.npz')
        s_bca, s_pbca = [], []
        for session in range(2):
            # model train
            x_train = np.concatenate([MI_x[session], ERP_x[session], SSVEP_x[session]], axis=0)
            s_train = np.concatenate([MI_s[session], ERP_s[session], SSVEP_s[session]])
            g_train = np.concatenate([MI_g[session], ERP_g[session], SSVEP_g[session]])
            e_train = np.concatenate([MI_e[session], ERP_e[session], SSVEP_e[session]])

            s_test = np.concatenate([MI_s[1-session], ERP_s[1-session], SSVEP_s[1-session]])
            g_test = np.concatenate([MI_g[1-session], ERP_g[1-session], SSVEP_g[1-session]])
            e_test = np.concatenate([MI_e[1-session], ERP_e[1-session], SSVEP_e[1-session]])

            g_pert, s_pert, e_pert = perturbation['g_pert'][session], perturbation['s_pert'][session], perturbation['e_pert'][session]
            protected_x = x_train + g_pert[g_train] + s_pert[s_train] + e_pert[e_train]

            logging.info(f'session {session + 1}')
            t_bca, t_pbca = [], []
            for task in ['MI', 'ERP', 'SSVEP']:
                if task == 'MI':
                    x_train, y_train, x_test, y_test = MI_x[session], MI_y[session]-1, MI_x[1-session], MI_y[1-session]-1
                    x_pert = protected_x[:len(MI_x[session])]
                elif task == 'ERP':
                    x_train, y_train, x_test, y_test = ERP_x[session], ERP_y[session]-1, ERP_x[1-session], ERP_y[1-session]-1
                    x_pert = protected_x[len(MI_x[session]):len(MI_x[session])+len(ERP_x[session])] 
                elif task == 'SSVEP':
                    x_train, y_train, x_test, y_test = SSVEP_x[session], SSVEP_y[session]-1, SSVEP_x[1-session], SSVEP_y[1-session]-1
                    x_pert = protected_x[len(MI_x[session])+len(ERP_x[session]):]                                        
            
                logging.info(f'train: {len(x_train)}, test:{len(x_test)}')

                bca, pbca = train(x_train, y_train, x_pert, x_test, y_test, args)

                t_bca.append(bca)
                t_pbca.append(pbca)

                logging.info(f'Task {task} bca/pbca: {bca}/{pbca}')
            
            s_bca.append(t_bca)
            s_pbca.append(t_pbca)
            logging.info(f'MI/ERP/SSVEP bca/pbca: {t_bca}/{t_pbca}')
        
        r_bca.append(s_bca)
        r_pbca.append(s_pbca)
        logging.info(f'repeat {repeat + 1}')
        logging.info(f'MI/ERP/SSVEP bca/pbca: {np.mean(s_bca, axis=(0))}/{np.mean(s_pbca, axis=(0))}')

    logging.info(f'Avg results')
    logging.info(f'MI/ERP/SSVEP bca/pbca: {np.mean(r_bca, axis=(0, 1))}/{np.mean(r_pbca, axis=(0, 1))}')

    np.savez(npz_path + '/result.npz',  bca=r_bca, pbca=r_pbca)
