import os
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

            train_recorder.append(train_bca)
            test_recorder.append(test_bca)
    return feature_ext, model, train_recorder, test_recorder


def train(x_train, s_train, g_train, e_train, x_test, s_test, g_test, e_test, npz_path: str, repeat: int, args):
    x_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor))
    s_train = Variable(torch.from_numpy(s_train).type(torch.LongTensor))
    g_train = Variable(torch.from_numpy(g_train).type(torch.LongTensor))
    e_train = Variable(torch.from_numpy(e_train).type(torch.LongTensor))

    x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
    s_test = Variable(torch.from_numpy(s_test).type(torch.LongTensor))
    g_test = Variable(torch.from_numpy(g_test).type(torch.LongTensor))
    e_test = Variable(torch.from_numpy(e_test).type(torch.LongTensor))

    # initialize the model
    # gender model
    chans, samples = x_train.shape[2], x_train.shape[3]
    g_feat = LoadModel(model_name=args.feature_c,
                            Chans=chans,
                            Samples=samples)

    g_dis = Discriminator(input_dim=CalculateOutSize(g_feat, chans, samples),
                          n_subjects=len(np.unique(g_train.numpy())))

    g_feat.to(args.device)
    g_dis.to(args.device)

    # identity model
    s_feat = LoadModel(model_name=args.feature_c,
                            Chans=chans,
                            Samples=samples)

    s_dis = Discriminator(input_dim=CalculateOutSize(s_feat, chans, samples),
                          n_subjects=len(np.unique(s_train.numpy())))

    s_feat.to(args.device)
    s_dis.to(args.device)

    # experience model
    e_feat = LoadModel(model_name=args.feature_c,
                            Chans=chans,
                            Samples=samples)

    e_dis = Discriminator(input_dim=CalculateOutSize(e_feat, chans, samples),
                          n_subjects=len(np.unique(e_train.numpy())))

    e_feat.to(args.device)
    e_dis.to(args.device)    

    criterion = nn.CrossEntropyLoss().to(args.device)

    # gender privacy 
    logging.info('*' * 25 + ' train gender classifier ' + '*' * 25)
    g_feat, g_dis, d_train_acc, d_test_acc = trainer(g_feat,
                                                     g_dis,
                                                     x_train,
                                                     g_train,
                                                     x_test,
                                                     g_test,
                                                     phase='d',
                                                     args=args)

    g_feat.eval()
    g_dis.eval()

    dis_loader = DataLoader(dataset=TensorDataset(x_test, g_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, g_acc, g_bca = eval(g_feat, g_dis, criterion, dis_loader)

    logging.info(f'Gender BCA: {g_bca}')

    # identity privacy
    logging.info('*' * 25 + ' train identity classifier ' + '*' * 25)
    s_feat, s_dis, d_train_acc, d_test_acc = trainer(s_feat,
                                                     s_dis,
                                                     x_train,
                                                     s_train,
                                                     x_test,
                                                     s_test,
                                                     phase='d',
                                                     args=args)

    s_feat.eval()
    s_dis.eval()

    dis_loader = DataLoader(dataset=TensorDataset(x_test, s_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, s_acc, s_bca = eval(s_feat, s_dis, criterion, dis_loader)

    logging.info(f'Gender BCA: {s_bca}')

    # experience privacy 
    logging.info('*' * 25 + ' train experience classifier ' + '*' * 25)
    e_feat, e_dis, d_train_acc, d_test_acc = trainer(e_feat,
                                                     e_dis,
                                                     x_train,
                                                     e_train,
                                                     x_test,
                                                     e_test,
                                                     phase='d',
                                                     args=args)

    e_feat.eval()
    e_dis.eval()

    dis_loader = DataLoader(dataset=TensorDataset(x_test, e_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, e_acc, e_bca = eval(e_feat, e_dis, criterion, dis_loader)

    logging.info(f'Experience BCA: {e_bca}')

    # unlearnable noise generation
    # gender perturbation
    g_pert = unlearnable_optim(x_train, g_train, args)
    s_pert = unlearnable_optim(x_train, s_train, args)
    e_pert = unlearnable_optim(x_train, e_train, args)
    protected_train = x_train + (g_pert[g_train] + s_pert[s_train] + e_pert[e_train])
    g_protected_train, s_protected_train, e_protected_train = protected_train, protected_train, protected_train

    # g_protected_train = x_train + g_pert[g_train]
    # e_protected_train = x_train + e_pert[e_train]

    # gender privacy protection 
    logging.info('*' * 25 + ' train gender classifier after protection ' + '*' * 25)
    g_feat.apply(init_weights)
    g_dis.apply(init_weights)
    g_feat, g_dis, d_train_acc, d_test_acc = trainer(g_feat,
                                                     g_dis,
                                                     g_protected_train,
                                                     g_train,
                                                     x_test,
                                                     g_test,
                                                     phase='d',
                                                     args=args)

    g_feat.eval()
    g_dis.eval()

    dis_loader = DataLoader(dataset=TensorDataset(x_test, g_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, p_g_acc, p_g_bca = eval(g_feat, g_dis, criterion, dis_loader)

    logging.info(f'Protected gender BCA: {p_g_bca}')

    # identity privacy protection
    logging.info('*' * 25 + ' train identity classifier after protection ' + '*' * 25)
    s_feat.apply(init_weights)
    s_dis.apply(init_weights)
    s_feat, s_dis, d_train_acc, d_test_acc = trainer(s_feat,
                                                     s_dis,
                                                     s_protected_train,
                                                     s_train,
                                                     x_test,
                                                     s_test,
                                                     phase='d',
                                                     args=args)

    s_feat.eval()
    s_dis.eval()

    dis_loader = DataLoader(dataset=TensorDataset(x_test, s_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, p_s_acc, p_s_bca = eval(s_feat, s_dis, criterion, dis_loader)

    logging.info(f'Protected identity BCA: {p_s_bca}')

    # gender privacy protection 
    logging.info('*' * 25 + ' train experience classifier after protection ' + '*' * 25)
    e_feat.apply(init_weights)
    e_dis.apply(init_weights)
    e_feat, e_dis, d_train_acc, d_test_acc = trainer(e_feat,
                                                     e_dis,
                                                     e_protected_train,
                                                     e_train,
                                                     x_test,
                                                     e_test,
                                                     phase='d',
                                                     args=args)

    e_feat.eval()
    e_dis.eval()

    dis_loader = DataLoader(dataset=TensorDataset(x_test, e_test),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    _, p_e_acc, p_e_bca = eval(e_feat, e_dis, criterion, dis_loader)

    logging.info(f'Protected gender BCA: {p_e_bca}')

    return g_bca, p_g_bca, s_bca, p_s_bca, e_bca, p_e_bca, g_pert.numpy(), s_pert.numpy(), e_pert.numpy()


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
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--batch_size', type=int, default=128) # 128
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--feature_c', type=str, default='EEGNet')
    parser.add_argument('--feature_d', type=str, default='EEGNet')
    parser.add_argument('--log', type=str, default='1e-5')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    model_path = f'model/{args.feature_c}_{args.feature_d}/'

    log_path = f'results/log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(
        log_path, f'protection_{args.feature_c}.log')

    npz_path = f'results/npz/protection_{args.feature_c}_1e-5'

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
    SSVEP_x, SSVEP_y, SSVEP_s, SSVEP_g, SSVEP_e = SSVEP_data['x'], SSVEP_data['y'], SSVEP_data['s'], SSVEP_data['g'], SSVEP_data['e']
    # np.savez('MI.npz', x=MI_x, y=MI_y, s=MI_s, g=MI_g, e=MI_e)
    # np.savez('ERP.npz', x=ERP_x, y=ERP_y, s=ERP_s, g=ERP_g, e=ERP_e)
    # np.savez('SSVEP.npz', x=SSVEP_x, y=SSVEP_y, s=SSVEP_s, g=SSVEP_g, e=SSVEP_e)

    
    # model train
    r_g_bca, r_p_g_bca, r_s_bca, r_p_s_bca, r_e_bca, r_p_e_bca = [], [], [], [], [], []
    for repeat in range(5):
        seed(repeat)
        s_g_bca, s_p_g_bca, s_s_bca, s_p_s_bca, s_e_bca, s_p_e_bca = [], [], [], [], [], []
        s_g_pert, s_s_pert, s_e_pert = [], [], []
        for session in range(2):
            # model train
            model_save_path = os.path.join(model_path, f'{session}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            x_train = np.concatenate([MI_x[session], ERP_x[session], SSVEP_x[session]], axis=0)
            y_train = np.concatenate([MI_y[session], ERP_y[session], SSVEP_y[session]])
            s_train = np.concatenate([MI_s[session], ERP_s[session], SSVEP_s[session]])
            g_train = np.concatenate([MI_g[session], ERP_g[session], SSVEP_g[session]])
            e_train = np.concatenate([MI_e[session], ERP_e[session], SSVEP_e[session]])

            x_test = np.concatenate([MI_x[1-session], ERP_x[1-session], SSVEP_x[1-session]], axis=0)
            y_test = np.concatenate([MI_y[1-session], ERP_y[1-session], SSVEP_y[1-session]])
            s_test = np.concatenate([MI_s[1-session], ERP_s[1-session], SSVEP_s[1-session]])
            g_test = np.concatenate([MI_g[1-session], ERP_g[1-session], SSVEP_g[1-session]])
            e_test = np.concatenate([MI_e[1-session], ERP_e[1-session], SSVEP_e[1-session]])
            
            logging.info(f'train: {len(x_train)}, test:{len(x_test)}')

            g_bca, p_g_bca, s_bca, p_s_bca, e_bca, p_e_bca, g_pert, s_pert, e_pert = train(x_train, s_train, g_train, e_train, x_test, s_test, g_test, e_test, npz_path, repeat, args)

            s_g_bca.append(g_bca)
            s_p_g_bca.append(p_g_bca)
            s_s_bca.append(s_bca)
            s_p_s_bca.append(p_s_bca)
            s_e_bca.append(e_bca)
            s_p_e_bca.append(p_e_bca)
            s_g_pert.append(g_pert)
            s_s_pert.append(s_pert)
            s_e_pert.append(e_pert)

            logging.info(f'session {session + 1}')
            logging.info(f'Mean gender/protected gender bca: {g_bca}/{p_g_bca}')
            logging.info(f'Mean identity/protected gender bca: {s_bca}/{p_s_bca}')
            logging.info(f'Mean experience/protected experience bca: {e_bca}/{p_e_bca}')
        
        r_g_bca.append(s_g_bca)
        r_p_g_bca.append(s_p_g_bca)
        r_s_bca.append(s_s_bca)
        r_p_s_bca.append(s_p_s_bca)
        r_e_bca.append(s_e_bca)
        r_p_e_bca.append(s_p_e_bca)
        logging.info(f'repeat {repeat + 1}')
        logging.info(f'Mean gender/protected gender bca: {np.mean(s_g_bca)}/{np.mean(s_p_g_bca)}')
        logging.info(f'Mean identity/protected gender bca: {np.mean(s_s_bca)}/{np.mean(s_p_s_bca)}')
        logging.info(f'Mean experience/protected experience bca: {np.mean(s_e_bca)}/{np.mean(s_p_e_bca)}')

        np.savez(npz_path + f'/repeat{repeat}.npz',  g_pert=s_g_pert, s_pert=s_s_pert, e_pert=s_e_pert)

    logging.info(f'Avg results')
    logging.info(f'Mean gender/protected gender bca: {np.mean(r_g_bca)}/{np.mean(r_p_g_bca)}')
    logging.info(f'Mean ident/protected gender bca: {np.mean(r_s_bca)}/{np.mean(r_p_s_bca)}')
    logging.info(f'Mean experience/protected experience bca: {np.mean(r_e_bca)}/{np.mean(r_p_e_bca)}')

    np.savez(npz_path + '/result.npz',  gender=r_g_bca, pgender=r_p_g_bca,
             identity=r_s_bca, pidentity=r_p_s_bca, exp=r_e_bca, pexp=r_p_e_bca)
