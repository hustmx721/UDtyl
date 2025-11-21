import scipy.linalg as la
import scipy.io as scio
import numpy as np
import pandas as pd


def split(x, y, ratio=0.8, shuffle=True):
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


def DataLoad(dataset='MI', isEA=False):
    data_path = f'../EEG_data/OpenBMI/processed/{dataset}/'
    info_path = '../EEG_data/OpenBMI/100542/Questionnaire_results.csv'
    info_data = data = pd.read_csv(info_path)
    x, y, s, g, e = [], [], [], [], []
    for session in [1, 2]:
        x_session, y_session, s_session, g_session, e_session = [], [], [], [], []
        for i in range(54):
            data = scio.loadmat(data_path + f's{i}_session{session}.mat')
            x_temp, y_temp = data['x'], data['y']
            x_temp, y_temp = x_temp.reshape(-1, x_temp.shape[-2], x_temp.shape[-1]), y_temp.reshape(-1)
            if isEA:
                x_temp = align(x_temp.squeeze())
            x_temp = x_temp.astype(np.float32)
            x_temp = standard_normalize(x_temp)

            if dataset == 'ERP':
                idx = np.random.permutation(np.arange(len(x_temp)))
                x_temp, y_temp = x_temp[idx[:200]], y_temp[idx[:200]]

            x_temp = x_temp[:, np.newaxis, :, :]

            u_data = info_data[f'subject{i+1}'].values
            x_session.append(x_temp)
            y_session.append(y_temp)
            s_session.extend([i] * len(x_temp))
            g_session.extend([int(u_data[5])] * len(x_temp))
            e_session.extend([1 if int(u_data[6]) > 0 else 0] * len(x_temp))
        x_session = np.concatenate(x_session, axis=0)
        y_session = np.concatenate(y_session)
        s_session = np.array(s_session)
        g_session = np.array(g_session)
        e_session = np.array(e_session)
        x.append(x_session)
        y.append(y_session)
        s.append(s_session)
        g.append(g_session)
        e.append(e_session)

    return x, y, s, g, e


def InfoLoad():
    data_path = '/mnt/data2/mlb/EEG_data/OpenBMI/100542/Questionnaire_results.csv'
    data = pd.read_csv(data_path)

    sex_list = []
    for i in range(54):
        uid = f'subject{i+1}'
        u_data = data[uid].values
        sex_list.append(int(u_data[5]))

    return sex_list


