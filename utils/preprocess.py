""" 
instruction:some data preprocessing methods
Author:hust-marx2
time: 2023/9/4
lastest:some little change of function
args: postdatalen--pdl,处理后数据长度;strip--step,数据切分时间间隔;dataroot--data_root,数据位置
"""
import os
import numpy as np
from scipy import signal
import math
from scipy import signal
import mne
from scipy.linalg import fractional_matrix_power
from numpy.linalg import eig
from sklearn.decomposition import PCA
import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 数据预处理，降采样、去趋势化、巴特沃斯带通滤波
def preprocess(data, fs, freq_end1= 8, freq_end2= 32):
    filted_data = np.array(data) # 通道选择，第一维度为trails
    # filted_data = mne.filter.resample(filted_data,down = 4,axis=-1) # 降采样

    wn1 = 2 * freq_end1 / fs # fs为降采样后的频率
    wn2 = 2 * freq_end2 / fs

    filted_data = signal.detrend(filted_data,axis= -1, type="linear") # 去趋势化
    b,a = signal.butter(6,[wn1, wn2],"bandpass")
    fda = signal.filtfilt(b,a,filted_data,axis=-1) # 5阶巴特沃斯滤波
    return fda

#==============preprocessing=======================#
class preprocessing():
    def __init__(self,fs,low_freq=8, high_freq=32):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
    
    def Resample(self,factor,data): #data(trials,C,T)
        filtedData = mne.filter.resample(data, down=factor, axis=-1)
        return filtedData
    
    def Detrend(self,data):
        filtedData = signal.detrend(data, axis=-1, type='linear')
        return filtedData
    
    def Commonref(self,data):
        # 对于所有通道进行平均
        data = data - data.mean(-2, keepdims=True)
        return data

    def Bandpassfilter(self, data, Fstop1:int=8, Fstop2:int=32):
        b, a = signal.butter(6, [2.0 * Fstop1 / self.fs, 2.0 * Fstop2 / self.fs], 'bandpass')  # 5阶巴特沃斯滤波器
        filtedData = signal.filtfilt(b, a, data, axis=-1)
        return filtedData
    
    def Notchfilter(self,data):
        w0 = 2.0 * 50 / self.fs
        a, b = signal.iirnotch(w0, self.fs)
        filtedData = signal.filtfilt(b, a, data, axis=-1)
        return filtedData

    def Normalize(self, data):
        data = (data - data.mean((1, 2), keepdims=True))/data.std((1,2), keepdims=True)
        return data
    
    def SlideWindowsAugment(self, x, y, window_length, stride):
        data = []
        repeats = 0
        i = window_length
        while 1:
            if i > x.shape[-1]:
                break
            data.append(x[:, :, i-window_length:i])
            i = i + stride
            repeats += 1

        data = np.concatenate(data)
        label = np.tile(y, repeats)
        return data, label

    def EA(self, x, ref=False):
        cov = np.zeros((x.shape[0], x.shape[1], x.shape[1])) #(bs,channel,channel)
        for i in range(x.shape[0]):
            cov[i] = np.cov(x[i])
        refEA = np.mean(cov, 0)
        sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])
        XEA = np.zeros(x.shape)
        for i in range(x.shape[0]):
            XEA[i] = np.dot(sqrtRefEA, x[i])
        if ref:
            return XEA, sqrtRefEA
        else:
            return XEA
    
    def torch_EA(self, x, ref=False):
        # Ensure x is a torch.Tensor and move to GPU
        x = torch.from_numpy(x.copy()).to(torch.float32).to('cuda:0')
        bs, channels = x.shape[0], x.shape[1]
        cov = torch.zeros((bs, channels, channels), device=x.device)
        for i in range(bs):
            cov[i] = torch.cov(x[i])
        
        # Calculate the reference EA and its matrix power
        refEA = torch.mean(cov, dim=0).cpu().numpy()
        sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (1e-8) * np.eye(channels)
        sqrtRefEA = torch.from_numpy(sqrtRefEA).to(x.device).to(torch.float32)
        # Perform the transformation
        XEA = torch.zeros_like(x)
        for i in range(bs):
            XEA[i] = torch.matmul(sqrtRefEA, x[i])
        XEA = XEA.cpu().detach().numpy()
        sqrtRefEA = sqrtRefEA.cpu().detach().numpy()
        torch.cuda.empty_cache()
        # Return the transformed data (and sqrtRefEA if ref is True)
        if ref:
            return XEA, sqrtRefEA
        else:
            return XEA

    
    def EEGpipline(self, x):
        # x = self.Commonref(x)
        # x = self.Detrend(x)
        x = self.Bandpassfilter(x, self.low_freq, self.high_freq)
        # x = self.Notchfilter(x)
        # x = self.Normalize(x)
        # x = self.EA(x)
        # x = self.torch_EA(x)
        return x
#==============preprocessing=======================#


#=================CSP==========================================#
"""
Used to calculate the common spatial pattern filter for four-class classification
"""
def csp(data_train, label_train):

    channle_num=data_train.shape[2] #(trials,points,channels)
    idx_0 = np.squeeze(np.where(label_train == 0))
    idx_1 = np.squeeze(np.where(label_train == 1))
    idx_2 = np.squeeze(np.where(label_train == 2))

    W = []
    for n_class in range(3):
        if n_class == 0:
            idx_L = idx_0
            idx_R = np.concatenate((idx_1, idx_2))
        elif n_class == 1:
            idx_L = idx_1
            idx_R = np.concatenate((idx_0, idx_2))
        elif n_class == 2:
            idx_L = idx_2
            idx_R = np.concatenate((idx_0, idx_1))

        idx_R = np.sort(idx_R)
        Cov_L = np.zeros([channle_num, channle_num, len(idx_L)])
        Cov_R = np.zeros([channle_num, channle_num, len(idx_R)])

        for nL in range(len(idx_L)): # 遍历所有trials
            E = data_train[idx_L[nL], :, :]
            EE = np.dot(E.transpose(), E)
            Cov_L[:, :, nL] = EE / np.trace(EE)
        for nR in range(len(idx_R)):
            E = data_train[idx_R[nR], :, :]
            EE = np.dot(E.transpose(), E)
            Cov_R[:, :, nR] = EE / np.trace(EE)

        Cov_L = np.mean(Cov_L, axis=2)
        Cov_R = np.mean(Cov_R, axis=2)
        CovTotal = Cov_L + Cov_R

        lam, Uc = eig(CovTotal)
        eigorder = np.argsort(lam)
        eigorder = eigorder[::-1]
        lam = lam[eigorder]
        Ut = Uc[:, eigorder]

        Ptmp = np.sqrt(np.diag(np.power(lam, -1)))
        P = np.dot(Ptmp, Ut.transpose())

        SL = np.dot(P, Cov_L)
        SLL = np.dot(SL, P.transpose())
        SR = np.dot(P, Cov_R)
        SRR = np.dot(SR, P.transpose())

        lam_R, BR = eig(SRR)
        erorder = np.argsort(lam_R)
        B = BR[:, erorder]

        w = np.dot(P.transpose(), B)
        W.append(w)

    Wb = np.concatenate((W[0][:, 0:4], W[1][:, 0:4], W[2][:, 0:4]), axis=1)
    # The original one is two use the first and last r row, I just use the first 2r.
    # Not significant difference, 2r could be better.

    return Wb





