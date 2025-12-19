""" 
instruction: all kinds of handifeatures for UsrId
methods : 
    - WaveletPacket: 小波包分解特征
    - STFT: 短时傅里叶变换
    - PSD: 功率谱密度和微分熵特征
    - ARMA: 自回归移动平均系数
    - Entropy : 样本熵、近似熵、模糊熵
Author:hust-marx2
time: 2024/1/23
lastest:
"""
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 
import time
import librosa
# how to import dependencies form other conda envs
# sys.path.append("/home/hustmx709/.conda/envs/tyl/lib/python3.10/site-packages")
# from librosa.feature import mfcc


import pywt
import pickle
import EntropyHub as EH
import pmdarima as pm
import numpy as np
from numpy.linalg import eig
from scipy.signal import spectrogram, get_window
from scipy.fftpack import *
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.regression.linear_model import burg
import warnings
warnings.filterwarnings("ignore")

# 对于脑电数据进行小波包分解为五个频段,提取特征
def WaveletPacket(signal):
    trials, channels, _ = signal.shape
    bands = {'delta': [1, 3], 
            'theta': [4, 7],
            'alpha': [8, 13],
            'beta': [14, 30],
            'gamma': [31, 50]}
    levels = [2, 3, 4, 5, 6]  
    features = np.empty((trials,channels,3*len(bands)))
    for m in range(trials):
        for n in range(channels):
            tepdata = np.squeeze(signal[m,n,:])
            coeffs = pywt.wavedec(tepdata, wavelet="db4", level=max(levels), axis=-1)
            for band, level in zip(bands.keys(), levels):
                coeff = coeffs[level] 
                freq_min, freq_max = bands[band]
                # print(f"{band} band: {freq_min} - {freq_max} Hz")               
                # 特征提取
                mean, std = np.mean(coeff), np.std(coeff)
                entropy = np.array([-np.abs(x)**2 * np.log(np.abs(x)**2) for x in coeff]).sum()
                features[m,n, (level-2)*3:(level-1)*3] = [mean,std,entropy]
    return features


# 提取信号的频域谱特征--STFT(short time fourier transform)
def STFT(X, time_length, fs):
    # X.shape = (9,288,22,1000)
    windowsize = int(fs * time_length)
    window = get_window('hann', windowsize)
    nfft = windowsize
    overlap = 0  # option: 0, windowsize/2
    FM_slice = np.zeros((X.shape[0],X.shape[1],X.shape[2],windowsize // 2 + 1)) # (9,288,22,501)
    for id in range(X.shape[0]): # 逐用户
        for p in range(X.shape[1]): # 逐试次
            for k in range(X.shape[2]): # 逐通道
                _, _, S = spectrogram(X[id,p,k],fs=fs,window=window, nperseg=windowsize, noverlap=overlap, nfft=nfft)
                FM_slice[id,p,k] = np.abs(S.squeeze())
    FM_slice = FM_slice.reshape((X.shape[0],X.shape[1],-1)) # (9,288,11022)

    return FM_slice

# 脑电信号的功率谱和微分熵特征
def PSD(data, stft_para):
    """ 
    paras: 
        - data  shape(trial_num,n,m)--n electrodes, m time points
        - stft_para.stftn     frequency domain sampling rate
        - stft_para.fStart    start frequency of each frequency band
        - stft_para.fStop      stop frequency of each frequency band
        - stft_para.EyeTime    EyeTime length of each trail(seconds)
        - stft_para.fs         original frequency 
        
    returns: 
        psd,DE [trial_num*n*l*k]        n electrodes, l windows, k frequency bands 
    """ 
    # initialize the parameters
    # tips:u could input the paras or just use the global variable
    STFTN = stft_para['stftn']
    fStart = stft_para['fStart']
    fStop = stft_para['fStop']
    fs = stft_para['fs']
    EyeTime = stft_para['EyeTime']

    TrialLength = int(fs * EyeTime)   
    Hwindow = np.hanning(TrialLength)

    fStartNum = np.array([int(f / fs * STFTN) for f in fStart])
    fStopNum = np.array([int(f / fs * STFTN) for f in fStop])

    trial_num, n, m = data.shape
    l = m // TrialLength
    k = len(fStart)   
    psd = np.zeros((trial_num, n, l, k))
    de = np.zeros((trial_num, n, l, k))

    for idx in range(trial_num):
        for i in range(l):
            dataNow = data[idx, :, TrialLength * i:TrialLength * (i + 1)] # (n,TrialLength)
            for j in range(n):
                temp = dataNow[j, :]
                Hdata = temp

                FFTdata = fft(Hdata)
                freqs = fftfreq(TrialLength) * STFTN
                magFFTdata = np.abs(FFTdata[0:int(STFTN / 2)]) 

                for p in range(k):
                    E = 0
                    for p0 in range(fStartNum[p],fStopNum[p]):
                        E += magFFTdata[p0] ** 2  # for every freqence point
                    E = E / (fStopNum[p] - fStartNum[p] + 1)
                    psd[idx, j, i, p] = E
                    de[idx, j, i, p] = np.log2(100 * E + 1e-6)

    return psd, de

    
# Extract ARMA model features from EEG data.
def ARMA(data, order:tuple=(2,1,2)):
    """
    Parameters:
    - data: np.ndarray, shape (n_trials, n_channels, n_samples)
    - order: tuple, optional
        The order of the ARMA model (p, d, q).

    Returns:
    - arma_features: np.ndarray, shape (n_trials, n_channels, p+q)
        Extracted ARMA model features for each channel.
        The features include AR coefficient (p elements) and MA coefficient (q elements).
    """
    T, C, S = data.shape # n_trials, n_channels, n_samples
    arma_features = np.empty((T, C, (order[0]+order[-1])))

    """ 
    # 参数寻优,寻找最优p,q
    # from  statsmodels.tsa.arima_model  import  ARIMA
    bic_matrix  =  []  #bic矩阵
    for  p  in  range(pmax+1):
        tmp  =  []
        for  q  in  range(qmax+1): 
            try:
                tmp.append(ARIMA(data["data"],order=(p,1,q)).fit().bic) 
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix  =  pd.DataFrame(bic_matrix)  #从中可以找出最小值
    p,q  =  bic_matrix.stack().idxmin() 
   """
    for trial in range(T):
        for channel in range(C):
            # Fit ARMA model
            arma_model = ARIMA(data[trial, channel, :], order=order, enforce_stationarity=False)
            arma_result = arma_model.fit()
            # Get AR and MA coefficients
            ar_coefs = arma_result.arparams
            ma_coefs = arma_result.maparams
            # Store coefficients in features
            arma_features[trial, channel, :] = np.concatenate([ar_coefs, ma_coefs])

    return arma_features

# extract ar coefs in burg's method
def AR_burg(data:np.ndarray|list,order:int=5,is_win:bool=False,windowlen:int=200,step:int=100):
    T, C, S = data.shape # n_trials, n_channels, n_samples
    if not is_win:
        AR_fea = np.empty((T,C,order))
        for i in range(T):
            for j in range(C):
                AR_fea[i,j], _ = burg(data[i,j],order=order)
    elif is_win:
        win_num = (S - windowlen) // step
        AR_fea = np.empty((T,C,order*win_num))
        for i in range(T):
            for j in range(C):
                for k in range(win_num):
                    tmp = data[i,j,k*step:k*step+windowlen]
                    AR_fea[i,j,k*order:(k+1)*order], _ =  burg(tmp,order=order)
    else:
        raise NotImplementedError
    
    return AR_fea

# multi-class CSP features extracion
def CSP(tx,ty):

    _, chans, _ = tx.shape # (trials,channels,samples)
    ncls = len(np.unique(ty))
    idx = [np.squeeze(np.where(ty == i)) for i in range(ncls)]

    W = []
    for cls in range(ncls):
        # OVM stratege for multi-class
        idx_L = np.array(idx[cls])
        idx_R = np.sort(np.delete(idx,idx[cls]))
        cov_L = np.zeros((len(idx_L),chans,chans)) 
        cov_R = np.zeros((len(idx_R),chans,chans))

        for nL in range(len(idx_L)):
            E = tx[idx_L[nL]]
            EE = np.dot(E, E.T)
            cov_L[nL] = EE / np.trace(EE)
        for nR in range(len(idx_R)):
            E = tx[idx_R[nR]]
            EE = np.dot(E, E.T)
            cov_R[nR] = EE / np.trace(EE)
            
        cov_L = np.mean(cov_L, axis=0)
        cov_R = np.mean(cov_R, axis=0)
        cov_mean = cov_L + cov_R # 平均归一化协方差矩阵

        lam, Uc = eig(cov_mean)
        eigorder = np.argsort(lam)[::-1] # 逆序索引
        lam = lam[eigorder]
        Ut = Uc[:, eigorder]

        Ptmp = np.sqrt(np.diag(np.power(lam, -1)))
        P = np.dot(Ptmp, Ut.T)

        SLL = np.dot(np.dot(P, cov_L), P.T)
        SRR = np.dot(np.dot(P, cov_R), P.T)

        lam_R, BR = eig(SRR)
        erorder = np.argsort(lam_R)
        B = BR[:, erorder]

        w = np.dot(P.T, B)
        W.append(w) # (ncls,chans,chans)

    Wb = [W[cls][:,:4] for cls in range(ncls)] 
    Wb = np.concatenate(Wb, axis=1) # (channels,f*ncls) f为最大f个特征值对应的特征向量

    return Wb

# 提取脑电信号的熵值
def Entropy(data:np.ndarray, r:float=0.2, m:int=2, split="fuzzy", windowlen:int=200, step:int=100): # split = "fuzzy", "sample", "app"
    """ 
    - r: Radius Distance Threshold, a positive scalar
    - m: Embedding Dimension, a positive integer
    - tips : 既有第三方实现也有调用库函数实现
            计算速度方面,发现对于较大的数据量,库计算近似熵和样本熵的速度比numpy矩阵运算速度慢,
            但模糊熵计算速度却比numpy矩阵运算速度快很多;
            故此处实现既有第三方实现,也有库函数实现
    """
    def ApEn2 (data :list|np.ndarray, r:float=0.2, m:int =2):
        data = np.squeeze(data)
        th = r * np.std(data) #容限阈值
        def phi (m):
            n = len(data)
            x = data[ np.arange(n-m+1).reshape(-1,1) + np.arange(m) ]
            ci = lambda xi: (( np.abs(x-xi).max(1) <=th).sum()) / (n-m+1) # 构建一个匿名函数
            c = [ci(xi) for xi in x]
            return np.sum(np.log(c)) /(n-m+1)
        return phi(m) - phi(m+1)
    
    def SampleEntropy(data:list|np.ndarray, r:float=0.2, m:int =2):
        list_len = len(data)  #总长度
        th = r * np.std(data) #容限阈值
        def Phi(k):
            list_split = [data[i:i+k] for i in range(0,list_len-k+(k-m))] #将其拆分成多个子列表
            #这里需要注意，2维和3维分解向量时的方式是不一样的！！！
            Bm = 0.0
            for i in range(0, len(list_split)): #遍历每个子向量
                Bm += ((np.abs(list_split[i] - list_split).max(1) <= th).sum()-1) / (len(list_split)-1) #注意分子和分母都要减1
            return Bm
        ## 多线程
        # x = Pool().map(Phi, [m,m+1])
        # H = - np.log(x[1] / x[0]) 
        H = - np.log(Phi(m+1) / Phi(m))
        return H

    def FuzzyEn(data:list|np.ndarray, r:float=0.2, m:int =2, n:int =2):
        # data:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数; n:模糊函数的指数
        N = len(data)  #总长度
        th = r * np.std(data) #容限阈值

        def Phi(k):
            list_split = [data[i:i+k] for i in range(0,N-k+(k-m))] #将其拆分成多个子列表
            B = np.zeros(len(list_split))
            for i in range(0, len(list_split)): #遍历每个子向量
                di = np.abs(list_split[i] - np.mean(list_split[i]) - list_split + np.mean(list_split,1).reshape(-1,1)).max(1)
                Di = np.exp(- np.power(di,n) / th)
                B[i] = (np.sum(Di) - 1) / (len(list_split)-1) #这里减1是因为要除去其本身，即exp(0)
            return np.sum(B) / len(list_split)
        H = - np.log(Phi(m+1) / Phi(m))
        return H

    th = r * np.std(data)
    N, C, T = data.shape
    windownum = (T-windowlen) // step  + 1
    entrp = np.zeros((N,C,windownum))
    if split == "fuzzy":
        for i in range(N):
            for j in range(C):
                for k in range(windownum):
                    tmp = data[i,j,k*step:k*step+windowlen]
                    entrp[i,j,k] =  EH.FuzzEn(tmp,m,r=(th,2))[0][-1]
                # entrp[i,j] =  FuzzyEn(data[i,j])
    elif split == "sample":
        for i in range(N):
            for j in range(C):
                for k in range(windownum):
                    tmp = data[i,j,k*step:k*step+windowlen]
                    entrp[i,j,k] =  EH.SampEn(tmp,m,r=th)[0][-1]
                # entrp[i,j] =  SampleEntropy(data[i,j])
    elif split == "app":
        for i in range(N):
            for j in range(C):
                for k in range(windownum):
                    tmp = data[i,j,k*step:k*step+windowlen]
                    entrp[i,j,k] =  EH.ApEn(tmp,m,r=th)[0][-1]
                # entrp[i,j] =  ApEn2(data[i,j])
    else:
        raise NotImplementedError
    
    return entrp


# MFCC特征提取函数
def trans_mfccs(wav_data, sample_rate, framesize, mel_band, hop_length):
    wav_data = np.array(wav_data)
    # 使用librosa.feature.mfcc直接调用
    mfccs = librosa.feature.mfcc(y=wav_data, 
                                 sr=sample_rate, 
                                 n_mfcc=mel_band,   # 注意参数含义变化
                                 n_fft=framesize, 
                                 hop_length=hop_length)
    mfccs = mfccs.squeeze()
    return mfccs
