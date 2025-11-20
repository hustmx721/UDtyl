import os
import sys
sys.path.append((os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from feautils import timedomain, freqdomain, timefreq, nonlinear
import itertools as it
import warnings

from os import walk

# Parameters
fs = 250
# 非线性特征参数
Tau = 4
M = 10
R = 0.3
Band = np.arange(1, fs // 2)
DE = 10

def feature_extraction(data, feature_save, feature_name):
    file_path = os.path.join(os.path.dirname(__file__), f'{feature_name}.npz')
    if feature_save == True:
        # Read files
        x0 = data

        ts = x0.shape[2]

        x0 = np.transpose(x0, (0, 2, 1))  # (sample_num, channel, time_samples) to (sample_num, time_samples, channel)

        # Extract features for inter-ictal states
        time_feature_19_0 = np.zeros((x0.shape[0], 2))
        freq_feature_19_0 = np.zeros((x0.shape[0], 2))
        tf_feature_19_0 = np.zeros((x0.shape[0], 2))
        nonlinear_feature_19_0 = np.zeros((x0.shape[0], 2))

        for c in range(x0.shape[2]):  # channel num
            print(x0.shape)
            print('channel: ', str(c))
            x = x0[:, :, c]
            time_feature = []
            freq_feature = []
            tf_feature = []
            nonlinear_feature = []
            for j in range(x.shape[0]): # sample num
                trans = x[j, :]

                trans = np.array([trans])

                trans = pd.DataFrame(trans, columns=[str(x) for x in range(ts)])

                temp1 = timedomain.timedomain(trans)
                matrix1 = np.array(temp1.time_main(mysteps=5))
                time_feature.append(matrix1)  # time domain features

                temp2 = freqdomain.freqdomain(trans, myfs=fs)
                matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
                freq_feature.append(matrix2)  # frequency domain features

                temp3 = timefreq.timefreq(trans, myfs=fs)
                matrix3 = np.array(temp3.main_tf(smoothwindow=100))  # 100 ; decrease?
                tf_feature.append(matrix3)  # time-frequency domain features

                temp4 = nonlinear.nonlinear(trans, myfs=fs)
                matrix4 = np.array(temp4.nonlinear_main(tau=Tau, m=M, r=R, de=DE, n_perm=4, n_lya=40, band=Band))
                nonlinear_feature.append(matrix4)  # nonlinear analysis
            time_feature = np.array(time_feature)
            freq_feature = np.array(freq_feature)
            tf_feature = np.array(tf_feature)
            nonlinear_feature = np.array(nonlinear_feature)

            time_feature_19_0 = np.append(time_feature_19_0, time_feature, axis=1)
            freq_feature_19_0 = np.append(freq_feature_19_0, freq_feature, axis=1)
            tf_feature_19_0 = np.append(tf_feature_19_0, tf_feature, axis=1)
            nonlinear_feature_19_0 = np.append(nonlinear_feature_19_0, nonlinear_feature, axis=1)
            # 16 17 33 8
        time_feature_19_0 = np.delete(time_feature_19_0, [0, 1], axis=1)
        freq_feature_19_0 = np.delete(freq_feature_19_0, [0, 1], axis=1)
        tf_feature_19_0 = np.delete(tf_feature_19_0, [0, 1], axis=1)
        nonlinear_feature_19_0 = np.delete(nonlinear_feature_19_0, [0, 1], axis=1)
        np.savez(file_path, time=time_feature_19_0, freq=freq_feature_19_0, tf=tf_feature_19_0, entropy=nonlinear_feature_19_0)
        print('features saved!')
        feature = np.concatenate((time_feature_19_0, freq_feature_19_0, tf_feature_19_0, nonlinear_feature_19_0), axis=1)
    elif feature_save == False:
        feature_file = np.load(file_path)
        feature = np.concatenate((feature_file['time'], feature_file['freq'], feature_file['tf'], feature_file['entropy']), axis=1)
    print(feature.shape)
    return feature

