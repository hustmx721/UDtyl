import os
import time

from numpy import linalg as la
import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat
from scipy.signal import cheb1ord, cheby1, filtfilt
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from scipy import stats


def trca(X):
    """Task-related component analysis.

    This function implements the method described in [1]_.

    Parameters
    ----------
    X : array, shape=(n_samples, n_chans[, n_trials])
        Training data.

    Returns
    -------
    W : array, shape=(n_chans,)
        Weight coefficients for electrodes which can be used as a spatial
        filter.

    References
    ----------
    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
       "Enhancing detection of SSVEPs for a high-speed brain speller using
       task-related component analysis", IEEE Trans. Biomed. Eng,
       65(1):104-112, 2018.

    """
    n_samples, n_chans, n_trials = theshapeof(X)

    # 1. Compute empirical covariance of all data (to be bounded)
    # -------------------------------------------------------------------------
    # Concatenate all the trials to have all the data as a sequence
    UX = np.zeros((n_chans, n_samples * n_trials))
    for trial in range(n_trials):
        UX[:, trial * n_samples:(trial + 1) * n_samples] = X[..., trial].T

    # Mean centering
    UX -= np.mean(UX, 1)[:, None]

    # Covariance
    Q = UX @ UX.T

    # 2. Compute average empirical covariance between all pairs of trials
    # -------------------------------------------------------------------------
    S = np.zeros((n_chans, n_chans))
    for trial_i in range(n_trials - 1):
        x1 = np.squeeze(X[..., trial_i])

        # Mean centering for the selected trial
        x1 -= np.mean(x1, 0)

        # Select a second trial that is different
        for trial_j in range(trial_i + 1, n_trials):
            x2 = np.squeeze(X[..., trial_j])

            # Mean centering for the selected trial
            x2 -= np.mean(x2, 0)

            # Compute empirical covariance between the two selected trials and
            # sum it
            S = S + x1.T @ x2 + x2.T @ x1

    # 3. Compute eigenvalues and vectors
    # -------------------------------------------------------------------------
    lambdas, W = linalg.eig(S, Q, left=True, right=False)

    # Select the eigenvector corresponding to the biggest eigenvalue
    W_best = W[:, np.argmax(lambdas)]

    return W_best


def trca_regul(X, method):
    """Task-related component analysis.

    This function implements a variation of the method described in [1]_. It is
    inspired by a riemannian geometry approach to CSP [2]_. It adds
    regularization to the covariance matrices and uses the riemannian mean for
    the inter-trial covariance matrix `S`.

    Parameters
    ----------
    X : array, shape=(n_samples, n_chans[, n_trials])
        Training data.

    Returns
    -------
    W : array, shape=(n_chans,)
        Weight coefficients for electrodes which can be used as a spatial
        filter.

    References
    ----------
    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
       "Enhancing detection of SSVEPs for a high-speed brain speller using
       task-related component analysis", IEEE Trans. Biomed. Eng,
       65(1):104-112, 2018.
    .. [2] Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2010,
       October). Common spatial pattern revisited by Riemannian geometry. In
       2010 IEEE International Workshop on Multimedia Signal Processing (pp.
       472-476). IEEE.

    """
    n_samples, n_chans, n_trials = theshapeof(X)

    # 1. Compute empirical covariance of all data (to be bounded)
    # -------------------------------------------------------------------------
    # Concatenate all the trials to have all the data as a sequence
    UX = np.zeros((n_chans, n_samples * n_trials))
    for trial in range(n_trials):
        UX[:, trial * n_samples:(trial + 1) * n_samples] = X[..., trial].T

    # Mean centering
    UX -= np.mean(UX, 1)[:, None]

    # Compute empirical variance of all data (to be bounded)
    cov = Covariances(estimator=method).fit_transform(UX[np.newaxis, ...])
    Q = np.squeeze(cov)

    # 2. Compute average empirical covariance between all pairs of trials
    # -------------------------------------------------------------------------
    # Intertrial correlation computation
    data = np.concatenate((X, X), axis=1)

    # Swapaxes to fit pyriemann Covariances
    data = np.swapaxes(data, 0, 2)
    cov = Covariances(estimator=method).fit_transform(data)

    # Keep only inter-trial
    S = cov[:, :n_chans, n_chans:] + cov[:, n_chans:, :n_chans]

    # If the number of samples is too big, we compute an approximate of
    # riemannian mean to speed up the computation
    if n_trials < 30:
        S = mean_covariance(S, metric="riemann")
    else:
        S = mean_covariance(S, metric="logeuclid")

    # 3. Compute eigenvalues and vectors
    # -------------------------------------------------------------------------
    lambdas, W = linalg.eig(S, Q, left=True, right=False)

    # Select the eigenvector corresponding to the biggest eigenvalue
    W_best = W[:, np.argmax(lambdas)]

    return W_best


def _check_data(X):
    """Check data is numpy array and has the proper dimensions."""
    if not isinstance(X, (np.ndarray, list)):
        raise AttributeError("data should be a list or a numpy array")

    dtype = np.complex128 if np.any(np.iscomplex(X)) else np.float64
    X = np.asanyarray(X, dtype=dtype)
    if X.ndim > 3:
        raise ValueError("Data must be 3D at most")

    return X


def theshapeof(X):
    """Return the shape of X."""
    X = _check_data(X)
    # if not isinstance(X, np.ndarray):
    #     raise AttributeError('X must be a numpy array')

    if X.ndim == 3:
        return X.shape[0], X.shape[1], X.shape[2]
    elif X.ndim == 2:
        return X.shape[0], X.shape[1], 1
    elif X.ndim == 1:
        return X.shape[0], 1, 1
    else:
        raise ValueError("Array contains more than 3 dimensions")


def bandpass(eeg, sfreq, Wp, Ws):
    """Filter bank design for decomposing EEG data into sub-band components.

    Parameters
    ----------
    eeg : np.array, shape=(n_samples, n_chans[, n_trials])
        Training data.
    sfreq : int
        Sampling frequency of the data.
    Wp : 2-tuple
        Passband for Chebyshev filter.
    Ws : 2-tuple
        Stopband for Chebyshev filter.

    Returns
    -------
    y: np.array, shape=(n_trials, n_chans, n_samples)
        Sub-band components decomposed by a filter bank.

    See Also
    --------
    scipy.signal.cheb1ord :
        Chebyshev type I filter order selection.

    """
    # Chebyshev type I filter order selection.
    N, Wn = cheb1ord(Wp, Ws, 3, 40, fs=sfreq)

    # Chebyshev type I filter design
    B, A = cheby1(N=N, rp=0.5, Wn=Wn, btype="bandpass", fs=sfreq)

    # the arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)'
    # correspond to Matlab filtfilt : https://dsp.stackexchange.com/a/47945
    y = filtfilt(B, A, eeg, axis=0, padtype="odd",
                 padlen=3 * (max(len(B), len(A)) - 1))
    return y


class TRCA(object):
    """Task-Related Component Analysis (TRCA).

    References
    ----------
    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
       "Enhancing detection of SSVEPs for a high-speed brain speller using
       task-related component analysis", IEEE Trans. Biomed. Eng,
       65(1):104-112, 2018.
    .. [2] Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2010,
       October). Common spatial pattern revisited by Riemannian geometry. In
       2010 IEEE International Workshop on Multimedia Signal Processing (pp.
       472-476). IEEE.

    """

    def __init__(self, sfreq, filterbank, ensemble=True, method="original",
                 estimator="scm"):
        self.sfreq = sfreq
        self.ensemble = ensemble
        self.filterbank = filterbank
        self.n_bands = len(self.filterbank)
        self.coef_ = None
        self.method = method
        self.estimator = estimator

    def fit(self, X, y):
        """Training stage of the TRCA-based SSVEP detection.

        Parameters
        ----------
        X : array, shape=(n_samples, n_chans[, n_trials])
            Training EEG data.
        y : array, shape=(trials,)
            True label corresponding to each trial of the data array.

        """
        n_samples, n_chans, _ = theshapeof(X)
        classes = np.unique(y)

        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))

        W = np.zeros((self.n_bands, len(classes), n_chans))

        for class_i in classes:  # 要对每一个类别单独做
            # Select data with a specific label
            eeg_tmp = X[..., y == class_i]
            for fb_i in range(self.n_bands):  # filterbank
                # Filter the signal with fb_i
                eeg_tmp = bandpass(eeg_tmp, self.sfreq,
                                   Wp=self.filterbank[fb_i][0],
                                   Ws=self.filterbank[fb_i][1])
                if (eeg_tmp.ndim == 3):
                    # Compute mean of the signal across trials
                    trains[class_i, fb_i] = np.mean(eeg_tmp, -1)
                else:
                    trains[class_i, fb_i] = eeg_tmp
                # Find the spatial filter for the corresponding filtered signal
                # and label
                if self.method == "original":
                    w_best = trca(eeg_tmp)
                elif self.method == "riemann":
                    w_best = trca_regul(eeg_tmp, self.estimator)
                else:
                    raise ValueError("Invalid `method` option.")

                W[fb_i, class_i, :] = w_best  # Store the spatial filter

        self.trains = trains
        self.coef_ = W
        self.classes = classes

        return self

    def predict(self, X):
        """Test phase of the TRCA-based SSVEP detection.

        Parameters
        ----------
        X: array, shape=(n_samples, n_chans[, n_trials])
            Test data.
        model: dict
            Fitted model to be used in testing phase.

        Returns
        -------
        pred: np.array, shape (trials)
            The target estimated by the method.

        """
        if self.coef_ is None:
            raise RuntimeError("TRCA is not fitted")

        # Alpha coefficients for the fusion of filterbank analysis
        fb_coefs = [(x + 1)**(-1.25) + 0.25 for x in range(self.n_bands)]
        _, _, n_trials = theshapeof(X)

        r = np.zeros((self.n_bands, len(self.classes)))
        pred = np.zeros((n_trials), "int")  # To store predictions

        for trial in range(n_trials):
            test_tmp = X[..., trial]  # pick a trial to be analysed
            for fb_i in range(self.n_bands):

                # filterbank on testdata
                testdata = bandpass(test_tmp, self.sfreq,
                                    Wp=self.filterbank[fb_i][0],
                                    Ws=self.filterbank[fb_i][1])

                for class_i in self.classes:
                    # Retrieve reference signal for class i
                    # (shape: n_chans, n_samples)
                    traindata = np.squeeze(self.trains[class_i, fb_i])
                    if self.ensemble:
                        # shape = (n_chans, n_classes)
                        w = np.squeeze(self.coef_[fb_i]).T
                    else:
                        # shape = (n_chans)
                        w = np.squeeze(self.coef_[fb_i, class_i])

                    # Compute 2D correlation of spatially filtered test data
                    # with ref
                    r_tmp = np.corrcoef((testdata @ w).flatten(),
                                        (traindata @ w).flatten())
                    r[fb_i, class_i] = r_tmp[0, 1]

            rho = np.dot(fb_coefs, r)  # fusion for the filterbank analysis

            tau = np.argmax(rho)  # retrieving index of the max
            pred[trial] = int(tau)

        return pred


class TRCA_raw(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.weight = None
        self.temp   = None

    def fit(self, X_ssvep):
        """
        Fit the independent Nf TRCA models using input data tensor `X_ssvep`
        :param
        X_ssvep: 4th order data tensor (Nf x Nc x Ns x Nt)
        """
        assert (
            len(X_ssvep.shape) == 4
        ), "Expected a 4th order data tensor with shape (Nf x Nc x Ns x Nt)"

        Nf, Nc, Ns, Nt = X_ssvep.shape
        self.weight = np.zeros((Nf, Nc))
        self.temp = np.zeros((Nf, Nc, Ns))
        for f in range(Nf):
            X_f = X_ssvep[f]
            w_tmp = self.trca(X_f)
            self.weight[f] = w_tmp[:, 1]
            self.temp[f, :, :] = np.average(X_f, axis=2)

    def transform(self, X_ssvep):
        Nf = X_ssvep.shape[0]
        corr = np.zeros((Nf, Nf))
        for f in range(Nf):
            X_f = X_ssvep[f]
            for class_i in range(Nf):
                temp = self.temp[class_i]
                w = self.weight[class_i, :]
                corr[f, class_i], _ = pearsonr(np.dot(w, X_f), np.dot(w, temp))

        return corr

    def trca(self, x):
        Nt = x.shape[2]
        for trial_i in range(Nt):
            x1 = x[:, :, trial_i]
            x[:, :, trial_i] = x1 - x1.mean(axis=1, keepdims=True)

        SX = np.sum(x, axis=2)
        S = np.dot(SX, SX.T)
        QX = x.reshape(x.shape[0], -1)
        Q = np.dot(QX, QX.T)
        W, V = la.eig(np.dot(la.inv(Q), S))
        # W, V = np.linalg.eig(np.dot(np.linalg.inv(Q), S))

        idx = W.argsort()[::-1]
        V = V[:, idx]

        return V

def normfit(data, ci=0.95):
    """Compute the mean, std and confidence interval for them.

    Parameters
    ----------
    data : array, shape=()
        Input data.
    ci : float
        Confidence interval (default=0.95).

    Returns
    -------
    m : float
        Mean.
    sigma : float
        Standard deviation
    [m - h, m + h] : list
        Confidence interval of the mean.
    [sigmaCI_lower, sigmaCI_upper] : list
        Confidence interval of the std.
    """
    arr = 1.0 * np.array(data)
    num = len(arr)
    avg, std_err = np.mean(arr), stats.sem(arr)
    h_int = std_err * stats.t.ppf((1 + ci) / 2., num - 1)
    var = np.var(data, ddof=1)
    var_ci_upper = var * (num - 1) / stats.chi2.ppf((1 - ci) / 2, num - 1)
    var_ci_lower = var * (num - 1) / stats.chi2.ppf(1 - (1 - ci) / 2, num - 1)
    sigma = np.sqrt(var)
    sigma_ci_lower = np.sqrt(var_ci_lower)
    sigma_ci_upper = np.sqrt(var_ci_upper)

    return avg, sigma, [avg - h_int, avg +
                        h_int], [sigma_ci_lower, sigma_ci_upper]


if __name__ == '__main__':
    path = '/data/xjiang/dataset/Benchmark/processed_new'
    # version2
    segment = 1  # seconds
    n_bands = 1  # number of sub-bands in filter bank analysis
    is_ensemble = True  # True = ensemble TRCA method; False = TRCA method
    method = "original"  # or riemann
    alpha_ci = 0.05  # 100*(1-alpha_ci): confidence interval for accuracy
    sfreq = 250  # sampling rate [Hz]

    ci = 100 * (1 - alpha_ci)
    filterbank = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
                  [(14, 90), (10, 100)],
                  [(22, 90), (16, 100)],
                  [(30, 90), (24, 100)],
                  [(38, 90), (32, 100)],
                  [(46, 90), (40, 100)],
                  [(54, 90), (48, 100)]]

    acc = []
    t = time.time()
    for s in range(35):

        trca_func = TRCA(sfreq, filterbank[0:n_bands], is_ensemble)
        data = loadmat(os.path.join(path, f'S{s}.mat'))

        x = data['x']
        x = x[:, :, :, int((0.5 + 0.14) * sfreq):int((0.5 + 0.14 + segment) * sfreq)]
        n_blocks, n_trials, n_chans, n_samples = x.shape
        eeg = np.reshape(x.transpose(3, 2, 0, 1), (n_samples, n_chans, n_trials * n_blocks))
        labels = np.array([x for x in range(n_trials)] * n_blocks)

        accs = np.zeros(n_blocks)
        for i in range(n_blocks):
            # Select all folds except one for training
            traindata = np.concatenate(
                (eeg[..., :i * n_trials],
                 eeg[..., (i + 1) * n_trials:]), 2)
            y_train = np.concatenate(
                (labels[:i * n_trials], labels[(i + 1) * n_trials:]), 0)

            # Construction of the spatial filter and the reference signals
            trca_func.fit(traindata, y_train)
            # Test stage
            testdata = eeg[..., i * n_trials:(i + 1) * n_trials]
            y_test = labels[i * n_trials:(i + 1) * n_trials]
            estimated = trca_func.predict(testdata)

            # Evaluation of the performance for this fold (accuracy and ITR)
            is_correct = estimated == y_test
            accs[i] = np.mean(is_correct) * 100
        # Mean accuracy and ITR computation
        print(f'N_block acc: {accs}   N_block Avg: {np.mean(accs)}')
        # mu, _, muci, _ = normfit(accs, alpha_ci)
        # print(f"S{s}\nMean accuracy = {mu:.1f}%\t({ci:.0f}% CI: {muci[0]:.1f}-{muci[1]:.1f}%)")  # noqa

        if is_ensemble:
            ensemble = "ensemble TRCA-based method"
        else:
            ensemble = "TRCA-based method"
        acc.append(np.mean(accs))
    print(f'N_block Avg: {np.mean(acc)}')
    print(f"\ntime: {time.time() - t:.1f} seconds")



    # # version1
    # acc = []
    # for s in range(35):
    #     data = loadmat(os.path.join(path, f'S{s}.mat'))
    #
    #     x = np.transpose(data['x'], (1, 2, 3, 0))
    #     y = data['y']
    #
    #     n_blocks = x.shape[3]
    #     accuracy = np.array(range(n_blocks))
    #     for loocv_i in range(n_blocks):
    #         x_train = np.delete(x, loocv_i, 3)
    #         x_test  = x[:, :, :, loocv_i]
    #         y_test  = y[loocv_i]
    #         trca = TRCA_raw()
    #         trca.fit(x_train)
    #         corr = trca.transform(x_test)
    #         y_pred = np.argmax(corr, axis=1)
    #         is_correct = y_test == y_pred
    #         accuracy[loocv_i] = is_correct.mean() * 100
    #     print(accuracy)
    #     acc.append(np.mean(accuracy))
    # print(np.mean(acc))