# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
import numpy as np
import scipy


def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))


class _BaseRKHSIV:

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
        '''
        delta -> Critical radius
        '''
        delta_scale = 5 if _check_auto(self.delta_scale) else self.delta_scale
        delta_exp = .4 if _check_auto(self.delta_exp) else self.delta_exp
        return delta_scale / (n**(delta_exp))

    def _get_alpha_scale(self):
        return 60 if _check_auto(self.alpha_scale) else self.alpha_scale

    def _get_alpha_scales(self):
        return ([c for c in np.geomspace(0.1, 1e4, self.n_alphas)]
                if _check_auto(self.alpha_scales) else self.alpha_scales)

    def _get_alpha(self, delta, alpha_scale):
        return alpha_scale * (delta**4)

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)


class RKHSIV(_BaseRKHSIV):

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto',
                 kernel_params=None):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomia kernel
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
            kernel_params : other kernel params passed to the kernel
        """
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5

    def fit(self, Z, T, Y):
        n = Y.shape[0]  # number of samples
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        Kh = self._get_kernel(T)
        Kf = self._get_kernel(Z)

        RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        # M = 2 * Kf @ (np.eye(n) - Kf/(n * delta**2) + Kf @ Kf/(n**2 * delta**4))
        # M = 2 * Kf @ (np.eye(n) - Kf/(n * delta**2))
        # M = Kf
        self.T = T.copy()
        self.a = np.linalg.pinv(Kh @ M @ Kh + alpha * Kh) @ Kh @ M @ Y
        return self

    def predict(self, T_test):
        return self._get_kernel(T_test, Y=self.T) @ self.a

    def score(self, Z, T, Y, delta='auto'):
        n = Y.shape[0]
        delta = self._get_delta(n)
        Kf = self._get_kernel(Z)
        RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        Y_pred = self.predict(T)
        return ((Y - Y_pred).T @ M @ (Y - Y_pred))[0, 0] / n**2


class RKHSIVCV(RKHSIV):

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomia kernel
            kernel_params : other kernel params passed to the kernel
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv

    def _get_alphas(self, delta, scales):
        return [c * (delta**4) for c in scales]

    def fit(self, Z, T, Y):
        n = Y.shape[0]

        Kh = self._get_kernel(T)
        Kf = self._get_kernel(Z)

        RootKf = scipy.linalg.sqrtm(Kf).astype(float)

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Z)):
            M_train = RootKf[np.ix_(train, train)] @ np.linalg.inv(
                Kf[np.ix_(train, train)] / (2 * n_train * (delta_train**2)) + np.eye(len(train)) / 2) @ RootKf[np.ix_(train, train)]
            M_test = RootKf[np.ix_(test, test)] @ np.linalg.inv(
                Kf[np.ix_(test, test)] / (2 * n_test * (delta_test**2)) + np.eye(len(test)) / 2) @ RootKf[np.ix_(test, test)]
            Kh_train = Kh[np.ix_(train, train)]
            KMK_train = Kh_train @ M_train @ Kh_train
            B_train = Kh_train @ M_train @ Y[train]
            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = self._get_alpha(delta_train, alpha_scale)
                a = np.linalg.pinv(KMK_train + alpha * Kh_train) @ B_train
                norm_squared = (a.T @ Kh_train @ a)[0, 0]
                res = Y[test] - Kh[np.ix_(test, train)] @ a
                scores[it].append((res.T @ M_test @ res)[
                                  0, 0] / (res.shape[0]**2))

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        # M = 2 * Kf @ (np.eye(n) - Kf/(n * delta**2) + Kf @ Kf/(n**2 * delta**4))
        # M = 2 * Kf @ (np.eye(n) - Kf/(n * delta**2))
        # M = Kf

        self.T = T.copy()
        self.a = np.linalg.pinv(
            Kh @ M @ Kh + self.best_alpha * Kh) @ Kh @ M @ Y
        return self


class ApproxRKHSIV(_BaseRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        """
        Parameters:
            kernel_approx : what approximator to use; either 'nystrom' or 'rbfsampler' (for kitchen sinks)
            n_components : how many approximation components to use
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomia kernel
            kernel_params : other kernel params passed to the kernel
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5

    def _get_new_approx_instance(self):
        if (self.kernel_approx == 'rbfsampler') and (self.kernel == 'rbf'):
            return RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=1)
        elif self.kernel_approx == 'nystrom':
            return Nystroem(kernel=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree, kernel_params=self.kernel_params,
                            random_state=1, n_components=self.n_components)
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, Z, T, Y):
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())
        self.featZ = self._get_new_approx_instance()
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance()
        RootKh = self.featT.fit_transform(T)
        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ Y
        self.a = np.linalg.pinv(W) @ B
        self.fitted_delta = delta
        return self

    def predict(self, T):
        return self.featT.transform(T) @ self.a

    def score(self, Z, T, Y, delta='auto'):
        n = Y.shape[0]
        delta = self._get_delta(n)
        featZ = self._get_new_approx_instance()
        RootKf = featZ.fit_transform(Z)
        RootKh = self.featT.fit_transform(T)
        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        Y_pred = self.predict(T)
        res = RootKf.T @ (Y - Y_pred)
        return (res.T @ Q @ res)[0, 0] / n**2


class ApproxRKHSIVCV(ApproxRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomia kernel
            kernel_params : other kernel params passed to the kernel
            n_components : how many nystrom components to use
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv

    def fit(self, Z, T, Y):
        n = Y.shape[0]

        self.featZ = self._get_new_approx_instance()
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance()
        RootKh = self.featT.fit_transform(T)

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Z)):
            RootKf_train, RootKf_test = RootKf[train], RootKf[test]
            RootKh_train, RootKh_test = RootKh[train], RootKh[test]
            Q_train = np.linalg.pinv(
                RootKf_train.T @ RootKf_train / (2 * n_train * (delta_train**2)) + np.eye(self.n_components) / 2)
            Q_test = np.linalg.pinv(
                RootKf_test.T @ RootKf_test / (2 * n_test * (delta_test**2)) + np.eye(self.n_components) / 2)
            A_train = RootKh_train.T @ RootKf_train
            AQA_train = A_train @ Q_train @ A_train.T
            B_train = A_train @ Q_train @ RootKf_train.T @ Y[train]
            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = self._get_alpha(delta_train, alpha_scale)
                a = np.linalg.pinv(AQA_train + alpha *
                                   np.eye(self.n_components)) @ B_train
                res = RootKf_test.T @ (Y[test] - RootKh_test @ a)
                scores[it].append((res.T @ Q_test @ res)[
                                  0, 0] / (len(test)**2))

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + self.best_alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ Y
        self.a = np.linalg.pinv(W) @ B
        self.fitted_delta = delta
        return self
