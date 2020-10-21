# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from .utilities import cross_product


class TSLasso:

    def __init__(self, first_stage=ElasticNet(alpha=0.01)):
        self.first_stage = first_stage
        return

    def fit(self, Z, X, Y):
        x_pred = np.zeros(X.shape)
        for i in range(X.shape[1]):
            x_pred[:, i] = clone(self.first_stage).fit(
                Z, X[:, i]).predict(Z)

        self.est = LassoCV(cv=3).fit(x_pred, Y)
        self.coef_ = self.est.coef_
        return self

    def predict(self, X):
        return self.est.predict(X)


class _SparseLinearAdversarialGMM:

    def __init__(self, lambda_theta=0.01, B=100, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, sparsity=None, fit_intercept=True):
        self.B = B
        self.lambda_theta = lambda_theta
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept

    def _check_input(self, Z, X, Y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            Z = np.hstack([np.ones((X.shape[0], 1)), Z])
        return Z, X, Y.flatten()

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.coef_)

    @property
    def coef(self):
        return self.coef_[1:] if self.fit_intercept else self.coef_

    @property
    def intercept(self):
        return self.coef_[0] if self.fit_intercept else 0


class _L1Adversary(_SparseLinearAdversarialGMM):

    def _check_duality_gap(self, Z, X, Y):
        self.max_response_loss_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=np.inf)\
            + self.lambda_theta * np.linalg.norm(self.coef_, ord=1)
        self.min_response_loss_ = self.B * np.clip(self.lambda_theta
                                                   - np.linalg.norm(np.mean(X * np.dot(Z, self.w_).reshape(-1, 1),
                                                                            axis=0),
                                                                    ord=np.inf),
                                                   -np.inf, 0)\
            - np.mean(Y * np.dot(Z, self.w_))
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, Z, X, Y):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (X.shape[0])**(2 / 3))
            filt = (np.abs(self.coef_) < thresh)
            self.coef_[filt] = 0
        self.max_violation_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=np.inf)
        self._check_duality_gap(Z, X, Y)


class SubGradientVsHedge(_L1Adversary):

    def fit(self, Z, X, Y):
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        eta_theta = np.sqrt(
            np.log(d_x + 1) / T) if self.eta_theta == 'auto' else self.eta_theta
        eta_w = np.sqrt(
            np.log(d_z + 1) / T) if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yx = np.mean(Y.reshape(-1, 1) * Z, axis=0)

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.zeros(d_x)
                theta_acc = np.zeros(d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)

            test_fn = np.dot(Z, w[:d_z] - w[d_z:]).reshape(-1, 1)
            pred_fn = np.dot(X, theta).reshape(-1, 1)
            res[:d_z] = np.mean(Z * pred_fn, axis=0) - yx
            res[d_z:] = - res[:d_z]

            theta[:] = theta - eta_theta * \
                (np.mean(test_fn * X, axis=0) +
                 lambda_theta * np.sign(theta))
            theta[:] = np.clip(theta, -self.B, self.B)

            w[:] = w * np.exp(eta_w * res)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t

            if t % 50 == 0:
                self.coef_ = theta_acc
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.coef_ = theta_acc
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)
        return self


class OptimisticHedgeVsOptimisticHedge(_L1Adversary):

    def fit(self, Z, X, Y):
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yx = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z),
                         axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            # quantities for updating theta
            if d_x * d_z < n**2:
                cors_t = xz @ (w[:d_z] - w[d_z:])
            else:
                test_fn = np.dot(Z, w[:d_z] -
                                 w[d_z:]).reshape(-1, 1)
                cors_t = np.mean(test_fn * X, axis=0)
            cors += cors_t

            # quantities for updating w
            if d_x * d_z < n**2:
                res[:d_z] = (theta[:d_x] -
                             theta[d_x:]).T @ xz - yx
            else:
                pred_fn = np.dot(X, theta[:d_x] -
                                 theta[d_x:]).reshape(-1, 1)
                res[:d_z] = np.mean(Z * pred_fn, axis=0) - yx
            res[d_z:] = - res[:d_z]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta *
                                 (cors + cors_t + (t + 1) * lambda_theta))
            theta[d_x:] = np.exp(-1 - eta_theta *
                                 (- cors - cors_t + (t + 1) * lambda_theta))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self


class StochasticOptimisticHedgeVsOptimisticHedge(_L1Adversary):

    def fit(self, Z, X, Y, L=10):
        """
        L : mini-bathc size
        """
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        n = X.shape[0]
        d_x = X.shape[1]
        d_z = Z.shape[1]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            # mini-batch indices
            indsU = t % n
            if indsU + L >= n:
                inds = np.concatenate(
                    (np.arange(indsU, n), np.arange(0, L - (n - indsU))))
            else:
                inds = np.arange(indsU, indsU + L)

            # quantities for updating theta
            test_fn = np.dot(Z[inds], w[:d_z] -
                             w[d_z:]).reshape(-1, 1)
            cors_t = np.mean(test_fn * X[inds], axis=0)
            cors += cors_t

            # quantities for updating w
            pred_fn = np.dot(X[inds], theta[:d_x] -
                             theta[d_x:]).reshape(-1, 1)
            res[:d_z] = np.mean(Z[inds] *
                                (pred_fn - Y[inds].reshape(-1, 1)), axis=0)
            res[d_z:] = - res[:d_z]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta *
                                 (cors + cors_t + (t + 1) * lambda_theta))
            theta[d_x:] = np.exp(-1 - eta_theta *
                                 (- cors - cors_t + (t + 1) * lambda_theta))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self


def prox(x, thres):
    y = x.copy()
    y[x > thres] -= thres
    y[x < - thres] += thres
    y[(x <= thres) & (x >= -thres)] = 0
    return y


class ProxGradientVsHedge(_L1Adversary):

    def fit(self, Z, X, Y):
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yx = np.mean(Y.reshape(-1, 1) * Z, axis=0)

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.zeros(d_x)
                theta_acc = np.zeros(d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            test_fn = np.dot(Z, w[:d_z] - w[d_z:]).reshape(-1, 1)
            cors_t = np.mean(test_fn * X, axis=0)
            cors += cors_t
            pred_fn = np.dot(X, theta).reshape(-1, 1)
            res[:d_z] = np.mean(Z * pred_fn, axis=0) - yx
            res[d_z:] = - res[:d_z]

            theta[:] = prox(- (cors + cors_t) * eta_theta,
                            lambda_theta * eta_theta * (t + 1))
            theta[:] = np.clip(theta, -self.B, self.B)

            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.coef_ = theta_acc
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self


class _L2Adversary(_SparseLinearAdversarialGMM):

    def _check_duality_gap(self, Z, X, Y):
        self.max_response_loss_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=2)\
            + self.lambda_theta * np.linalg.norm(self.coef_, ord=1)
        self.min_response_loss_ = self.B * np.clip(self.lambda_theta
                                                   - np.linalg.norm(np.mean(X * np.dot(Z, self.w_).reshape(-1, 1),
                                                                            axis=0),
                                                                    ord=np.inf),
                                                   -np.inf, 0)\
            - np.mean(Y * np.dot(Z, self.w_))
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, Z, X, Y):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (X.shape[0])**(2 / 3))
            filt = (np.abs(self.coef_) < thresh)
            self.coef_[filt] = 0
        self.max_violation_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=np.inf)
        self._check_duality_gap(Z, X, Y)


class L2SubGradient(_L2Adversary):

    def fit(self, Z, X, Y):
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        eta_theta = np.sqrt(
            np.log(d_x + 1) / T) if self.eta_theta == 'auto' else self.eta_theta
        lambda_theta = self.lambda_theta

        yx = np.mean(Y.reshape(-1, 1) * Z, axis=0)

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.zeros(d_x)
                theta_acc = np.zeros(d_x)
                w = np.zeros(d_z)
                w_acc = np.zeros(d_z)
                res = np.zeros(d_z)

            pred_fn = np.dot(X, theta).reshape(-1, 1)
            res = np.mean(Z * pred_fn, axis=0) - yx
            w[:] = res / np.linalg.norm(res, ord=2)

            test_fn = np.dot(Z, w).reshape(-1, 1)
            theta[:] = theta - eta_theta * \
                (np.mean(test_fn * X, axis=0) +
                 lambda_theta * np.sign(theta))
            theta[:] = np.clip(theta[:], -self.B, self.B)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t

            if t % 50 == 0:
                self.coef_ = theta_acc
                self.w_ = w_acc
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                last_gap = self.duality_gap_

        self.coef_ = theta_acc
        self.w_ = w_acc

        self._post_process(Z, X, Y)

        return self


class L2ProxGradient(_L2Adversary):

    def fit(self, Z, X, Y):
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        eta_theta = np.sqrt(
            np.log(d_x + 1) / T) if self.eta_theta == 'auto' else self.eta_theta
        lambda_theta = self.lambda_theta

        yx = np.mean(Y.reshape(-1, 1) * Z, axis=0)

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.zeros(d_x)
                theta_acc = np.zeros(d_x)
                w = np.zeros(d_z)
                w_acc = np.zeros(d_z)
                res = np.zeros(d_z)
                cors = 0

            pred_fn = np.dot(X, theta).reshape(-1, 1)
            res = np.mean(Z * pred_fn, axis=0) - yx
            w[:] = res / np.linalg.norm(res, ord=2)

            test_fn = np.dot(Z, w).reshape(-1, 1)
            cors_t = np.mean(test_fn * X, axis=0)
            cors += cors_t
            theta[:] = prox(- cors * eta_theta,
                            lambda_theta * eta_theta * t)
            theta[:] = np.clip(theta, -self.B, self.B)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t

            if t % 50 == 0:
                self.coef_ = theta_acc
                self.w_ = w_acc
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                last_gap = self.duality_gap_

        self.coef_ = theta_acc
        self.w_ = w_acc

        self._post_process(Z, X, Y)

        return self


class L2OptimisticHedgeVsOGD(_L2Adversary):

    def fit(self, Z, X, Y):
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = np.sqrt(
            np.log(d_x + 1) / T) if self.eta_theta == 'auto' else self.eta_theta
        eta_w = np.sqrt(
            np.log(d_z + 1) / T) if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yx = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z), axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.zeros(d_z)
                w_acc = np.zeros(d_z)
                res = np.zeros(d_z)
                res_pre = np.zeros(d_z)
                cors = 0

            if d_x * d_z < n**2:
                cors_t = xz @ w
            else:
                test_fn = np.dot(Z, w).reshape(-1, 1)
                cors_t = np.mean(test_fn * X, axis=0)
            cors += cors_t

            if d_x * d_z < n**2:
                res[:] = (theta[:d_x] -
                          theta[d_x:]).T @ xz - yx
            else:
                pred_fn = np.dot(X, theta[:d_x] -
                                 theta[d_x:]).reshape(-1, 1)
                res[:] = np.mean(Z * pred_fn, axis=0) - yx

            theta[:d_x] = np.exp(-1 - eta_theta *
                                 (cors + cors_t + (t + 1) * lambda_theta))
            theta[d_x:] = np.exp(-1 - eta_theta *
                                 (- cors - cors_t + (t + 1) * lambda_theta))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            w[:] = w + 2 * eta_w * res - eta_w * res_pre
            norm_w = np.linalg.norm(w, ord=2)
            w[:] = w / norm_w if norm_w > 1 else w

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc

        self._post_process(Z, X, Y)
        return self
