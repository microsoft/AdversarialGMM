# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sklearn.isotonic import IsotonicRegression
from .curve_fit import project_convex_lip


class _BaseShapeIV:

    def predict(self, X):
        inds = np.searchsorted(self.x_, X[:, 0])
        lb_x = self.x_[np.clip(inds - 1, 0, len(self.coef_) - 1)]
        lb_y = self.coef_[np.clip(inds - 1, 0, len(self.coef_) - 1)]
        ub_x = self.x_[np.clip(inds, 0, len(self.coef_) - 1)]
        ub_y = self.coef_[np.clip(inds, 0, len(self.coef_) - 1)]

        y_pred = lb_y.copy()
        filt = (ub_x != lb_x)
        y_pred[filt] += (ub_y[filt] - lb_y[filt]) * \
            (X[filt, 0] - lb_x[filt]) / (ub_x[filt] - lb_x[filt])
        return y_pred


class ShapeIV(_BaseShapeIV):

    def __init__(self, lambda_w=1, y_min=0, y_max=1, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, monotonic=None):
        self.lambda_w = lambda_w
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.monotonic = monotonic
        self.y_min = y_min
        self.y_max = y_max

    def fit(self, Z, X, Y):
        T = self.n_iter
        assert X.shape[1] == 1
        assert Z.shape[1] == 1
        Y = Y.flatten()
        n = X.shape[0]
        eta_theta = np.sqrt(
            1 / (n * T)) if self.eta_theta == 'auto' else self.eta_theta
        eta_w = np.sqrt(
            1 / (n * T)) if self.eta_w == 'auto' else self.eta_w

        theta_plus = np.zeros((T + 1, n))
        theta_minus = np.zeros((T + 1, n))
        w_plus = np.zeros((T + 1, n))
        w_minus = np.zeros((T + 1, n))
        inds_x = np.argsort(X[:, 0])
        inds_z = np.argsort(Z[:, 0])

        est_theta_plus = IsotonicRegression(
            y_min=self.y_min, y_max=self.y_max, out_of_bounds='clip')
        est_theta_minus = IsotonicRegression(
            y_min=self.y_min, y_max=self.y_max, out_of_bounds='clip', increasing=False)

        est_w_plus = IsotonicRegression(
            y_min=self.y_min - self.y_max, y_max=2 * self.y_max, out_of_bounds='clip')
        est_w_minus = IsotonicRegression(
            y_min=self.y_min - self.y_max, y_max=2 * self.y_max, out_of_bounds='clip', increasing=False)

        for t in np.arange(1, T + 1):

            cor = w_plus[t - 1] - w_minus[t - 1]
            if self.monotonic != 'decreasing':
                theta_plus[t] = est_theta_plus.fit(
                    X[:, 0], theta_plus[t - 1] + eta_theta * cor).predict(X[:, 0])
            if self.monotonic != 'increasing':
                theta_minus[t] = - est_theta_minus.fit(
                    X[:, 0], - (theta_minus[t - 1] - eta_theta * cor)).predict(X[:, 0])

            res = Y - (theta_plus[t - 1] - theta_minus[t - 1])
            reg_w = 2 * self.lambda_w * (w_plus[t - 1] - w_minus[t - 1])
            w_plus[t] = est_w_plus.fit(Z[:, 0], w_plus[t - 1] +
                                       eta_theta * (res - reg_w)).predict(Z[:, 0])
            w_minus[t] = - est_w_minus.fit(Z[:, 0], - (w_minus[t - 1] +
                                                       eta_w * (- res + reg_w))).predict(Z[:, 0])

        self.coef_ = np.mean(theta_plus - theta_minus, axis=0)[inds_x]
        self.x_ = X[inds_x, 0]
        self.w_ = np.mean(w_plus - w_minus, axis=0)[inds_z]
        self.z_ = Z[inds_z, 0]
        self.all_coef_ = theta_plus[:, inds_x] - theta_minus[:, inds_x]
        self.all_w = w_plus[:, inds_z] - w_minus[:, inds_z]

        return self


class LipschitzShapeIV(_BaseShapeIV):

    def __init__(self, L=1, convexity=None, lambda_w=1, y_min=0, y_max=1, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, monotonic=None, n_projection_subsamples=None,
                 max_projection_iters=100):
        self.convexity = convexity
        self.L = L
        self.lambda_w = lambda_w
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.monotonic = monotonic
        self.y_min = y_min
        self.y_max = y_max
        self.n_projection_subsamples = n_projection_subsamples
        self.max_projection_iters = max_projection_iters

    def fit(self, Z, X, Y):
        T = self.n_iter
        assert X.shape[1] == 1
        assert Z.shape[1] == 1
        Y = Y.flatten()
        n = X.shape[0]
        eta_theta = np.sqrt(
            1 / (n * T)) if self.eta_theta == 'auto' else self.eta_theta
        eta_w = np.sqrt(
            1 / (n * T)) if self.eta_w == 'auto' else self.eta_w

        theta_plus = np.zeros((T + 1, n))
        theta_minus = np.zeros((T + 1, n))
        w_plus = np.zeros((T + 1, n))
        w_minus = np.zeros((T + 1, n))
        inds_x = np.argsort(X[:, 0])
        inds_z = np.argsort(Z[:, 0])

        for t in np.arange(1, T + 1):
            cor = w_plus[t - 1] - w_minus[t - 1]
            if self.monotonic != 'decreasing':
                theta_plus[t] = project_convex_lip(
                    X[:, 0], theta_plus[t - 1] + eta_theta * cor,
                    convexity=self.convexity, monotone='increasing',
                    L=self.L, ymin=self.y_min, ymax=self.y_max,
                    n_subsamples=self.n_projection_subsamples, max_iters=self.max_projection_iters)
            if self.monotonic != 'increasing':
                theta_minus[t] = - project_convex_lip(
                    X[:, 0], - (theta_minus[t - 1] - eta_theta * cor),
                    convexity=self.convexity, monotone='decreasing',
                    L=self.L, ymin=self.y_min, ymax=self.y_max,
                    n_subsamples=self.n_projection_subsamples, max_iters=self.max_projection_iters)

            res = Y - (theta_plus[t - 1] - theta_minus[t - 1])
            reg_w = 2 * self.lambda_w * (w_plus[t - 1] - w_minus[t - 1])
            w_plus[t] = project_convex_lip(Z[:, 0], w_plus[t - 1] +
                                           eta_w * (res - reg_w),
                                           convexity=self.convexity, monotone='increasing', L=(2 * self.L if self.L is not None else None),
                                           ymin=self.y_min - self.y_max, ymax=2 * self.y_max,
                                           n_subsamples=self.n_projection_subsamples,
                                           max_iters=self.max_projection_iters)
            w_minus[t] = - project_convex_lip(Z[:, 0], - (w_minus[t - 1] +
                                                          eta_w * (- res + reg_w)),
                                              convexity=self.convexity, monotone='decreasing', L=(2 * self.L if self.L is not None else None),
                                              ymin=(self.y_min - self.y_max), ymax=2 * self.y_max,
                                              n_subsamples=self.n_projection_subsamples,
                                              max_iters=self.max_projection_iters)

        self.coef_ = np.mean(theta_plus - theta_minus, axis=0)[inds_x]
        self.x_ = X[inds_x, 0]
        self.w_ = np.mean(w_plus - w_minus, axis=0)[inds_z]
        self.z_ = Z[inds_z, 0]
        self.all_coef_ = theta_plus[:, inds_x] - theta_minus[:, inds_x]
        self.all_w = w_plus[:, inds_z] - w_minus[:, inds_z]

        return self
