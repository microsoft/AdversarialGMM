# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, LinearRegression,\
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import mliv.dgps as dgps
from mliv.ensemble import EnsembleIV, EnsembleIVStar
from mliv.rkhs import ApproxRKHSIVCV
from mliv.shape import LipschitzShapeIV, ShapeIV
from mliv.linear import OptimisticHedgeVsOptimisticHedge, StochasticOptimisticHedgeVsOptimisticHedge
from mliv.linear import L2OptimisticHedgeVsOGD, L2ProxGradient
from sklearn.pipeline import Pipeline
from mcpy.utils import filesafe


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def gen_data(opts):
    """
    opts : the dgp_opts from the config file
    """
    tau_fn = dgps.get_tau_fn(opts['fn'])
    z, p, y, tau_fn = dgps.get_data(opts['n_samples'], opts['n_instruments'],
                                    opts['iv_strength'], tau_fn, opts['dgp_num'])

    if opts['gridtest']:
        T_test = np.zeros((opts['n_test'], p.shape[1]))
        T_test += np.median(p, axis=0, keepdims=True)
        T_test[:, 0] = np.linspace(np.percentile(
            p[:, 0], 5), np.percentile(p[:, 0], 95), 1000)
    else:
        _, T_test, _, _ = dgps.get_data(opts['n_test'], opts['n_instruments'],
                                        opts['iv_strength'], tau_fn, opts['dgp_num'])
        T_test = T_test[np.argsort(T_test[:, 0])]
    expected_te = tau_fn(T_test)

    # data, true_param
    return (T_test, z, p, y), expected_te


def ivfit(data, model, fitargs=None):
    T_test, Z, T, Y = data
    if fitargs is not None:
        model.fit(Z, T, Y, **fitargs)
    else:
        model.fit(Z, T, Y)
    y_pred = model.predict(T_test)
    return y_pred.reshape(T_test.shape[:1] + Y.shape[1:])


def nystromrkhsfit(data, opts):
    """
    data: the data returned by gen_data
    opts: the method_opts from the config file
    """
    alpha_scales = np.geomspace(1, 10000, 10)
    model = ApproxRKHSIVCV(kernel_approx='nystrom', n_components=_get(opts, 'nstrm_n_comp', 100),
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=alpha_scales, cv=5)
    return ivfit(data, model)


def ensembleiv(data, opts):
    model = EnsembleIV(n_iter=200, max_abs_value=2)
    return ivfit(data, model)


def ensemblestariv(data, opts):
    model = EnsembleIVStar(n_iter=200, max_abs_value=2)
    return ivfit(data, model)


def tviv(data, opts):
    n_iter = 5000
    eta_theta = 2 / np.sqrt(n_iter)
    eta_w = 2 / np.sqrt(n_iter)
    est = ShapeIV(n_iter=n_iter, eta_theta=eta_theta, eta_w=eta_w,
                  y_min=-5, y_max=5, lambda_w=2, monotonic=_get(opts, 'shiv_mon', None))
    return ivfit(data, est)


def lipschitztviv(data, opts):
    n_iter = 5000
    eta_theta = 2 / np.sqrt(n_iter)
    eta_w = 2 / np.sqrt(n_iter)
    est = LipschitzShapeIV(L=_get(opts, 'shiv_L', 2), convexity=None, n_iter=n_iter,
                           eta_theta=eta_theta, eta_w=eta_w,
                           y_min=-5, y_max=5, lambda_w=2,
                           n_projection_subsamples=50, max_projection_iters=20)
    return ivfit(data, est)


def convexiv(data, opts):
    n_iter = 5000
    eta_theta = 2 / np.sqrt(n_iter)
    eta_w = 2 / np.sqrt(n_iter)
    est = LipschitzShapeIV(L=_get(opts, 'shiv_L', 2), convexity='convex', n_iter=n_iter,
                           eta_theta=eta_theta, eta_w=eta_w,
                           y_min=-5, y_max=5, lambda_w=2,
                           n_projection_subsamples=50, max_projection_iters=20)
    return ivfit(data, est)


def l1sparselinear(data, opts):
    T_test, Z, T, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = OptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=_get(opts, 'lin_l1', .05),
                                             eta_theta=.1,
                                             eta_w=.1,
                                             n_iter=_get(opts, 'lin_nit', 10000), tol=.0001, sparsity=None)
    model.fit(trans.fit_transform(Z), trans.fit_transform(T), Y)
    return model.predict(trans.fit_transform(T_test)).reshape(T_test.shape[:1] + Y.shape[1:])


def stochasticl1sparselinear(data, opts):
    T_test, Z, T, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = StochasticOptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=_get(opts, 'lin_l1', .05),
                                                       eta_theta=.05, eta_w=.05,
                                                       n_iter=_get(opts, 'lin_nit', 20000), tol=0.0001)
    model.fit(trans.fit_transform(Z), trans.fit_transform(T), Y, L=100)
    return model.predict(trans.fit_transform(T_test)).reshape(T_test.shape[:1] + Y.shape[1:])


def l2sparselinear(data, opts):
    T_test, Z, T, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = L2OptimisticHedgeVsOGD(B=3, tol=0.0001, lambda_theta=_get(opts, 'lin_l1', .05),
                                   n_iter=_get(opts, 'lin_nit', 20000), eta_theta=.001, eta_w=.001, sparsity=None)
    model.fit(trans.fit_transform(Z), trans.fit_transform(T), Y)
    return model.predict(trans.fit_transform(T_test)).reshape(T_test.shape[:1] + Y.shape[1:])


def tsls(data, opts):
    T_test, Z, T, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    polyT = trans.fit_transform(T)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', LinearRegression())])
    first.fit(Z, polyT)
    second = LinearRegression()
    second.fit(first.predict(Z), Y)
    polyT_test = trans.fit_transform(T_test)
    return second.predict(polyT_test)


def regtsls(data, opts):
    T_test, Z, T, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    polyT = trans.fit_transform(T)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', MultiTaskElasticNetCV(cv=3))])
    first.fit(Z, polyT)
    second = ElasticNetCV(cv=3)
    second.fit(first.predict(Z), Y.ravel())
    polyT_test = trans.fit_transform(T_test)
    return second.predict(polyT_test).reshape(T_test.shape[:1] + Y.shape[1:])


def mse(param_estimates, true_params):
    return np.mean((np.array(param_estimates) - np.array(true_params))**2)


def rsquare(param_estimates, true_params):
    return 1 - mse(param_estimates, true_params) / np.var(true_params)


def _key(dic, value):
    return list(iter(dic.keys()))[np.argwhere(np.array(list(iter(dic.values()))) == value)[0, 0]]


def print_metrics(param_estimates, metric_results, config):
    out = open(os.path.join(config['target_dir'],
                            'print_metrics.csv'), 'a')
    methods = list(next(iter(metric_results.values())).keys())
    metrics = list(
        next(iter(next(iter(metric_results.values())).values())).keys())
    print(config['param_str'], file=out)
    for metric_name in metrics:
        if metric_name != 'raw':
            print(metric_name, file=out)
            print("&", "&".join(methods), file=out)
            for dgp_name, mdgp in metric_results.items():
                print(dgp_name, _key(dgps.fn_dict,
                                     config['dgp_opts']['fn']), end=" ", file=out)
                for method_name in mdgp.keys():
                    res = mdgp[method_name][metric_name]
                    mean_res = res.mean()
                    std_res = res.std() / np.sqrt(len(res))
                    print(r"& {:.3f} $\pm$ {:.3f}".format(
                        mean_res, 2 * std_res), end=" ", file=out)
            print(" ", file=out)
    out.close()
    return
