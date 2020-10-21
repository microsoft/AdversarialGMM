# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

# continuously differentiable
fn_dict_cdiff = {'2dpoly': 1, 'sigmoid': 2,
                 'sin': 3, 'frequent_sin': 4,
                 '3dpoly': 7, 'linear': 8}
# continuous but not differentiable
fn_dict_cont = {'abs': 0, 'abs_sqrt': 5, 'rand_pw': 9,
                'abspos': 10, 'sqrpos': 11, 'pwlinear': 15}

# discontinuous
fn_dict_disc = {'step': 6, 'band': 12, 'invband': 13,
                'steplinear': 14}

# monotone
fn_dict_monotone = {'sigmoid': 2,
                    'step': 6, 'linear': 8,
                    'abspos': 10, 'sqrpos': 11, 'pwlinear': 15}

# convex
fn_dict_convex = {'abs': 0, '2dpoly': 1, 'linear': 8,
                  'abspos': 10, 'sqrpos': 11}

# all functions
fn_dict = {'abs': 0, '2dpoly': 1, 'sigmoid': 2,
           'sin': 3, 'frequent_sin': 4, 'abs_sqrt': 5,
           'step': 6, '3dpoly': 7, 'linear': 8, 'rand_pw': 9,
           'abspos': 10, 'sqrpos': 11, 'band': 12, 'invband': 13,
           'steplinear': 14, 'pwlinear': 15}


def generate_random_pw_linear(lb=-2, ub=2, n_pieces=5):
    splits = np.random.choice(np.arange(lb, ub, 0.1),
                              n_pieces - 1, replace=False)
    splits.sort()
    slopes = np.random.uniform(-4, 4, size=n_pieces)
    start = []
    start.append(np.random.uniform(-1, 1))
    for t in range(n_pieces - 1):
        start.append(start[t] + slopes[t] * (splits[t] -
                                             (lb if t == 0 else splits[t - 1])))
    return lambda x: [start[ind] + slopes[ind] * (x - (lb if ind == 0 else splits[ind - 1])) for ind in [np.searchsorted(splits, x)]][0]


def get_tau_fn(func):
    def first(x):
        return x[:, [0]] if len(x.shape) == 2 else x
    # func describes the relation between response and treatment
    if func == fn_dict['abs']:
        def tau_fn(x): return np.abs(first(x))
    elif func == fn_dict['2dpoly']:
        def tau_fn(x): return -1.5 * first(x) + .9 * (first(x)**2)
    elif func == fn_dict['sigmoid']:
        def tau_fn(x): return 2 / (1 + np.exp(-2 * first(x)))
    elif func == fn_dict['sin']:
        def tau_fn(x): return np.sin(first(x))
    elif func == fn_dict['frequent_sin']:
        def tau_fn(x): return np.sin(3 * first(x))
    elif func == fn_dict['abs_sqrt']:
        def tau_fn(x): return np.sqrt(np.abs(first(x)))
    elif func == fn_dict['step']:
        def tau_fn(x): return 1. * (first(x) < 0) + 2.5 * (first(x) >= 0)
    elif func == fn_dict['3dpoly']:
        def tau_fn(x): return -1.5 * first(x) + .9 * \
            (first(x)**2) + first(x)**3
    elif func == fn_dict['linear']:
        def tau_fn(x): return first(x)
    elif func == fn_dict['rand_pw']:
        pw_linear = generate_random_pw_linear()

        def tau_fn(x):
            return np.array([pw_linear(x_i) for x_i in first(x).flatten()]).reshape(-1, 1)
    elif func == fn_dict['abspos']:
        def tau_fn(x): return np.abs(first(x)) * (first(x) >= 0)
    elif func == fn_dict['sqrpos']:
        def tau_fn(x): return (first(x)**2) * (first(x) >= 0)
    elif func == fn_dict['band']:
        def tau_fn(x): return 1.0 * (first(x) >= -.75) * (first(x) <= .75)
    elif func == fn_dict['invband']:
        def tau_fn(x): return 1. - 1. * (first(x) >= -.75) * (first(x) <= .75)
    elif func == fn_dict['steplinear']:
        def tau_fn(x): return 2. * (first(x) >= 0) - first(x)
    elif func == fn_dict['pwlinear']:
        def tau_fn(x):
            q = first(x)
            return (q + 1) * (q <= -1) + (q - 1) * (q >= 1)
    else:
        raise NotImplementedError()

    return tau_fn


def standardize(z, p, y, fn):
    ym = y.mean()
    ystd = y.std()
    y = (y - ym) / ystd

    def newfn(x): return (fn(x) - ym) / ystd
    return z, p, y, newfn


def get_data(n_samples, n_instruments, iv_strength, tau_fn, dgp_num):
    # Construct dataset
    # z:- instruments (features included here, can be high-dimensional)
    # p :- treatments (features included here as well, can be high-dimensional)
    # y :- response (is a scalar always)
    confounder = np.random.normal(0, 1, size=(n_samples, 1))
    z = np.random.normal(0, 1, size=(n_samples, n_instruments))
    fn = tau_fn

    if dgp_num == 1:
        # DGP 1 in the paper
        p = 2 * z[:, [0]] * (z[:, [0]] > 0) * iv_strength \
            + 2 * z[:, [1]] * (z[:, [1]] < 0) * iv_strength \
            + 2 * confounder * (1 - iv_strength) + \
            np.random.normal(0, .1, size=(n_samples, 1))
        y = fn(p) + 2 * confounder + \
            np.random.normal(0, .1, size=(n_samples, 1))
    elif dgp_num == 2:
        # DGP 2 in the paper
        p = 2 * z[:, [0]] * iv_strength \
            + 2 * confounder * (1 - iv_strength) + \
            np.random.normal(0, .1, size=(n_samples, 1))
        y = fn(p) + 2 * confounder + \
            np.random.normal(0, .1, size=(n_samples, 1))
    elif dgp_num == 3:
        # DeepIV's DGP - has feature variables as well
        # z is 3-dimensional: composed of (1) 1D z, (2) t - time unif~(0,10), and (3) s - customer type {1,...,7}
        # y is related to p and z in a complex non-linear, non separable manner
        # p is related to z again in a non-separable manner, rho is endogeneity parameter
        rho = 0.8
        psd = 3.7
        pmu = 17.779
        ysd = 158.
        ymu = -292.1
        z_1 = np.random.normal(0, 1, size=(n_samples, 1))
        v = np.random.normal(0, 1, size=(n_samples, 1))
        t = np.random.uniform(0, 10, size=(n_samples, 1))
        s = np.random.randint(1, 8, size=(n_samples, 1))
        e = rho * v + \
            np.random.normal(0, np.sqrt(1 - rho**2), size=(n_samples, 1))

        def psi(t): return 2 * (np.power(t - 5, 4) / 600 +
                                np.exp(-4 * np.power(t - 5, 2)) + t / 10 - 2)
        p = 25 + (z_1 + 3) * psi(t) + v
        p = (p - pmu) / psd
        g = (10 + p) * s * psi(t) - 2 * p + e
        y = (g - ymu) / ysd
        z = np.hstack((z_1, s, t))
        p = np.hstack((p, s, t))

        def fn(p): return ((10 + p[:, 0]) * p[:, 1]
                           * psi(p[:, 2]) - 2 * p[:, 0] - ymu) / ysd
    elif dgp_num == 4:
        # Many weak Instruments DGP - n_instruments can be very large
        z = np.random.normal(0.5, 1, size=(n_samples, n_instruments))
        p = np.amin(z, axis=1).reshape(-1, 1) * iv_strength + confounder * \
            (1 - iv_strength) + np.random.normal(0, 0.1, size=(n_samples, 1))
        y = fn(p) + 2 * confounder + \
            np.random.normal(0, 0.1, size=(n_samples, 1))
    else:
        # Here we have equal number of treatments and instruments and each
        # instrument affects a separate treatment. Only the first treatment
        # matters for the outcome.
        z = np.random.normal(0, 2, size=(n_samples, n_instruments))
        U = np.random.normal(0, 2, size=(n_samples, 1))
        delta = np.random.normal(0, .1, size=(n_samples, 1))
        zeta = np.random.normal(0, .1, size=(n_samples, 1))
        p = iv_strength * z + (1 - iv_strength) * U + delta
        y = fn(p) + U + zeta

    return standardize(z, p, y, fn)
