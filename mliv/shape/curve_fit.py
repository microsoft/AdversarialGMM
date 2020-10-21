# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import cvxopt


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, *, max_iters=100):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = max_iters
    sol = cvxopt.solvers.qp(*args)
    # if 'optimal' not in sol['status']:
    #     return None
    return np.array(sol['x']).reshape((P.shape[1],))


def summarize(T, Y, ints=10):
    q = np.percentile(T, np.linspace(0, 100, ints), interpolation='nearest')
    #q = np.linspace(np.min(T), np.max(T), ints)
    M = np.zeros((T.shape[0], ints))
    lb = np.searchsorted(q, T, side='right') - 1
    ub = np.clip(lb + 1, 0, ints - 1)
    all_inds = np.arange(T.shape[0]).astype(int)
    filt = (q[ub] != q[lb])
    M[all_inds[filt], lb[filt]] = (
        q[ub[filt]] - T[filt]) / (q[ub[filt]] - q[lb[filt]])
    M[all_inds[filt], ub[filt]] = (
        T[filt] - q[lb[filt]]) / (q[ub[filt]] - q[lb[filt]])
    filt = (q[ub] == q[lb])
    M[all_inds[filt], lb[filt]] = 1
    return M, q


def project_convex_lip(T, Y, convexity=None, L=None,
                       monotone=None, ymin=None, ymax=None, epsilon=0,
                       subgradient_penalty=0.00, n_subsamples=None, max_iters=100):
    T = T.flatten()
    Y = Y.flatten()
    n = T.shape[0]
    inds = np.argsort(T)
    T = T[inds]
    Y = Y[inds]
    if n_subsamples is not None:
        M, T = summarize(T, Y, ints=n_subsamples)
        n = n_subsamples
    else:
        M = np.eye(n)
    if convexity is not None:
        P = np.zeros((2 * n, 2 * n))
        P[:n, :n] = np.dot(M.T, M)
        P[n:, n:] = subgradient_penalty * np.eye(n)
        q = np.zeros(2 * n)
        q[:n] = -np.dot(M.T, Y)
        G1 = np.zeros((n - 1, 2 * n))
        h1 = np.zeros(n - 1) + epsilon
        G2 = np.zeros((n - 1, 2 * n))
        h2 = np.zeros(n - 1) + epsilon
    else:
        P = np.dot(M.T, M)
        q = -np.dot(M.T, Y)

    if (L is not None):
        G1b = np.zeros((n - 1, P.shape[1]))
        h1b = np.zeros(n - 1)
        G2b = np.zeros((n - 1, P.shape[1]))
        h2b = np.zeros(n - 1)
    if monotone == 'increasing':
        G1b = np.zeros((n - 1, P.shape[1]))
        h1b = np.zeros(n - 1)
    if monotone == 'decreasing':
        G2b = np.zeros((n - 1, P.shape[1]))
        h2b = np.zeros(n - 1)

    for i in range(n - 1):
        if convexity == 'convex':
            G1[i, i] = 1
            G1[i, i + 1] = -1
            G1[i, n + i] = T[i + 1] - T[i]
        elif convexity == 'concave':
            G1[i, i] = -1
            G1[i, i + 1] = 1
            G1[i, n + i] = - (T[i + 1] - T[i])
        if (L is not None) or (monotone == 'increasing'):
            G1b[i, i] = 1
            G1b[i, i + 1] = -1
            if monotone == 'increasing':
                h1b[i] = 0
            else:
                h1b[i] = L * (T[i + 1] - T[i]) + epsilon

    for i in range(n - 1):
        if convexity == 'convex':
            G2[i, i] = -1
            G2[i, i + 1] = 1
            G2[i, n + i + 1] = T[i] - T[i + 1]
        elif convexity == 'concave':
            G2[i, i] = 1
            G2[i, i + 1] = -1
            G2[i, n + i + 1] = - (T[i] - T[i + 1])
        if (L is not None) or (monotone == 'decreasing'):
            G2b[i, i] = -1
            G2b[i, i + 1] = 1
            if monotone == 'decreasing':
                h2b[i] = 0
            else:
                h2b[i] = L * (T[i + 1] - T[i]) + epsilon

    G = None
    h = None
    if convexity is not None:
        if (L is not None):
            G = np.vstack([G1, G1b, G2, G2b])
            h = np.concatenate((h1, h1b, h2, h2b))
        elif monotone == 'increasing':
            G = np.vstack([G1, G1b, G2])
            h = np.concatenate((h1, h1b, h2))
        elif monotone == 'decreasing':
            G = np.vstack([G1, G2, G2b])
            h = np.concatenate((h1, h2, h2b))
        else:
            G = np.vstack([G1, G2])
            h = np.concatenate((h1, h2))
    else:
        if (L is not None):
            G = np.vstack([G1b, G2b])
            h = np.concatenate((h1b, h2b))
        elif monotone == 'increasing':
            G = G1b
            h = h1b
        elif monotone == 'decreasing':
            G = G2b
            h = h2b

    if ymin is not None:
        LB = np.zeros((n, G.shape[1]))
        LB[:n, :n] = - np.eye(n)
        if G is not None:
            G = np.vstack([G, LB])
            h = np.concatenate((h, - np.ones(n) * ymin))
        else:
            G = LB
            h = - np.ones(n) * ymin
    if ymax is not None:
        UB = np.zeros((n, G.shape[1]))
        UB[:n, :n] = np.eye(n)
        if G is not None:
            G = np.vstack([G, UB])
            h = np.concatenate((h, np.ones(n) * ymax))
        else:
            G = UB
            h = np.ones(n) * ymax
    sol = cvxopt_solve_qp(P, q, G, h, max_iters=max_iters)
    coef = np.zeros(M.shape[0])
    coef[inds] = np.dot(M, sol[:n])
    return coef
