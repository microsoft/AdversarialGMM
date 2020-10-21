# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from functools import reduce


def cross_product(*XS):
    """
    Compute the cross product of features.

    Parameters
    ----------
    X1 : n x d1 matrix
        First matrix of n samples of d1 features
        (or an n-element vector, which will be treated as an n x 1 matrix)
    X2 : n x d2 matrix
        Second matrix of n samples of d2 features
        (or an n-element vector, which will be treated as an n x 1 matrix)

    Returns
    -------
    A : n x (d1*d2*...) matrix
        Matrix of n samples of d1*d2*... cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0]*..., ..., X1[t,d1-1]*X2[t,0]*..., X1[t,0]*X2[t,1]*..., ..., X1[t,d1-1]*X2[t,1]*..., ...]

    """
    for X in XS:
        assert 2 >= np.ndim(X) >= 1
    n = np.shape(XS[0])[0]
    for X in XS:
        assert n == np.shape(X)[0]

    def cross(XS):
        k = len(XS)
        XS = [np.reshape(XS[i], (n,) + (1,) * (k - i - 1) + (-1,) + (1,) * i)
              for i in range(k)]
        return np.reshape(reduce(np.multiply, XS), (n, -1))
    return cross(XS)
