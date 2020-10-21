# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from sklearn.model_selection import train_test_split
from .mnist_dgps import AbstractMNISTxz


def generate_data(
    X_IMAGE=False,
    Z_IMAGE=False,
    tau_fn="abs",
    n_samples=10000,
    n_dev_samples=5000,
    n_instruments=2,
    iv_strength=0.5,
    device=None,
):
    mnist_dgp = AbstractMNISTxz(X_IMAGE, Z_IMAGE, tau_fn)
    n_test = n_samples // 10
    n_t = 1

    T, Z, Y, G, _ = mnist_dgp.generate_data(
        n_samples, tau_fn=tau_fn, n_instruments=n_instruments, iv_strength=iv_strength
    )

    T_test, Z_test, Y_test, G_test, _ = mnist_dgp.generate_data(
        n_test, tau_fn=tau_fn, n_instruments=n_instruments, iv_strength=iv_strength,
    )

    T_dev, Z_dev, Y_dev, G_dev, _ = mnist_dgp.generate_data(
        n_dev_samples, tau_fn=tau_fn, n_instruments=n_instruments, iv_strength=iv_strength
    )

    Z_train, Z_val, T_train, T_val, Y_train, Y_val, G_train, G_val = train_test_split(
        Z, T, Y, G, test_size=0.1, shuffle=True
    )
    Z_train, T_train, Y_train, G_train = map(
        lambda x: torch.Tensor(x), (Z_train, T_train, Y_train, G_train)
    )
    Z_val, T_val, Y_val, G_val = map(
        lambda x: torch.Tensor(x).to(device), (Z_val, T_val, Y_val, G_val)
    )
    Z_test, T_test, Y_test, G_test = map(
        lambda x: torch.Tensor(x).to(device), (Z_test, T_test, Y_test, G_test)
    )

    Z_dev, T_dev, Y_dev, G_dev = map(
        lambda x: torch.Tensor(x).to(device), (Z_dev, T_dev, Y_dev, G_dev)
    )

    data_array = []
    data_array.append((Z_train, T_train, Y_train, G_train))
    data_array.append((Z_val, T_val, Y_val, G_val))
    data_array.append((Z_test, T_test, Y_test, G_test))
    data_array.append((Z_dev, T_dev, Y_dev, G_dev))
    return data_array
