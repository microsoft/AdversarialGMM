# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
#from mliv.neuralnet.deepiv_fit import deep_iv_fit
from mliv.neuralnet.rbflayer import gaussian, inverse_multiquadric
from mliv.neuralnet import AGMM, KernelLayerMMDGMM, CentroidMMDGMM, KernelLossAGMM, MMDGMM

p = 0.1  # dropout prob of dropout layers throughout notebook
n_hidden = 100  # width of hidden layers throughout notebook

# For any method that use a projection of z into features g(z)
g_features = 100

# The kernel function
kernel_fn = gaussian
# kernel_fn = inverse_multiquadric

# Training params
learner_lr = 1e-4
adversary_lr = 1e-4
learner_l2 = 1e-3
adversary_l2 = 1e-4
adversary_norm_reg = 1e-3
n_epochs = 300
bs = 100
sigma = 2.0 / g_features
n_centers = 100
device = torch.cuda.current_device() if torch.cuda.is_available() else None


def _get_learner(n_t):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))


def _get_adversary(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))


def _get_adversary_g(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, g_features), nn.ReLU())


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def _get_model_opt(opts, key, default):
    model_enc = _get(opts, 'model', default)
    return ('avg' if model_enc == 0 else 'final')


def agmm(data, opts):
    print("GPU:", torch.cuda.is_available())
    T_test, Z, T, Y = map(lambda x: torch.Tensor(x), data)
    learner = _get_learner(T.shape[1])
    adversary_fn = _get_adversary(Z.shape[1])
    agmm = AGMM(learner, adversary_fn).fit(Z, T, Y, learner_lr=learner_lr, adversary_lr=adversary_lr,
                                           learner_l2=learner_l2, adversary_l2=adversary_l2,
                                           n_epochs=_get(
                                               opts, 'n_epochs', n_epochs),
                                           bs=_get(opts, 'bs', bs),
                                           model_dir=str(Path.home()),
                                           device=device)
    return agmm.predict(T_test.to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klayerfixed(data, opts):
    T_test, Z, T, Y = map(lambda x: torch.Tensor(x), data)
    n_z = Z.shape[1]
    centers = np.tile(
        np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, n_z))
    sigmas = np.ones((n_centers,)) * 2 / n_z

    learner = _get_learner(T.shape[1])

    mmdgmm_fixed = KernelLayerMMDGMM(learner, lambda x: x, n_z, n_centers, kernel_fn,
                                     centers=centers, sigmas=sigmas, trainable=False)
    mmdgmm_fixed.fit(Z, T, Y, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    return mmdgmm_fixed.predict(T_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klayertrained(data, opts):
    T_test, Z, T, Y = map(lambda x: torch.Tensor(x), data)
    centers = np.random.uniform(-4, 4, size=(n_centers, g_features))
    sigmas = np.ones((n_centers,)) * sigma
    learner = _get_learner(T.shape[1])
    adversary_g = _get_adversary_g(Z.shape[1])
    klayermmdgmm = KernelLayerMMDGMM(learner, adversary_g, g_features,
                                     n_centers, kernel_fn, centers=centers, sigmas=sigmas)
    klayermmdgmm.fit(Z, T, Y, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    return klayermmdgmm.predict(T_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def centroidmmd(data, opts):
    _, Z, _, _ = data
    centers = KMeans(n_clusters=n_centers).fit(Z).cluster_centers_
    T_test, Z, T, Y = map(lambda x: torch.Tensor(x), data)
    learner = _get_learner(T.shape[1])
    adversary_g = _get_adversary_g(Z.shape[1])
    centroid_mmd = CentroidMMDGMM(
        learner, adversary_g, kernel_fn, centers, np.ones(n_centers) * sigma)
    centroid_mmd.fit(Z, T, Y, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    return centroid_mmd.predict(T_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klossgmm(data, opts):
    T_test, Z, T, Y = map(lambda x: torch.Tensor(x), data)
    learner = _get_learner(T.shape[1])
    adversary_g = _get_adversary_g(Z.shape[1])
    kernelgmm = KernelLossAGMM(learner, adversary_g, kernel_fn, sigma)
    kernelgmm.fit(Z, T, Y, learner_l2=learner_l2**2, adversary_l2=adversary_l2,
                  learner_lr=learner_lr, adversary_lr=adversary_lr,
                  n_epochs=_get(opts, 'n_epochs', n_epochs),
                  bs=_get(opts, 'bs', bs),
                  model_dir=str(Path.home()),
                  device=device)
    return kernelgmm.predict(T_test.to(device),
                             model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))
