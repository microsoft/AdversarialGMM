# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from .oadam import OAdam
from .rbflayer import RBF

# TODO. This epsilon is used only because pytorch 1.5 has an instability in torch.cdist
# when the input distance is close to zero, due to instability of the square root in
# automatic differentiation. Should be removed once pytorch fixes the instability.
# It can be set to 0 if using pytorch 1.4.0
EPSILON = 1e-2


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def _kernel(x, y, basis_func, sigma):
    return basis_func(torch.cdist(x, y + EPSILON) * torch.abs(sigma))


class _BaseAGMM:

    def _pretrain(self, Z, T, Y,
                  learner_l2, adversary_l2, adversary_norm_reg,
                  learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, logger, model_dir, device, verbose, add_sample_inds=False):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name

        self.n_epochs = n_epochs

        if add_sample_inds:
            sample_inds = torch.tensor(np.arange(Y.shape[0]))
            self.train_ds = TensorDataset(Z, T, Y, sample_inds)
        else:
            self.train_ds = TensorDataset(Z, T, Y)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)
        self.adversary = self.adversary.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizerD = OAdam(add_weight_decay(self.learner, learner_l2),
                                lr=learner_lr, betas=(beta1, .01))
        self.optimizerG = OAdam(add_weight_decay(
            self.adversary, adversary_l2, skip_list=self.skip_list), lr=adversary_lr, betas=(beta1, .01))

        if logger is not None:
            self.writer = SummaryWriter()

        return Z, T, Y

    def predict(self, T, model='avg', burn_in=0, alpha=None):
        """
        Parameters
        ----------
        T : treatments
        model : one of ('avg', 'final'), whether to use an average of models or the final
        burn_in : discard the first "burn_in" epochs when doing averaging
        alpha : if not None but a float, then it also returns the a/2 and 1-a/2, percentile of
            the predictions across different epochs (proxy for a confidence interval)
        """
        if model == 'avg':
            preds = np.array([torch.load(os.path.join(self.model_dir,
                                                      "epoch{}".format(i)))(T).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            if alpha is None:
                return np.mean(preds, axis=0)
            else:
                return np.mean(preds, axis=0),\
                    np.percentile(
                        preds, 100 * alpha / 2, axis=0), np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))(T).cpu().data.numpy()
        if isinstance(model, int):
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(model)))(T).cpu().data.numpy()


class _BaseSupLossAGMM(_BaseAGMM):

    def fit(self, Z, T, Y,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            ols_weight=0., warm_start=False, logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        Z : instruments
        T : treatments
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adveresary norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        ols_weight : weight on OLS (square loss) objective
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        """

        Z, T, Y = self._pretrain(Z, T, Y,
                                 learner_l2, adversary_l2, adversary_norm_reg,
                                 learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                                 warm_start, logger, model_dir, device, verbose)

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (zb, xb, yb) in enumerate(self.train_dl):

                zb, xb, yb = map(lambda x: x.to(device), (zb, xb, yb))

                if (it % train_learner_every == 0):
                    self.learner.train()
                    pred = self.learner(xb)
                    test = self.adversary(zb)
                    D_loss = torch.mean(
                        (yb - pred) * test) + ols_weight * torch.mean((yb - pred)**2)
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if (it % train_adversary_every == 0):
                    self.adversary.train()
                    pred = self.learner(xb)
                    reg = 0
                    if self.adversary_reg:
                        test, reg = self.adversary(zb, reg=True)
                    else:
                        test = self.adversary(zb)
                    G_loss = - torch.mean((yb - pred) *
                                          test) + torch.mean(test**2)
                    G_loss += adversary_norm_reg * reg
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()

            torch.save(self.learner, os.path.join(
                self.model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self


class AGMM(_BaseSupLossAGMM):

    def __init__(self, learner, adversary):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        """
        self.learner = learner
        self.adversary = adversary
        # whether we have a norm penalty for the adversary
        self.adversary_reg = False
        # which adversary parameters to not ell2 penalize
        self.skip_list = []


class KernelLayerMMDGMM(_BaseSupLossAGMM):

    def __init__(self, learner, adversary_g, g_features,
                 n_centers, kernel, centers=None, sigmas=None, trainable=True):
        """
        Parameters
        ----------
        learner : a pytorch neural net module for the learner
        adversary_g : a pytorch neural net module for the g function of the adversary
        g_features : what is the output number of features of g
        n_centers : how many centers to use in the kernel layer
        kernel : the kernel function
        centers : numpy array that contains the inital value of the centers in the g(Z) space
        sigmas : numpy arra that contains the initial value of the sigma for each center
            (e.g. the precition of the kernel)
        trainable : whether to train the centers and the sigmas
        """
        class Adversary(torch.nn.Module):

            def __init__(self, g, g_features, n_centers, basis_func,
                         centers=None, sigmas=None, trainable=True):
                super(Adversary, self).__init__()
                self.g = g
                self.rbf = RBF(g_features, n_centers, basis_func,
                               centres=centers, sigmas=sigmas, trainable=trainable)
                self.beta = nn.Linear(n_centers, 1)

            def forward(self, x, reg=False):
                test = self.beta(self.rbf(self.g(x)))
                if not reg:
                    return test

                beta = self.beta.weight
                K = self.rbf(self.rbf.centres + EPSILON)
                K = (K + K.T) / 2
                rkhs_norm = (beta @ K @ beta.T)[0][0]
                return test, rkhs_norm

        self.learner = learner
        self.adversary = Adversary(adversary_g, g_features, n_centers,
                                   kernel, centers=centers, sigmas=sigmas, trainable=trainable)
        # whether we have a norm penalty for the adversary
        self.adversary_reg = True
        # which adversary parameters to not ell2 penalize
        self.skip_list = ['rbf.centres', 'beta.weight']


class CentroidMMDGMM(_BaseSupLossAGMM):

    def __init__(self, learner, adversary_g,
                 kernel, centers, sigma):
        """
        Parameters
        ----------
        learner : a pytorch neural net module for the learner
        adversary_g : a pytorch neural net module for the g function of the adversary
        kernel : the kernel function
        centers : numpy array that contains the inital value of the centers in the Z space
        sigma : float that corresponds to the precition of the kernel
        """
        class Adversary(torch.nn.Module):

            def __init__(self, g, basis_func, centers, sigma):
                super(Adversary, self).__init__()
                self.g = g
                self.centers = nn.Parameter(
                    torch.Tensor(centers), requires_grad=False)
                self.basis_func = basis_func
                if hasattr(sigma, '__len__'):
                    self.init_sigma = sigma.reshape(1, -1)
                    self.sigma = nn.Parameter(torch.Tensor(self.init_sigma))
                else:
                    self.init_sigma = sigma
                    self.sigma = nn.Parameter(torch.tensor(self.init_sigma))
                self.beta = nn.Linear(centers.shape[0], 1)
                self.reset_parameters()

            def reset_parameters(self):
                if hasattr(self.init_sigma, '__len__'):
                    self.sigma.data = torch.Tensor(
                        self.init_sigma).to(self.sigma.device)
                else:
                    self.sigma.data = torch.tensor(
                        self.init_sigma).to(self.sigma.device)

            def forward(self, x, reg=False):
                x1, x2 = self.g(x), self.g(self.centers)
                K12 = _kernel(x1, x2, self.basis_func, self.sigma)
                test = self.beta(K12)
                if reg:
                    K22 = _kernel(x2, x2, self.basis_func, self.sigma)
                    rkhs_reg = (self.beta.weight @ (K22 + K22.T) @
                                self.beta.weight.T)[0][0] / 2
                    return test, rkhs_reg
                return test

        self.learner = learner
        self.adversary = Adversary(
            adversary_g, kernel, centers, sigma=sigma)
        # whether we have a norm penalty for the adversary
        self.adversary_reg = True
        # which adversary parameters to not ell2 penalize
        self.skip_list = ['beta.weight']


class KernelLossAGMM(_BaseAGMM):

    def __init__(self, learner, adversary_g, kernel, sigma):
        """
        Parameters
        ----------
        learner : a pytorch neural net module for the learner
        adversary_g : a pytorch neural net module for the g function of the adversary
        kernel : the kernel function
        sigma : float that corresponds to the precition of the kernel
        """
        class Adversary(torch.nn.Module):

            def __init__(self, g, basis_func, sigma):
                super(Adversary, self).__init__()
                self.g = g
                self.basis_func = basis_func
                if hasattr(sigma, '__len__'):
                    self.init_sigma = sigma.reshape(1, -1)
                    self.sigma = nn.Parameter(torch.Tensor(self.init_sigma))
                else:
                    self.init_sigma = sigma
                    self.sigma = nn.Parameter(torch.tensor(self.init_sigma))
                self.reset_parameters()

            def reset_parameters(self):
                if hasattr(self.init_sigma, '__len__'):
                    self.sigma.data = torch.Tensor(
                        self.init_sigma).to(self.sigma.device)
                else:
                    self.sigma.data = torch.tensor(
                        self.init_sigma).to(self.sigma.device)

            def forward(self, x1, x2):
                return _kernel(self.g(x1), self.g(x2), self.basis_func, self.sigma)

        self.learner = learner
        self.adversary = Adversary(adversary_g, kernel, sigma)
        self.skip_list = []

    def fit(self, Z, T, Y,
            learner_l2=1e-3, adversary_l2=1e-4,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            ols_weight=0.0, warm_start=False, logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        Z : instruments
        T : treatments
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        ols_weight : weight on OLS (square loss) objective
        warm_start : whehter to reset weights or not
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        """

        Z, T, Y = self._pretrain(Z, T, Y,
                                 learner_l2, adversary_l2, 0,
                                 learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                                 warm_start, logger, model_dir, device, verbose)

        train_dl2 = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        for epoch in range(n_epochs):
            print("Epoch #", epoch, sep="")
            for it, ((zb1, xb1, yb1), (zb2, xb2, yb2)) in enumerate(zip(self.train_dl, train_dl2)):

                zb1, xb1, yb1 = map(lambda x: x.to(device), (zb1, xb1, yb1))
                zb2, xb2, yb2 = map(lambda x: x.to(device), (zb2, xb2, yb2))

                if it % train_learner_every == 0:
                    self.learner.train()
                    psi1, psi2 = yb1 - \
                        self.learner(xb1), yb2 - self.learner(xb2)
                    kernel = self.adversary(zb1, zb2)
                    D_loss = psi1.T @ kernel @ psi2 / (bs**2)
                    D_loss += ols_weight * \
                        (torch.mean(psi1**2) + torch.mean(psi2**2)) / 2
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if it % train_adversary_every == 0:
                    self.adversary.train()
                    psi1, psi2 = yb1 - \
                        self.learner(xb1), yb2 - self.learner(xb2)
                    kernel = self.adversary(zb1, zb2)
                    G_loss = - psi1.T @ kernel @ psi2 / (bs**2)
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()

            torch.save(self.learner, os.path.join(
                self.model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self


class MMDGMM(_BaseAGMM):

    def __init__(self, learner, adversary_g, n_samples, kernel, sigma):
        """
        Parameters
        ----------
        learner : a pytorch neural net module for the learner
        adversary_g : a pytorch neural net module for the g function of the adversary
        kernel : the kernel function
        sigma : float that corresponds to the precition of the kernel
        """
        class Adversary(torch.nn.Module):

            def __init__(self, g, n_samples, basis_func, sigma):
                super(Adversary, self).__init__()
                self.g = g
                self.basis_func = basis_func
                if hasattr(sigma, '__len__'):
                    self.init_sigma = sigma.reshape(1, -1)
                    self.sigma = nn.Parameter(torch.Tensor(self.init_sigma))
                else:
                    self.init_sigma = sigma
                    self.sigma = nn.Parameter(torch.tensor(self.init_sigma))
                self.beta = nn.Parameter(torch.Tensor(n_samples, 1))
                self.reset_parameters()

            def reset_parameters(self):
                if hasattr(self.init_sigma, '__len__'):
                    self.sigma.data = torch.Tensor(
                        self.init_sigma).to(self.sigma.device)
                else:
                    self.sigma.data = torch.tensor(
                        self.init_sigma).to(self.sigma.device)
                stdv = 1. / np.sqrt(self.beta.size(0))
                nn.init.uniform_(self.beta, -stdv, stdv)

            def forward(self, x1, x2, x3, id1, id2, id3, reg=False):
                x1, x2 = self.g(x1), self.g(x2)
                K12 = _kernel(x1, x2, self.basis_func, self.sigma[:, id2]) / 2
                K12 += _kernel(x2, x1, self.basis_func,
                               self.sigma[:, id1]).T / 2
                ratio2 = self.beta.size(0) / id2.shape[0]
                test = K12 @ self.beta[id2] * ratio2
                if reg:
                    x3 = self.g(x3)
                    K31 = _kernel(x3, x1, self.basis_func,
                                  self.sigma[:, id1]) / 2
                    K31 += _kernel(x1, x3, self.basis_func,
                                   self.sigma[:, id3]).T / 2
                    K32 = _kernel(x3, x2, self.basis_func,
                                  self.sigma[:, id2]) / 2
                    K32 += _kernel(x2, x3, self.basis_func,
                                   self.sigma[:, id3]).T / 2
                    ratio3 = self.beta.size(0) / id3.shape[0]
                    rkhs_reg = (self.beta[id3].T @ K32 @ self.beta[id2] *
                                ratio3 * ratio2)[0][0]
                    u = self.beta[id3].T @ K31 * ratio3
                    l2_reg = (u @ test)[0][0] / x1.size(0)
                    return test, rkhs_reg, l2_reg
                return test

        self.learner = learner
        self.adversary = Adversary(adversary_g, n_samples, kernel, sigma)
        self.skip_list = ['beta']

    def fit(self, Z, T, Y,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs1=100, bs2=100, bs3=100, train_learner_every=1, train_adversary_every=1,
            ols_weight=0.0, warm_start=False, logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        Z : instruments
        T : treatments
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        ols_weight : weight on OLS (square loss) objective
        warm_start : whehter to reset weights or not
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        """

        Z, T, Y = self._pretrain(Z, T, Y,
                                 learner_l2, adversary_l2, adversary_norm_reg,
                                 learner_lr, adversary_lr, n_epochs, bs1, train_learner_every, train_adversary_every,
                                 warm_start, logger, model_dir, device, verbose, add_sample_inds=True)

        sample_inds = np.arange(Y.shape[0]).astype(int)

        for epoch in range(n_epochs):
            print("Epoch #", epoch, sep="")
            for it, (zb1, xb1, yb1, idb1) in enumerate(self.train_dl):

                zb1, xb1, yb1, idb1 = map(
                    lambda x: x.to(device), (zb1, xb1, yb1, idb1))

                idb2 = np.random.choice(sample_inds, bs2, replace=False)
                zb2 = Z[idb2].to(device)
                idb3 = np.random.choice(sample_inds, bs3, replace=False)
                zb3 = Z[idb3].to(device)

                if it % train_learner_every == 0:
                    self.learner.train()
                    psi = yb1 - self.learner(xb1)
                    test = self.adversary(zb1, zb2, zb3, idb1, idb2, idb3)
                    D_loss = torch.mean(psi * test)
                    D_loss += ols_weight * torch.mean(psi**2)
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if it % train_adversary_every == 0:
                    self.adversary.train()
                    psi = yb1 - self.learner(xb1)
                    test, rkhs_reg, l2_reg = self.adversary(
                        zb1, zb2, zb3, idb1, idb2, idb3, reg=True)
                    G_loss = - torch.mean(psi * test)
                    G_loss += adversary_norm_reg * rkhs_reg
                    G_loss += l2_reg
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()

            torch.save(self.learner, os.path.join(
                model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self
