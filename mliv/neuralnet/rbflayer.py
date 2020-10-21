# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Forked from the repository:
https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
By Hamish Flynn on April 15, 2020
"""
import torch
import torch.nn as nn

# RBF Layer


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func, centres=None, sigmas=None,
                 trainable=True):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.centres.requires_grad = trainable
        self.sigmas = nn.Parameter(torch.Tensor(1, out_features))
        self.sigmas.requires_grad = trainable
        self.basis_func = basis_func
        self.pd = nn.PairwiseDistance()
        self.init_centres = centres
        self.init_sigmas = sigmas
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_centres is not None:
            self.centres.data = torch.Tensor(
                self.init_centres).to(self.centres.device)
        else:
            nn.init.normal_(self.centres, 0, 1)
        if self.init_sigmas is not None:
            self.sigmas.data = torch.Tensor(
                self.init_sigmas).to(self.sigmas.device).T
        else:
            nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        distances = torch.cdist(input, self.centres) * torch.abs(self.sigmas)
        return self.basis_func(distances)


# RBFs

def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(
        alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
        * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5 * alpha) * \
        torch.exp(-3**0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5 * alpha + (5 / 3)
           * alpha.pow(2)) * torch.exp(-5**0.5 * alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
