# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from .utilities import standardize
from ..dgps import get_tau_fn, fn_dict

# DGP from Bennet et al. paper


class AbstractMNISTxz(object):
    def __init__(self, use_x_images, use_z_images, tau_fn):
        self.splits = {"test": None, "train": None, "dev": None}
        self.setup_args = None
        self.initialized = False
        self._function_name = tau_fn

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST("datasets", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=60000)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("datasets", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=10000)
        train_data, test_data = list(train_loader), list(test_loader)
        images_list = [train_data[0][0].numpy(), test_data[0][0].numpy()]
        labels_list = [train_data[0][1].numpy(), test_data[0][1].numpy()]
        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        idx = list(range(self.images.shape[0]))
        random.shuffle(idx)
        self.images = self.images[idx]
        self.labels = self.labels[idx]
        self.data_i = 0
        self.tau_fn = tau_fn

        self.use_x_images = use_x_images
        self.use_z_images = use_z_images

    def _sample_images(self, sample_digits, images, labels):
        # image_array = np.zeros(shape=(len(sample_digits), 1, 28, 28))
        # for d in range(10):
        #     d_idx = np.array([int(d_) for d_ in labels if d_ == d])
        #     fill_idx = np.array([int(d_) for d_ in sample_digits if d_ == d])
        #     num_digits = len(d_idx)
        #     num_fill = len(fill_idx)
        #     d_images = images[d_idx]
        #     idx_sample = np.random.choice(list(range(num_digits)), num_fill)
        #     image_sample = d_images[idx_sample]
        #     image_array[fill_idx] = image_sample
        # return image_array
        digit_dict = defaultdict(list)
        for l, image in zip(labels, images):
            digit_dict[int(l)].append(image)
        images = np.stack([random.choice(digit_dict[int(d)])
                           for d in sample_digits], axis=0)
        return images

    def _true_tau_function_np(self, x):
        func = self._function_name
        return get_tau_fn(fn_dict[func])(x)

    def generate_data(self, num_data, tau_fn='linear', two_gps=False, n_instruments=1, iv_strength=0.5, **kwargs):
        idx = list(range(self.data_i, self.data_i + num_data))
        images = self.images[idx]
        labels = self.labels[idx]
        self.data_i += num_data

        confounder = np.random.normal(0, 1, size=(num_data, 1))
        # Z, (size: nx2) (range: -3,3)
        z = np.random.uniform(-3, 3, size=(num_data, n_instruments))
        if two_gps:
            x = 2 * z[:, 0].reshape(-1, 1) * (z[:, 0] > 0).reshape(-1, 1) * iv_strength \
                + 2 * z[:, 1].reshape(-1, 1) * (z[:, 1] < 0).reshape(-1, 1) * iv_strength \
                + 2 * confounder * (1 - iv_strength) + \
                np.random.normal(0, .1, size=(num_data, 1))
        else:
            # x = 2z_1*iv_strength + 2confounder(1-iv_strength)+noise (size: nx1)
            x = 2 * z[:, 0].reshape(-1, 1) * iv_strength \
                + 2 * confounder * (1 - iv_strength) + \
                np.random.normal(0, .1, size=(num_data, 1))
        # g = tau(x) (size: nx1)
        g = self._true_tau_function_np(x)
        # y = g+2*confounder + noise (size: nx1)
        y = g + 2 * confounder + \
            np.random.normal(0, .1, size=(num_data, 1))

        toy_x, toy_z, toy_y, toy_g = x, z, y, g
        if self.use_x_images:
            # x_digits = round(max(0, min(1.5x+5,9))) (each x_digit is between 0 to 9)
            x_digits = np.clip(1.5 * toy_x[:, 0] + 5.0, 0, 9).round()
            x = self._sample_images(
                x_digits, images, labels).reshape(-1, 1, 28, 28)
            g = self._true_tau_function_np(
                (x_digits - 5.0) / 1.5).reshape(-1, 1)
            w = x_digits.reshape(-1, 1)
        else:
            x = toy_x.reshape(-1, 1) * 1.5 + 5.0
            g = toy_g.reshape(-1, 1)
            w = toy_x.reshape(-1, 1) * 1.5 + 5.0

        if self.use_z_images:
            z_digits = np.clip(1.5 * toy_z[:, 0] + 5.0, 0, 9).round()
            z = self._sample_images(
                z_digits, images, labels).reshape(-1, 1, 28, 28)
        else:
            z = toy_z.reshape(-1, 1)

        # print(np.stack([w[:20, 0], g[:20, 0]]))
        x, toy_y, z, g, w = standardize(x, toy_y, z, g, w)
        return x, z, toy_y, g, w
