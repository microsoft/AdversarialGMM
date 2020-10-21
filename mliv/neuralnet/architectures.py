# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Z_agmm(nn.Module):
    def __init__(self):
        super(CNN_Z_agmm, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x  # F.log_softmax(x, dim=1)
        return output


class CNN_Z_kernel(nn.Module):
    def __init__(self, g_features=100):
        super(CNN_Z_kernel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, g_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x  # F.log_softmax(x, dim=1)
        return output


class CNN_X(nn.Module):
    def __init__(self):
        super(CNN_X, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x  # F.log_softmax(x, dim=1)
        return output


class CNN_X_bn(nn.Module):
    def __init__(self):
        super(CNN_X_bn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.fc3(x)
        output = x  # F.log_softmax(x, dim=1)
        return output


def fc_z_kernel(n_z, n_hidden, g_features, dropout_p):
    FC_Z_kernel = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(n_z, n_hidden),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(n_hidden, g_features),
        nn.ReLU(),
    )
    return FC_Z_kernel


def fc_z_agmm(n_z, n_hidden, dropout_p):
    FC_Z_agmm = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(n_z, n_hidden),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(n_hidden, 20),
        nn.Linear(20, 1),
    )
    return FC_Z_agmm


def fc_x(n_t, n_hidden, dropout_p):
    FC_X = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(n_t, n_hidden),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(n_hidden, 1),
    )
    return FC_X
