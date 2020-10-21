
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import numpy as np
from datetime import datetime
import joblib
import argparse
import itertools
import inspect
from .deepiv.models import Treatment, Response
from .deepiv import architectures
from .deepiv import densities
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate

# The deep_iv benchmark


def deep_iv_fit(z, t, y, x=None, epochs=100, hidden=[128, 64, 32]):
    # z - instruments
    # t - treatment
    # y - response
    # x - features

    n = z.shape[0]
    dropout_rate = min(1000. / (1000. + n), 0.5)
    batch_size = 100
    images = False
    act = "relu"

    # Build and fit treatment model
    n_components = 10
    instruments = Input(shape=(z.shape[1],), name="instruments")
    treatment = Input(shape=(t.shape[1],), name="treatment")
    if x is None:
        treatment_input = instruments
        est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, n_components),
                                                   hidden_layers=hidden,
                                                   dropout_rate=dropout_rate, l2=0.0001,
                                                   activations=act)

        treatment_model = Treatment(inputs=[instruments], outputs=est_treat)
        treatment_model.compile('adam',
                                loss="mixture_of_gaussians",
                                n_components=n_components)

        treatment_model.fit([z], t, epochs=epochs, batch_size=batch_size)

    else:
        features = Input(shape=(x.shape[1],), name="features")
        treatment_input = Concatenate(axis=1)([instruments, features])
        est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, n_components),
                                                   hidden_layers=hidden,
                                                   dropout_rate=dropout_rate, l2=0.0001,
                                                   activations=act)

        treatment_model = Treatment(
            inputs=[instruments, features], outputs=est_treat)
        treatment_model.compile('adam',
                                loss="mixture_of_gaussians",
                                n_components=n_components)
        treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

    # Build and fit response model
    if x is None:
        response_input = treatment
        est_response = architectures.feed_forward_net(response_input, Dense(1),
                                                      activations=act,
                                                      hidden_layers=hidden,
                                                      l2=0.001,
                                                      dropout_rate=dropout_rate)
        response_model = Response(treatment=treatment_model,
                                  inputs=[treatment],
                                  outputs=est_response)
        response_model.compile('adam', loss='mse')
        response_model.fit([z], y, epochs=epochs, verbose=1,
                           batch_size=batch_size, samples_per_batch=2)
    else:
        features = Input(shape=(x.shape[1],), name="features")
        response_input = Concatenate(axis=1)([features, treatment])
        est_response = architectures.feed_forward_net(response_input, Dense(1),
                                                      activations=act,
                                                      hidden_layers=hidden,
                                                      l2=0.001,
                                                      dropout_rate=dropout_rate)
        response_model = Response(treatment_model,
                                  inputs=[features, treatment],
                                  outputs=est_response)
        response_model.compile('adam', loss='mse')
        response_model.fit([z, x], y, epochs=epochs, verbose=1,
                           batch_size=batch_size, samples_per_batch=2)

    return response_model
