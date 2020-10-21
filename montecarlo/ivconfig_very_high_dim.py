# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions
from mcpy import metrics
from mcpy import plotting
from mliv.dgps import fn_dict
from raw_plots import raw_metric, plot_raw


CONFIG = {
    "target_dir": "veryhighdimlin",
    "reload_results": True,
    "dgps": {
        "dgp1": ivfunctions.gen_data
    },
    "dgp_opts": {
        'dgp_num': 5,
        'fn': fn_dict['linear'],
        'n_samples': 400,
        'n_instruments': [100000, 1000000],
        'iv_strength': [.6, .8],
        'n_test': 1000,
        'gridtest': [1, 0]
    },
    "methods": {
        "StSpLinL1": ivfunctions.stochasticl1sparselinear
    },
    "method_opts": {
        'lin_l1': .5,
        'lin_nit': 2000
    },
    "metrics": {
        'rmse': ivfunctions.mse,
        'rsquare': ivfunctions.rsquare,
        'raw': raw_metric
    },
    "plots": {
        'metrics': {'metrics': ['rmse', 'rsquare']},
        'est': plot_raw,
        'print_metrics': ivfunctions.print_metrics
    },
    "sweep_plots": {
    },
    "mc_opts": {
        'n_experiments': 100,  # number of monte carlo experiments
        "seed": 123,
        "n_jobs": 1
    },
    "cluster_opts": {
        "node_id": __NODEID__,
        "n_nodes": __NNODES__
    },
    "proposed_method": "StSpLinL1",
}
