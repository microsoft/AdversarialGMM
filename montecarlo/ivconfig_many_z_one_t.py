# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions
from mcpy import metrics
from mcpy import plotting
from mliv.dgps import fn_dict
from raw_plots import raw_metric, plot_raw


CONFIG = {
    "target_dir": "many_z_one_t",
    "reload_results": False,
    "dgps": {
        "dgp1": ivfunctions.gen_data
    },
    "dgp_opts": {
        'dgp_num': 2,
        'fn': list(iter(fn_dict.values())),
        'n_samples': [2000, 5000],
        'n_instruments': [2, 5, 10],
        'iv_strength': [.6, .8],
        'n_test': 1000,
        'gridtest': [1, 0]
    },
    "methods": {
        "NystromRKHS": ivfunctions.nystromrkhsfit,
        "2SLS": ivfunctions.tsls,
        "Reg2SLS": ivfunctions.regtsls,
        "SpLinL1": ivfunctions.l1sparselinear,
        "StSpLinL1": ivfunctions.stochasticl1sparselinear,
        "SpLinL2": ivfunctions.l2sparselinear,
        "RFIV": ivfunctions.ensembleiv,
        "RFStarIV": ivfunctions.ensemblestariv
    },
    "method_opts": {
        'nstrm_n_comp': 100,
        'shiv_L': 2,
        'shiv_mon': None,
        'lin_degree': 3
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
        **{"vsa{}".format(it): {'varying_params': ['n_samples'], 'metrics': ['rsquare'],
                                'select_vals': {'iv_strength': [.6], 'fn': [it], 'gridtest': [0], 'n_instruments': [5]}}
           for it in list(iter(fn_dict.values()))},
        **{"vst{}".format(it): {'varying_params': ['iv_strength'], 'metrics': ['rsquare'],
                                'select_vals': {'n_samples': [5000], 'fn': [it], 'gridtest': [0], 'n_instruments': [5]}}
           for it in list(iter(fn_dict.values()))},
        **{"vi{}".format(it): {'varying_params': ['n_instruments'], 'metrics': ['rsquare'],
                               'select_vals': {'n_samples': [5000], 'fn': [it], 'gridtest': [0], 'iv_strength': [.6]}}
           for it in list(iter(fn_dict.values()))},
        **{"vb{}".format(it): {'varying_params': [('n_samples', 'iv_strength')], 'metrics': ['rsquare'],
                               'select_vals': {'fn': [it], 'gridtest': [0], 'n_instruments': [5]}}
           for it in list(iter(fn_dict.values()))},
        **{"vsagr{}".format(it): {'varying_params': ['n_samples'], 'metrics': ['rsquare'],
                                  'select_vals': {'iv_strength': [.6], 'fn': [it], 'gridtest': [1], 'n_instruments': [5]}}
           for it in list(iter(fn_dict.values()))},
        **{"vstgr{}".format(it): {'varying_params': ['iv_strength'], 'metrics': ['rsquare'],
                                  'select_vals': {'n_samples': [5000], 'fn': [it], 'gridtest': [1], 'n_instruments': [5]}}
           for it in list(iter(fn_dict.values()))},
        **{"vigr{}".format(it): {'varying_params': ['n_instruments'], 'metrics': ['rsquare'],
                                 'select_vals': {'n_samples': [5000], 'fn': [it], 'gridtest': [1], 'iv_strength': [.6]}}
           for it in list(iter(fn_dict.values()))},
        **{"vbgr{}".format(it): {'varying_params': [('n_samples', 'iv_strength')], 'metrics': ['rsquare'],
                                 'select_vals': {'fn': [it], 'gridtest': [1], 'n_instruments': [5]}}
           for it in list(iter(fn_dict.values()))},
    },
    "mc_opts": {
        'n_experiments': 100,  # number of monte carlo experiments
        "seed": 123,
    },
    "cluster_opts": {
        "node_id": __NODEID__,
        "n_nodes": __NNODES__
    },
    "proposed_method": "NystromRKHS",
}
