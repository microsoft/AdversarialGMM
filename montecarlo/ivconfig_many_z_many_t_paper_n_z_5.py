# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions
from mcpy import metrics
from mcpy import plotting
from mliv.dgps import fn_dict
from raw_plots import raw_metric, plot_raw
import papertables


CONFIG = {
    "target_dir": "many_z_many_t",
    "reload_results": True,
    "dgps": {
        "dgp1": ivfunctions.gen_data
    },
    "dgp_opts": {
        'dgp_num': 5,
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000,
        'n_instruments': 5,
        'iv_strength': .6,
        'n_test': 1000,
        'gridtest': 0
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
        'print_metrics': lambda x, y, z: papertables.paper_table(x, y, z,
                                                     filename='many_z_many_t_print_metrics_paper_n_2000_gamma_6_n_z_5.csv',
                                                     nn=False)
    },
    "sweep_plots": {
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
