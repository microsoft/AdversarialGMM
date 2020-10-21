# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions
import nnetivfunctions
from mcpy import metrics
from mcpy import plotting
from mliv.dgps import fn_dict
from raw_plots import raw_metric, plot_raw
import papertables

CONFIG = {
    "target_dir": "many_z_many_t_nnet",
    "reload_results": True,
    "dgps": {
        "dgp1": ivfunctions.gen_data
    },
    "dgp_opts": {
        'dgp_num': 5,
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000,
        'n_instruments': 10,
        'iv_strength': .6,
        'n_test': 1000,
        'gridtest': 0
    },
    "methods": {
        "AGMM": nnetivfunctions.agmm,
        "KLayerFixed": nnetivfunctions.klayerfixed,
        "KLayerTrained": nnetivfunctions.klayertrained,
        "CentroidMMD": nnetivfunctions.centroidmmd,
        "KLossMMD": nnetivfunctions.klossgmm
    },
    "method_opts": {
        'n_epochs': 300,
        'model': 0,  # 0 is avg, 1 is final
        'burnin': 200
    },
    "metrics": {
        'rmse': ivfunctions.mse,
        'rsquare': ivfunctions.rsquare,
        'raw': raw_metric
    },
    "plots": {
        'print_metrics': lambda x, y, z: papertables.paper_table(x, y, z,
                                                     filename='many_z_many_t_nnet_print_metrics_paper_n_2000_gamma_6_n_z_10.csv',
                                                     nn=True)
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
