import os
import sys
import numpy as np
from joblib import Parallel, delayed
import joblib
import argparse
import importlib
from itertools import product
import collections
from copy import deepcopy
from mcpy.utils import filesafe
from mcpy import plotting


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def _check_valid_config(config):
    assert 'dgps' in config, "config dict must contain dgps"
    assert 'dgp_opts' in config, "config dict must contain dgp_opts"
    assert 'method_opts' in config, "config dict must contain method_opts"
    assert 'mc_opts' in config, "config dict must contain mc_opts"
    assert 'metrics' in config, "config dict must contain metrics"
    assert 'methods' in config, "config dict must contain methods"
    assert 'plots' in config, "config dict must contain plots"
    assert 'target_dir' in config, "config must contain target_dir"
    assert 'reload_results' in config, "config must contain reload_results"
    assert 'n_experiments' in config['mc_opts'], "config[mc_opts] must contain n_experiments"
    assert 'seed' in config['mc_opts'], "config[mc_opts] must contain seed"


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


class MonteCarlo:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['method_opts'].items()])
        return

    def experiment(self, exp_id):
        ''' Runs an experiment on a single randomly generated instance and sample and returns
        the parameter estimates for each method and the evaluated metrics for each method
        '''
        np.random.seed(exp_id)

        param_estimates = {}
        true_params = {}
        for dgp_name, dgp_fn in self.config['dgps'].items():
            data, true_param = dgp_fn(self.config['dgp_opts'])
            true_params[dgp_name] = true_param
            param_estimates[dgp_name] = {}
            for method_name, method in self.config['methods'].items():
                param_estimates[dgp_name][method_name] = method(
                    data, self.config['method_opts'])

        return param_estimates, true_params

    def run(self):
        ''' Runs multiple experiments in parallel on randomly generated instances and samples and returns
        the parameter estimates for each method and the evaluated metrics for each method across all
        experiments
        '''
        random_seed = self.config['mc_opts']['seed']

        if not os.path.exists(self.config['target_dir']):
            os.makedirs(self.config['target_dir'])

        results_file = os.path.join(
            self.config['target_dir'], 'results_{}.jbl'.format(self.config['param_str']))
        if self.config['reload_results'] and os.path.exists(results_file):
            results = joblib.load(results_file)
        else:
            results = Parallel(n_jobs=_get(self.config['mc_opts'], 'n_jobs', -1), verbose=1)(
                delayed(self.experiment)(random_seed + exp_id)
                for exp_id in range(self.config['mc_opts']['n_experiments']))
            joblib.dump(results, results_file)

        param_estimates = {}
        metric_results = {}
        for dgp_name in self.config['dgps'].keys():
            param_estimates[dgp_name] = {}
            metric_results[dgp_name] = {}
            for method_name in self.config['methods'].keys():
                param_estimates[dgp_name][method_name] = np.array(
                    [results[i][0][dgp_name][method_name] for i in range(self.config['mc_opts']['n_experiments'])])
                metric_results[dgp_name][method_name] = {}
                for metric_name, metric_fn in self.config['metrics'].items():
                    metric_results[dgp_name][method_name][metric_name] = np.array([metric_fn(results[i][0][dgp_name][method_name], results[i][1][dgp_name])
                                                                                   for i in range(self.config['mc_opts']['n_experiments'])])

        for plot_name, plot_fn in self.config['plots'].items():
            if isinstance(plot_fn, dict):
                plotting.instance_plot(
                    plot_name, param_estimates, metric_results, self.config, plot_fn)
            else:
                plot_fn(param_estimates, metric_results, self.config)

        return param_estimates, metric_results


class MonteCarloSweep:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(['{}_{}'.format(filesafe(
            k), self._stringify_param(v)) for k, v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(
            k), self._stringify_param(v)) for k, v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(
            k), self._stringify_param(v)) for k, v in self.config['method_opts'].items()])
        return

    def _stringify_param(self, param):
        if hasattr(param, "__len__"):
            return '{}_to_{}'.format(np.min(param), np.max(param))
        else:
            return param

    def run(self):
        dgp_sweep_params = []
        dgp_sweep_param_vals = []
        for dgp_key, dgp_val in self.config['dgp_opts'].items():
            if hasattr(dgp_val, "__len__"):
                dgp_sweep_params.append(dgp_key)
                dgp_sweep_param_vals.append(dgp_val)

        n_sweeps = len(list(product(*dgp_sweep_param_vals)))
        if 'cluster_opts' in self.config:
            n_nodes = _get(self.config['cluster_opts'], 'n_nodes', 1)
            node_id = _get(self.config['cluster_opts'], 'node_id', 0)
        else:
            n_nodes = 1
            node_id = 0
        start_sweep, end_sweep = 0, 0
        if node_id < n_nodes - 1:
            node_splits = np.array_split(np.arange(n_sweeps), n_nodes - 1)
            start_sweep, end_sweep = node_splits[node_id][0], node_splits[node_id][-1]

        sweep_keys = []
        sweep_params = []
        sweep_metrics = []
        inst_config = deepcopy(self.config)
        # This is the node that loads results and plots sweep plots
        if (n_nodes > 1) and (node_id == n_nodes - 1):
            inst_config['reload_results'] = True
            inst_config['plots'] = {}
        for it, vec in enumerate(product(*dgp_sweep_param_vals)):
            if (node_id == n_nodes - 1) or ((it >= start_sweep) and (it <= end_sweep)):
                setting = list(zip(dgp_sweep_params, vec))
                for k, v in setting:
                    inst_config['dgp_opts'][k] = v
                params, metrics = MonteCarlo(inst_config).run()
                sweep_keys.append(setting)
                sweep_params.append(params)
                sweep_metrics.append(metrics)

        if node_id == n_nodes - 1:
            for plot_key, plot_fn in self.config['sweep_plots'].items():
                if isinstance(plot_fn, dict):
                    plotting.sweep_plot(
                        plot_key, sweep_keys, sweep_params, sweep_metrics, self.config, plot_fn)
                else:
                    plot_fn(plot_key, sweep_keys, sweep_params,
                            sweep_metrics, self.config)

        return sweep_keys, sweep_params, sweep_metrics


def monte_carlo_main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args(sys.argv[1:])

    config = importlib.import_module(args.config)
    MonteCarlo(config.CONFIG).run()


if __name__ == "__main__":
    monte_carlo_main()
