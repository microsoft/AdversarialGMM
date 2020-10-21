# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# module imports
from mliv.neuralnet.experiment import experiment
import torch
import numpy as np
import itertools
import warnings
warnings.simplefilter('ignore')

# import from our files


def main():
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    print("Using GPU", device)
    VERBOSE = False

    tau_fns = ["abs"]  # , "sin", "2dpoly", "rand_pw", "3dpoly"]
    iv_strengths = [0.5]
    estimators = ["AGMM", "KernelLayerMMDGMM"]
    dgps = ["z_image", "x_image", "xz_image"]
    num_datas = [20000]

    settings = list(itertools.product(
        tau_fns, iv_strengths, dgps, num_datas, estimators))
    result_dict = {}
    monte_carlo = 10  # number of monte carlo runs to perform

    for (tau_fn, iv_strength, dgp, num_data, est) in settings:
        print("------ Setting ------")
        print(tau_fn)
        print("iv_strength", iv_strength)
        print("dgp", dgp)
        print("estimator", est)
        results = []
        for run in range(monte_carlo):
            print("Run", run+1)
            result = experiment(dgp, iv_strength, tau_fn,
                                num_data, est, device, VERBOSE)
            results.append(list(result))

        np_results = np.array(results)
        result_dict[(tau_fn, iv_strength, dgp, num_data, est)
                    ] = np_results.mean(axis=0)
        print("----- Results -----")
        print("Average MSE", np_results.mean(axis=0)[5])
        print("Standard deviation MSE", np_results.std(axis=0)[5])


if __name__ == '__main__':
    main()
