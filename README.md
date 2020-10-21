# Introduction

Code for replication of experiments in the paper:

Dikkala, Nishanth, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. "Minimax estimation of conditional moment models." arXiv preprint arXiv:2006.07201 (2020).

https://arxiv.org/abs/2006.07201

# Installation

Before running any script you need to install the mliv package by running:

```
python setup.py develop
```

The main dependencies of the package are, cvxopt, scikit-learn, numpy, pytorch.

cvxopt is installed by running the setup.py command above. The rest need to be installed independently.

# MLIV Package

This will install the package `mliv` on your python environment, which contains all the non-parametric instrumental variable estimators proposed in the paper. It contains the components:
* `mliv.rkhs`: estimators based on Reproducing Kernel Hilbert Spaces. See [RKHS Notebook](local_notebooks/RkhsExamples.ipynb) for example usage.
* `mliv.linear`: estimators based on high-dimensional sparse linear represetations. See [Sparse Linear Notebook](local_notebooks/SparseLinearExamples.ipynb) for example usage.
* `mliv.ensemble`: estimators based on random forests. See [Random Forest Notebook](local_notebooks/EnsemblesExamples.ipynb) for example usage.
* `mliv.neuralnet`: estimators based on neural nets, using pytorch. See [Neural Net Notebook](local_notebooks/NeuralNetExamples.ipynb) and [MNIST Experiments Notebook](local_notebooks/MNIST_Experiments.ipynb) for example usage.
* `mliv.shape`: estimators based on shape constraints. See [Shape Constraints Notebook](local_notebooks/ShapeExamples.ipynb) for example usage.

# Table Generation for Non-Image Experiments

To generate figures that are not related to the mnist experiments you need to run:
```
cd montecarlo
chmod +x local_script.sh
local_script.sh {config_file}
```
where config file is different for each figure. The above is for windows. For linux replace `local_script.sh` with
`local_script_linux.sh` and for Mac replace it with `local_script_osx.sh`. Results are saved in a .csv file with the
corresponding name. 

* To generate Figure 14:
```
local_script.sh ivconfig_one_z_one_t_paper_n_300_gamma_6
```
* To generate Figure 15:
```
local_script.sh ivconfig_one_z_one_t_paper_n_2000_gamma_6
```
* To generate Figure 16:
```
local_script.sh ivconfig_one_z_one_t_paper_n_2000_gamma_8
```
* To generate Figure 17:
```
local_script.sh ivconfig_many_z_one_t_paper_n_z_5
```
* To generate Figure 18:
```
local_script.sh ivconfig_many_z_one_t_paper_n_z_10
```
* To generate Figure 19:
```
local_script.sh ivconfig_many_z_many_t_paper_n_z_5
```
* To generate Figure 20:
```
local_script.sh ivconfig_many_z_many_t_paper_n_z_10
```
* To generate Figure 21:
```
local_script.sh ivconfig_many_z_many_t_nnet_paper
```
* To generate Figure 22:
```
local_script.sh ivconfig_high_dim_paper
local_script.sh ivconfig_very_high_dim_paper
```
In this case the results are stored in the "print_metrics.csv" file of the newly created folders: `highdimlin` and
`veryhighdimlin`.

These experiments were run on a CPU cluster within a linux environment and parallelism over 1000 cores.

# Table Generation for MNIST experiments

To generate the mnist results, from the main folder run:
```
chmod +x run_mnist_experiments.sh
./run_mnist_experiments.sh"
```

These experiments were run on a single linux GPU node.

# Further Examples

Further examples on how to use the methods and run single instances can be found in the folder `local_notebooks`. Each
notebook contains examples from each part of the library (RandomForests, NeuralNets, RKHS, SparseLinear, ShapeConstraints).

# Third-Party Material

The folder `deepgmm` contains a frozen copy of the git repo https://github.com/CausalML/DeepGMM that contains the code
from the prior work of Bennet et al. Deep Generalized Method of Moments for Instrumental Variable Analysis.

The code in `mliv/neuralnet/oadam.py` and `mliv/neuralnet/rbflayer.py` is forked and modified from the repository:
https://github.com/georgepar/optimistic-adam
By George Paraskevopoulos on April 15, 2020

The code in `montecarlo/mcpy` was forked and modified from the repository:
https://github.com/vsyrgkanis/plugin_regularized_estimation


# Contributing and Feedback

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademark Notice

Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.


# Security

Microsoft takes the security of our software products and services seriously, which includes all source code repositories managed through our GitHub organizations, which include [Microsoft](https://github.com/Microsoft), [Azure](https://github.com/Azure), [DotNet](https://github.com/dotnet), [AspNet](https://github.com/aspnet), [Xamarin](https://github.com/xamarin), and [our GitHub organizations](https://opensource.microsoft.com/).

If you believe you have found a security vulnerability in any Microsoft-owned repository that meets Microsoft's [Microsoft's definition of a security vulnerability](https://docs.microsoft.com/en-us/previous-versions/tn-archive/cc751383(v=technet.10)), please report it to us as described below.

## Reporting Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them to the Microsoft Security Response Center (MSRC) at [https://msrc.microsoft.com/create-report](https://msrc.microsoft.com/create-report).

If you prefer to submit without logging in, send email to [secure@microsoft.com](mailto:secure@microsoft.com).  If possible, encrypt your message with our PGP key; please download it from the the [Microsoft Security Response Center PGP Key page](https://www.microsoft.com/en-us/msrc/pgp-key-msrc).

You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Additional information can be found at [microsoft.com/msrc](https://www.microsoft.com/msrc).

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

  * Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
  * Full paths of source file(s) related to the manifestation of the issue
  * The location of the affected source code (tag/branch/commit or direct URL)
  * Any special configuration required to reproduce the issue
  * Step-by-step instructions to reproduce the issue
  * Proof-of-concept or exploit code (if possible)
  * Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

If you are reporting for a bug bounty, more complete reports can contribute to a higher bounty award. Please visit our [Microsoft Bug Bounty Program](https://microsoft.com/msrc/bounty) page for more details about our active programs.

## Preferred Languages

We prefer all communications to be in English.

## Policy

Microsoft follows the principle of [Coordinated Vulnerability Disclosure](https://www.microsoft.com/en-us/msrc/cvd).
