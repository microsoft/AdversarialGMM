#!/bin/bash
python3 setup.py develop
python3 run_mnist_ours.py
cd deepgmm
python3 run_mnist_experiments_ours.py
