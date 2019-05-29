#!/bin/sh

python compute_hpss.py --data_dir ./../../dataset/train --out_file train_hpssfeats_balanced.pkl

python compute_hpss.py --data_dir ./../../dataset/test --out_file test_hpssfeats_all.pkl
