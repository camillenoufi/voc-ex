#!/bin/sh

python compute_stft.py --data_dir ./../../dataset/train --out_file train_stftfeats_balanced.pkl

python compute_stft.py --data_dir ./../../dataset/test --out_file test_stftfeats_all.pkl
