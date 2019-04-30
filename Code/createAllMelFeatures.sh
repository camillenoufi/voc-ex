#!/bin/sh

python compute_melfeatures.py --data_dir /home/group/dataset/train/VAD_top10_balanced --out_file train_melfeats_balanced.pkl

python compute_melfeatures.py --data_dir /home/group/dataset/dev/VAD_top10_all --out_file dev_melfeats_all.pkl

python compute_melfeatures.py --data_dir /home/group/dataset/test/VAD_top10_all --out_file test_melfeats_all.pkl
