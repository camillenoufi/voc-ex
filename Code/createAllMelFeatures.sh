#!/bin/sh

python compute_melfeatures.py --data_dir ./../../dataset/train --out_file train_melfeats_balanced.pkl

#python compute_melfeatures.py --data_dir /home/group/dataset/dev --out_file dev_melfeats_all.pkl

python compute_melfeatures.py --data_dir ./../../dataset/test --out_file test_melfeats_all.pkl
