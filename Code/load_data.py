"""
Data Load Class to load features and labels
"""

import os
import numpy as np, scipy as sp
import pickle
import csv, pandas as pd
import copy

class DataLoader:
    def __init__(self, data_dir,split_dir):
        """
        describe
        """
        super(DataLoader, self).__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir

    def load_filelist(self):
        """
        describe
        """
        assert os.path.isdir(self.data_dir), "Couldn't find the dataset at {}".format(data_dir)
        filenames = os.listdir(self.data_dir)
        filepath_list = [os.path.join(self.data_dir, self.split_dir,f) for f in filenames if f.endswith('.wav')]
        return filepath_list

    def load_embedding_dict(self,fname):
        """
        describe
        """
        with open(os.path.join(self.data_dir, self.split_dir, fname), 'rb') as handle:
            embedding_dict = pickle.load(handle)
        return embedding_dict

    def load_onehot_dict(self,label_list_fname='label_nums.csv'):
        #create one-hot vectors from class labels
        label_vec_dict = {}
        df = pd.read_csv(os.path.join(self.data_dir, label_list_fname))
        label_list = df.values[0]
        onehot = np.zeros(1,len(label_list))
        for i,l in enumerate(label_list):
            this_onehot = copy.deepcopy(onehot)
            this_onehot[i] = 1
            label_vec_dict[l] = this_onehot
        return label_vec_dict

    def load_label_dict(self,metadata_file='local_train_bal.csv'):
        feat_label_dict = {}
        df = pd.read_csv(os.path.join(self.data_dir, metadata_file))
        data = df.values
        for entry in data:
            path_key = os.path.join(self.data_dir, self.split_dir, entry[0])
            label = entry[-1]
            feat_label_dict[path_key] = label
        return feat_label_dict
