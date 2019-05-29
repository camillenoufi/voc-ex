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
        self.split_dir = data_dir + '/' + split_dir

    def load_filelist(self):
        """
        describe
        """
        assert os.path.isdir(self.split_dir), "Couldn't find the dataset at {}".format(self.split_dir)
        print("partition dir is {} \n".format(self.split_dir))
        filenames = os.listdir(self.split_dir)
        filepath_list = [os.path.join(self.split_dir,f) for f in filenames if f.endswith('.wav')]
        return filepath_list

    def load_embedding_dict(self,fname):
        """
        describe
        """
        with open(os.path.join(self.split_dir, fname), 'rb') as handle:
            embedding_dict = pickle.load(handle, encoding='latin1') #for python2 and python3 compatibility
        return embedding_dict

    def convert_embed_dict_to_local(self, embedding_dict):
        """
        Function that converts embedding dict returned by  load_embedding_dict to be dictionary from
        local path names to values rather than absolute
        """
        embedding_dict_local = {}
        for k,v in embedding_dict.items():
            file = os.path.basename(k)
            local_path = os.path.join(self.split_dir,file)
            embedding_dict_local[local_path] = v

        return embedding_dict_local

    def load_onehot_dict(self,label_list_fname='label_nums.csv'):
        #create one-hot vectors from class labels
        label_vec_dict = {}
        df = pd.read_csv(os.path.join(self.data_dir, label_list_fname))
        label_list = df.values[0]
        onehot = np.zeros((len(label_list),1))
        for i,l in enumerate(label_list):
            this_onehot = copy.deepcopy(onehot)
            this_onehot[i] = 1
            label_vec_dict[l] = this_onehot
        return label_vec_dict

    def load_label_dict(self, metadata_file):
        label_dict = {}
        df = pd.read_csv(os.path.join(self.split_dir, metadata_file))
        data = df.values
        for entry in data:
            path_key = os.path.join(self.split_dir, entry[0])
            label = entry[-1] #this is because the country-locale code in the csv file is the last (farthest right) column in the table
            label_dict[path_key] = label
        return label_dict
