import os
import numpy as np
from load_data import DataLoader

# Main TEST Routine
if __name__ == '__main__':

   #### CHANGE THIS PATH FOR YOUR LOCAL MACHINE'S COPY OF THE DATA FOLDER ###############
   data_dir = '/Users/camillenoufi/Documents/datasets/VocEx-local/'
   ####################

   partition = '/local_balanced/'
   dl = DataLoader(data_dir,split)

   filepath_list = dl.load_filelist()
   print('FileList Length:')
   print(len(filepath_list))

   embed_dict = dl.load_embedding_dict(fname='dict_trainBal_feats.pkl')
   print('Length of Feature Embeddings Dict:')  #note, there are about ~170 "inputs" per audio track, will elaborate more here tomorrow
   print(len(embed_dict))

   onehot_dict = dl.load_onehot_dict(label_list_fname='label_nums.csv')
   print('Num unique labels:')
   print(len(onehot_dict))

   label_dict = dl.load_label_dict(metadata_file='local_train_bal.csv')
   print('Length of embedding labels dict:')
   print(len(label_dict))
