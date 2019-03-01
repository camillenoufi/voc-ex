import os
import numpy as np
from load_data import DataLoader

# Main TEST Routine
if __name__ == '__main__':

   #### CHANGE THIS PATH FOR YOUR LOCAL MACHINE'S COPY OF THE DATA FOLDER ###############
   data_dir = '/Users/camillenoufi/Documents/datasets/VocEx-local/'
   ####################

   # Specify which partition you want (ie. local_balanced vs local_representative vs. development(validation) set)
   partition = '/local_balanced/'
   # Declare an instance of the data loader
   dl = DataLoader(data_dir,partition)


   """
   Load list of files in the local training(or dev) set folder :
   Returns list[str]
   - len(list) is number of different audio files in the training(or dev) set
   - str is a string of the specific /path/to/audiofile.wav
   """
   filepath_list = dl.load_filelist()
   print('FileList Length:')
   print(len(filepath_list))



   """
   Load Pre-computed log-mel-spectrogram features:
   Call with Pickle (pkl) filename containing saved mel features
   Returns dict{key:value}:
   - key: (str) is a string of a specific /path/to/audiofile.wav in the local training(or dev) set folder
   - value: (list[ 2D np.array ])
            * len(list) = total number of feature arrays (inputs) created from one audio file
            * np.array = 2D slice of mel spectrogram (this is a single "input" into the neural network)
            * np.array.shape = (96 fbins, 172 time-frames)   **each time frame is ~10ms
   """
   embed_dict = dl.load_embedding_dict(fname='dict_trainBal_feats.pkl')
   print('Length of Feature Embeddings Dict:')
   print(len(embed_dict))


   """
   Loads dictionary of numeric country label "truths" for each specific audio file:
   Call with .csv filename containing the metadata for all audio files in the set of interest
   Returns dict{key:value}:
   - key: (str) is a string of a specific /path/to/audiofile.wav in the local training(or dev) set folder
   - value: (int) is the numeric country/accent label for this specific audio file
            * use this as the 'key' to acces the one-hot vector representation of the label
            * all feature inputs created from a specific audio file will use that audio file's numeric label

   """
   label_dict = dl.load_label_dict(metadata_file='local_train_bal.csv')
   print('Length of embedding labels dict:')
   print(len(label_dict))



    """
    Load Dictionary of {numeric country labels:one-hot vector representations}:
    Call with .csv filename containing list of all possible "truth" labels
    Returns dict{key:value}:
    - key: (int) is a number representing a specific country/accent label (arbitrary)
    - value: (1D np.array)
           * len(np.array) = length of one-hot vector
           * np.array = one-hot vector representing the "label" answer.  Use this to compare to the softmax output
           * np.array.shape = (1, nPossibleLabels)
    """
    onehot_dict = dl.load_onehot_dict(label_list_fname='label_nums.csv')
    print('Num unique labels:')
    print(len(onehot_dict))
