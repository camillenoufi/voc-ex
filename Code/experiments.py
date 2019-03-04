from argparse import ArgumentParser
import random
import numpy as np
from models import simpleKNN, VanillaCNN, train_model
from load_data import DataLoader


def main():
    
    train_dicts, dev_dicts = load_train_and_dev()
    #runKNN(train_dicts, dev_dicts)
    runVanillaCNN(train_dicts, dev_dicts)



def load_train_and_dev():

    data_dir = '/Users/sarahciresi/Desktop/CS224final/localTrainingData'
    train_partition = 'local_balanced/'
    dl_train = DataLoader(data_dir, train_partition)
    filepath_list_train = dl_train.load_filelist()

    dev_partition = 'local_dev/'
    dl_dev = DataLoader(data_dir, dev_partition)
    filepath_list_dev = dl_dev.load_filelist()

    '''Load precomputed log-mel-spectrogram features.  Returns dict{key:value}: 
    - key: (str) is a string of a specific /path/to/audiofile.wav in the local training(or dev) set folder    
    - value: (list[ 2D np.array ])  
    * len(list) = total number of feature arrays (inputs) created from one audio file    
    * np.array = 2D slice of mel spectrogram (this is a single "input" into the neural network)                 
    * np.array.shape = (96 fbins, 172 time-frames)   **each time frame is ~10ms     
    '''
    train_embed_dict = dl_train.load_embedding_dict(fname='dict_trainBal_feats.pkl')   
    train_embed_dict = dl_train.convert_embed_dict_to_local(train_embed_dict)

    '''key: (str) is a string of a specific /path/to/audiofile.wav in the local training(or dev) set folder  
    value: (int) is the numeric country/accent label for this specific audio file  
    use this as the 'key' to acces the one-hot vector representation of the label'''
    train_label_dict = dl_train.load_label_dict(metadata_file='local_train_bal.csv')

    '''key: (int) is a number representing a specific country/accent label (arbitrary)     
    value: (1D np.array) : one-hot vector representing the "label" answer.  Use this to compare to the softmax output '''
    train_onehot_dict = dl_train.load_onehot_dict(label_list_fname='label_nums.csv')


    dev_embed_dict = dl_dev.load_embedding_dict(fname='dict_dev_feats.pkl')
    dev_embed_dict = dl_dev.convert_embed_dict_to_local(dev_embed_dict)

    dev_label_dict = dl_dev.load_label_dict(metadata_file='local_dev.csv')
    dev_onehot_dict = dl_dev.load_onehot_dict(label_list_fname='label_nums.csv')

    train_dicts = train_embed_dict, train_label_dict, train_onehot_dict
    dev_dicts = dev_embed_dict, dev_label_dict, dev_onehot_dict

    
    return train_dicts, dev_dicts





def runKNN(train_dicts, dev_dicts):

    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    
    # convert labels from one-hot-encoding into range (0,num_class) 
    # really should be done in data loader but here for now
    for k,v in train_onehot_dict.items():
        train_labels_range[k] = np.where(v==1)[0][0]

    for k,v in dev_onehot_dict.items():
        dev_labels_range[k] = np.where(v==1)[0][0]

    # Put together X_train
    # Concatenate all feature slices of one file into one input 
    num_samples = len(train_embed_dict)
    len_slice = np.zeros((96, 172)) # (96 fbins, 172 time-frames)   
    X_train = np.zeros((num_samples, 96, 8053))
    y_train = np.zeros((num_samples))

    print("Loading X_train")

    idx = 0
    min_c_length = 10000
    for file, feature_list in train_embed_dict.items():
        num_slices = len(feature_list)
        concat_features = None
        first = True
        for slice in feature_list:
            if first == True:
                concat_features = slice
                first = False
            else:
                concat_features = np.concatenate((concat_features, slice), axis=1)
        concat_features = concat_features[:,:8053]
        X_train[idx] = concat_features
        y_train[idx] = train_labels_range[train_label_dict[file]]
        idx+=1 

    print("Loading X_dev")

    # Do the same for X_dev
    num_samples = len(dev_embed_dict)
    len_slice = np.zeros((96, 172)) # (96 fbins, 172 time-frames)                                                                           
    X_dev  = np.zeros((num_samples, 96, 8053))
    y_dev = np.zeros((num_samples))
    idx = 0
    min_c_length = 10000
    for file, feature_list in dev_embed_dict.items():
        num_slices = len(feature_list)
        concat_features = None
        first = True
        for slice in feature_list:
            if first == True:
                concat_features = slice
                first = False
            else:
                concat_features = np.concatenate((concat_features, slice), axis=1)
        concat_features = concat_features[:,:8053]
        X_dev[idx] = concat_features
        y_dev[idx] = dev_labels_range[dev_label_dict[file]]
        idx+=1


    num_classes = len(train_onehot_dict)
    weighting = "uniform"
    knn = simpleKNN(num_classes, weighting)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_dev)
    score = metrics.accuracy_score(y_dev, y_pred)
             



def runVanillaCNN(train_dicts, dev_dicts):
    
    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    num_classes = len(train_onehot_dict)

    # Put together X_train   
    # Concatenate all feature slices of one file into one input 
    num_samples = len(train_embed_dict)
    #len_slice = np.zeros((96, 172)) # (96 fbins, 172 time-frames)                                                                               
    X_train = np.zeros((num_samples, 96, 8053))
    y_train = np.zeros((num_samples, num_classes))

    print("Loading X_train")

    idx = 0
    min_c_length = 10000
    for file, feature_list in train_embed_dict.items():
        num_slices = len(feature_list)
        concat_features = None
        first = True
        for slice in feature_list:
            if first == True:
                concat_features = slice
                first = False
            else:
                concat_features = np.concatenate((concat_features, slice), axis=1)
        concat_features = concat_features[:,:8053]
        X_train[idx] = concat_features
        y_train[idx] = train_onehot_dict[train_label_dict[file]].flatten()
        idx+=1


    # Do the same for X_dev
    print("Loading X_dev")

    num_samples = len(dev_embed_dict)
    X_dev  = np.zeros((num_samples, 96, 8053))
    y_dev = np.zeros((num_samples, num_classes))

    idx = 0
    min_c_length = 10000
    for file, feature_list in dev_embed_dict.items():
        num_slices = len(feature_list)
        concat_features = None
        first = True
        for slice in feature_list:
            if first == True:
                concat_features = slice
                first = False
            else:
                concat_features = np.concatenate((concat_features, slice), axis=1)
        concat_features = concat_features[:,:8053]
        X_dev[idx] = concat_features
        y_dev[idx] = dev_onehot_dict[dev_label_dict[file]].flatten()
        idx+=1

    kernel_size = 3
    in_channels = 96 #? 
    num_filters = 50 #unused
    dropout = 0.3
    learning_rate = 0.01 
    num_epochs = 50

    # Batch the data for training and dev sets
    train_data = (X_train, y_train)
    dev_data = (X_dev, y_dev)
    
    cnn = VanillaCNN(kernel_size, in_channels, num_filters, num_classes, dropout)
    train_model(cnn, train_data, num_samples, learning_rate, num_epochs)

if __name__ == '__main__':
    main()
