from argparse import ArgumentParser
import random
import numpy as np
from models import simpleKNN, VanillaCNN, train_model, eval_model
from load_data import DataLoader
import torch.utils.data as data_utils
import torch
from sklearn.metrics import accuracy_score

def main():
    
    train_dicts, dev_dicts = load_train_and_dev()
    runKNN(train_dicts, dev_dicts) 
    #runKNN_withConcat(train_dicts, dev_dicts)
    #runVanillaCNN(train_dicts, dev_dicts)



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




def runKNN_withConcat(train_dicts, dev_dicts):
    '''
    Runs KNN model with input that is concatenated spectogram chunks, i.e. input is of size 
    (batch size   x  96 fq   x   172  timesteps * 48 numslices/file) (or about -- use 8053 instead because its the min) 
    '''
    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    
    for k,v in train_onehot_dict.items():
        train_labels_range[k] = np.where(v==1)[0][0]

    for k,v in dev_onehot_dict.items():
        dev_labels_range[k] = np.where(v==1)[0][0]

    num_samples = len(train_embed_dict)
    num_slices = 48
    X_train = np.zeros((num_samples, 96, 8053))                                                                                                                                               
    y_train = np.zeros((num_samples))   
    print("Loading X_train")

    idx = 0                                                                                                                                                                                 
    min_c_length = 10000   
    # Put together X_train                                                                                                                                                                    
    # Concatenate all feature slices of one file into one input      
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

    num_dev_samples = len(dev_embed_dict)  
    X_dev = np.zeros((num_dev_samples, 96, 8053)) 
    y_dev = np.zeros((num_dev_samples))     

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
        
    X_train = X_train.reshape(num_samples, 96 * 8053)
    X_dev = X_dev.reshape(num_dev_samples, 96 * 8053)

    num_classes = len(train_onehot_dict)
    weighting = "uniform"
    knn = simpleKNN(num_classes, weighting)

    print("Fitting Simple KNN Model")
    knn.fit(X_train, y_train)

    print("Predicting on the dev set")
    y_pred = knn.predict(X_dev)
    score = accuracy_score(y_dev, y_pred)
    print("Accuracy score: {} ".format(score))


def runKNN(train_dicts, dev_dicts):
    ''' Runs KNN model with all chunks treated as separate input with their own labels 
    i.e. input is of shape (batchsize * 48 num_slices,  96 fq bins,   172 time) 
    '''

    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    
    # convert labels from one-hot-encoding into range (0,num_class) - should be done in data loader but here for now
    for k,v in train_onehot_dict.items():
        train_labels_range[k] = np.where(v==1)[0][0]

    for k,v in dev_onehot_dict.items():
        dev_labels_range[k] = np.where(v==1)[0][0]

    # Put together X_train
    num_samples = len(train_embed_dict)
    num_slices = 48
    X_train = np.zeros((num_samples*num_slices, 13, 172)) # try with just 13 fq bins?  #96 ,172)) 
    y_train = np.zeros((num_samples*num_slices)) #, 96, 172))                                                           
    
    print("Loading X_train")
        
    sample_no = 0
    for file, feature_list in train_embed_dict.items():
        num_slices = len(feature_list)
        for slice in feature_list:
            # discard slices not of (96 x 172)                                                                           
            if slice.shape == (96, 172):
                X_train[sample_no] = slice[:13,:]
                y_train[sample_no] = train_labels_range[train_label_dict[file]]
                sample_no += 1

    print("Loading X_dev")

    num_dev_samples = len(dev_embed_dict)
    X_dev  = np.zeros((num_dev_samples*num_slices, 13, 172)) # 96, 172))
    y_dev = np.zeros((num_dev_samples*num_slices))
    
    sample_no = 0
    for file, feature_list in dev_embed_dict.items():
        num_slices = len(feature_list)
        for slice in feature_list:
            if slice.shape == (96, 172):
                X_dev[sample_no] = slice[:13,:]
                y_dev[sample_no] = dev_labels_range[dev_label_dict[file]]
                sample_no += 1


    X_train = X_train.reshape(num_samples*num_slices, 13*172) # 96 * 172)
    X_dev = X_dev.reshape(num_dev_samples*num_slices, 13*172) # 96 * 172)

    num_classes = len(train_onehot_dict)
    weighting = "uniform"
    knn = simpleKNN(num_classes, weighting)
    print("Fitting Simple KNN Model")
    knn.fit(X_train, y_train)
    print("Predicting on the dev set")
    y_pred = knn.predict(X_dev)
    score = accuracy_score(y_dev, y_pred)
    print("Accuracy score: {} ".format(score))






def setup_data_CNN(train_dicts, dev_dicts):
    '''
    Function that handles all the extra data set up before training / evaluting the CNN
    '''
    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    num_classes = len(train_onehot_dict)

    for k,v in train_onehot_dict.items():
        train_labels_range[k] = np.where(v==1)[0][0]

    for k,v in dev_onehot_dict.items():
        dev_labels_range[k] = np.where(v==1)[0][0]

    # Put together X_train   
    num_samples = len(train_embed_dict)
    num_slices = 48  

    X_train = np.zeros((num_samples*num_slices, 96, 172))
    y_train = np.zeros((num_samples*num_slices)) #, 96, 172))

    print("Loading X_train")

    sample_no = 0
    for file, feature_list in train_embed_dict.items():
        num_slices = len(feature_list)
        for slice in feature_list:
            # discard slices not of (96 x 172) 
            if slice.shape == (96, 172):
                X_train[sample_no] = slice
                y_train[sample_no] = train_labels_range[train_label_dict[file]] 
                sample_no += 1
                
    # Do the same for X_dev
    print("Loading X_dev")

    num_dev_samples = len(dev_embed_dict)
    X_dev  = np.zeros((num_dev_samples*num_slices, 96, 172))
    y_dev = np.zeros((num_dev_samples*num_slices)) 
    
    sample_no = 0
    for file, feature_list in dev_embed_dict.items():
        num_slices = len(feature_list)
        for slice in feature_list:
            if slice.shape == (96, 172):
                X_dev[sample_no] = slice
                y_dev[sample_no] = dev_labels_range[dev_label_dict[file]]
                sample_no += 1

    kernel_size = 3
    in_channels = 1
    num_filters = 18 # unused
    dropout = 0.3
    learning_rate = 0.001 
    num_epochs = 10

    # Batch the data for training and dev set
    X_train = torch.from_numpy(X_train).unsqueeze(1).double()
    y_train = torch.from_numpy(y_train).long()
    X_dev = torch.from_numpy(X_dev).unsqueeze(1).double()
    y_dev = torch.from_numpy(y_dev).long()
    

    train_data = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size = 80, shuffle=True)
    dev_data = data_utils.TensorDataset(X_dev, y_dev)
    dev_loader = data_utils.DataLoader(dev_data, batch_size = 80, shuffle=True)

    return train_loader, dev_loader



def runVanillaCNN(train_dicts, dev_dicts):

    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    num_classes = len(train_onehot_dict)
    num_samples = len(train_embed_dict)
    num_dev_samples = len(dev_embed_dict)

    kernel_size = 3
    in_channels = 1
    num_filters = 18 # unused                                                                                        
    dropout = 0.5
    learning_rate = 0.001
    num_epochs = 8

    train_loader, dev_loader = setup_data_CNN(train_dicts, dev_dicts)
    cnn = VanillaCNN(kernel_size, in_channels, num_filters, num_classes, dropout)
    train_model(cnn, train_loader, num_samples, learning_rate, num_epochs)
    torch.save(cnn.state_dict(), "trained_model_params.bin")
    eval_model(cnn, dev_loader)




if __name__ == '__main__':
    main()
