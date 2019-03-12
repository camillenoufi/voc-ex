from argparse import ArgumentParser
import random
import numpy as np
from models import simpleKNN, VanillaCNN, train_model, eval_model
from load_data import DataLoader
from crnn import CRNN
import torch.utils.data as data_utils
import torch
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=np.nan)

def main():

    parser = ArgumentParser()
    parser.add_argument("--dir", help="location of local data dir", default=".")
    args = parser.parse_args()
    dir =  args.dir if args.dir else '/Users/sarahciresi/Desktop/CS224final/localTrainingData'
    print(dir)
    train_dicts, dev_dicts = load_train_and_dev(dir)
    #runKNN(train_dicts, dev_dicts) 
    #runKNN_withConcat(train_dicts, dev_dicts)
    #runVanillaCNN(train_dicts, dev_dicts)
    runCRNN(train_dicts, dev_dicts)



def load_train_and_dev(dir):


    #data_dir = '/Users/sarahciresi/Desktop/CS224final/localTrainingData'
    #data_dir = '/Users/sarahciresi/Desktop/CS224final/VocEx-local'
    data_dir = dir
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
    # Put together X_train by concatenating all feature slices of one file into one input      
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
    X_train = np.zeros((num_samples*num_slices, 96, 172)) # try with just 13 fq bins?  #96 ,172)) 
    y_train = np.zeros((num_samples*num_slices)) #, 96, 172))                           
    
    print("Loading X_train")
        
    sample_no = 0
    for file, feature_list in train_embed_dict.items():
        num_slices = len(feature_list)
        for slice in feature_list:
            # discard slices not of (96 x 172)                                                                           
            if slice.shape == (96, 172):
                X_train[sample_no] = slice #[:13,:]
                y_train[sample_no] = train_labels_range[train_label_dict[file]]
                sample_no += 1

    print("Loading X_dev")

    num_dev_samples = len(dev_embed_dict)
    X_dev  = np.zeros((num_dev_samples*num_slices, 96, 172)) # 96, 172))
    y_dev = np.zeros((num_dev_samples*num_slices))
    
    sample_no = 0
    for file, feature_list in dev_embed_dict.items():
        num_slices = len(feature_list)
        for slice in feature_list:
            if slice.shape == (96, 172):
                X_dev[sample_no] = slice  #[:13,:]
                y_dev[sample_no] = dev_labels_range[dev_label_dict[file]]
                sample_no += 1


    X_train = X_train.reshape(num_samples*num_slices, 96*172) # 96 * 172)
    X_dev = X_dev.reshape(num_dev_samples*num_slices, 96*172) # 96 * 172)

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
    num_samples =  len(train_embed_dict)
    num_total_slices = 6238   # all slices across all audio files
    fbins = 80
    time_steps = 204

    for k,v in train_onehot_dict.items():
        train_labels_range[k] = np.where(v==1)[0][0]

    for k,v in dev_onehot_dict.items():
        dev_labels_range[k] = np.where(v==1)[0][0]

    # Put together X_train   
    X_train = np.empty((fbins, time_steps))
    y_train = np.empty(1)
    #X_train = np.zeros((num_samples*num_slices, 96, 172))
    ##X_train = np.zeros((num_total_slices, fbins, time_steps)) 
    ##y_train = np.zeros((num_total_slices))
    #y_train = np.zeros((num_samples*num_slices)) #, 96, 172))
    
    print("Loading X_train")
    train_list_X = []
    train_list_y = []
    sample_no = 0
    sliced_class_label = 0 # keys should range from 0 to 479 mapping to 48 slices of 10 og classes
    for file, feature_list in train_embed_dict.items():
        feature_list = feature_list[4:]
        #num_slices_2 = len(feature_list)
        for i, slice in enumerate(feature_list):
            # discard slices not of (80 x    ) - this discards 1 slice in every file
            if slice.shape == (fbins, time_steps):
                train_list_X.append(slice)
                train_list_y.append(train_labels_range[train_label_dict[file]])
                #X_train[sample_no] = slice
                #y_train[sample_no] = train_labels_range[train_label_dict[file]] # 0-9
                sample_no += 1
    X_train = np.stack(train_list_X, axis = 2)          
    y_train = np.stack(train_list_y, axis = 0)
    #print("X shape: {} \n y shape: {} \n".format(X_train.shape, y_train.shape))

    # Do the same for X_dev
    print("Loading X_dev")
           
    num_dev_samples = len(dev_embed_dict)-1  ###
    #X_dev  = np.zeros((num_dev_samples*num_slices, 96, 172))
    #y_dev = np.zeros((num_dev_samples*num_slices)) 
    

    sample_no = 0
    dev_list_X = []
    dev_list_y = []
    for file, feature_list in dev_embed_dict.items():
        feature_list = feature_list[4:]
        #num_slices_2 = len(feature_list)
        for i, slice in enumerate(feature_list):
            if slice.shape == (fbins, time_steps):
                dev_list_X.append(slice)
                dev_list_y.append(dev_labels_range[dev_label_dict[file]])
                #X_dev[sample_no] = slice
                #y_dev[sample_no] = dev_labels_range[dev_label_dict[file]]                
        
                sample_no += 1
    X_dev = np.stack(dev_list_X, axis=2)
    y_dev = np.stack(dev_list_y, axis=0)


    # Batch the data for training and dev set
    X_train = torch.from_numpy(X_train)
    X_train = X_train.permute(2,0,1).unsqueeze(1).double()
    y_train = torch.from_numpy(y_train).long()

    X_dev = torch.from_numpy(X_dev).permute(2,0,1).unsqueeze(1).double()
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
    num_dev_samples = len(dev_embed_dict)-1 ###

    kernel_size = 3
    in_channels = 1
    num_filters = 18 # unused                                                                                        
    dropout = 0.5    # best is 0.0001 lr, 0.6 dropout, 8 epochs, up to 62.50% train ac, 14.0526% dev ac
    learning_rate = 0.0001
    num_epochs = 8
    
    #num_classes2 = 470  # jk there are 470 classes b/c we throw the shortest slice out ni every case

    train_loader, dev_loader = setup_data_CNN(train_dicts, dev_dicts)
    cnn = VanillaCNN(kernel_size, in_channels, num_filters, num_classes, dropout)
    train_model(cnn, train_loader, num_samples, learning_rate, num_epochs)
    torch.save(cnn.state_dict(), "trained_model_params.bin")
    eval_model(cnn, dev_loader)



def runCRNN(train_dicts, dev_dicts):
    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    num_classes = len(train_onehot_dict)
    num_samples = len(train_embed_dict)
    num_dev_samples = len(dev_embed_dict)

    
    # Hyper parameters
    dropout_rate = 0.3
    embed_size = 128
    hidden_size = 128
    num_layers = 2
    input_size = 42
    num_epochs = 10
    learning_rate = 0.001
    sequence_length = 172
    num_classes = 10
    batch_size = 100
    num_epochs = 8


    train_loader, dev_loader = setup_data_CNN(train_dicts, dev_dicts)
    cnn = CRNN(input_size, embed_size, hidden_size, num_layers, num_classes, dropout_rate)
    train_model(cnn, train_loader, num_samples, learning_rate, num_epochs)
    torch.save(cnn.state_dict(), "trained_model_params.bin")
    eval_model(cnn, dev_loader)


if __name__ == '__main__':
    main()
