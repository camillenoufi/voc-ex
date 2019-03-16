from argparse import ArgumentParser
import random
import numpy as np
from models import simpleKNN, VanillaCNN, train_model, eval_model
from load_data import DataLoader
from crnn import CRNN, CRNNNoLSTM
import torch.utils.data as data_utils
import torch
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=np.nan)

def main():
    parser = ArgumentParser()
    parser.add_argument("--dir", help="location of local data dir", default=".")
    parser.add_argument("--train_dir", help="name of the train partition folder, must be a subfolder of 'dir' ", default="train")
    parser.add_argument("--dev_dir", help="name of the dev partition folder, must be a subfolder of 'dir' ", default="dev")
    parser.add_argument("--model", help="one of  knn, cnn, crnn or nolstm ", default="crnn")
    parser.add_argument("--vm_flag", help="0 = local, 1 = Azure VM", default=0)
    args = parser.parse_args()
    dir =  args.dir if args.dir else '/home/group/dataset'  #dataset location in Azure VM
    train_dir =  args.train_dir if args.train_dir else 'train'  #dataset location in Azure VM
    dev_dir =  args.dev_dir if args.dev_dir else 'dev'  #dataset location in Azure VM
    vm_flag = args.vm_flag if args.dev_dir else '0'
    print(dir)

    if ( vm_flag=='1' and torch.cuda.is_available() ):
        print("Training on Azure VM GPU...")
        device = torch.device('cuda')
    else:
        print("Training on local datasets...")
        device = torch.device('cpu')

    train_dicts, dev_dicts = load_train_and_dev(dir,train_dir,dev_dir,vm_flag)
    model = args.model
    fbins = 80
    time_steps = 204;
    input_dims = (fbins,time_steps)

    if args.model == 'knn':
        print("Running KNN...")
        runKNN(train_dicts, dev_dicts, input_dims, device)
        #runKNN_withConcat(train_dicts, dev_dicts)
    if args.model == 'cnn':
        print("Running CNN...")
        runVanillaCNN(train_dicts, dev_dicts, input_dims, device)
    if args.model == 'crnn':
        print("Running CRNN...")
        runCRNN(train_dicts, dev_dicts, input_dims, device)

    if args.model == 'nolstm':
        print("Running CRNN without LSTM ...")
        runCRNN(train_dicts, dev_dicts, input_dims, device, True)


def load_train_and_dev(dir,train_dir,dev_dir,vm_flag):

    print("Train:")
    dl_train = DataLoader(dir, train_dir)
    filepath_list_train = dl_train.load_filelist()

    print("Validation:")
    dl_dev = DataLoader(dir, dev_dir)
    filepath_list_dev = dl_dev.load_filelist()

    '''
    Set label/metadata filenames depending on run version (local or full-azure)
    '''
    if (vm_flag=='1'):
        train_fname = 'train_melfeats_balanced.pkl'
        train_labels_fname = 'trainBalanced_labels.csv'
        dev_fname = 'dev_melfeats_all.pkl'
        dev_labels_fname = 'dev_labels.csv'
    else:
        train_fname = 'dict_trainBal_feats.pkl'
        train_labels_fname = 'local_train_bal.csv'
        dev_fname = 'dict_dev_feats.pkl'
        dev_labels_fname = 'local_dev.csv'
    # For both versions:
    label_list_fname = 'label_nums.csv'


    '''Load precomputed log-mel-spectrogram features.  Returns dict{key:value}:
    - key: (str) is a string of a specific /path/to/audiofile.wav in the local training(or dev) set folder
    - value: (list[ 2D np.array ])
    * len(list) = total number of feature arrays (inputs) created from one audio file
    * np.array = 2D slice of mel spectrogram (this is a single "input" into the neural network)
    * np.array.shape = (fbins, time-frames)   **each time frame is ~10ms
    '''
    train_embed_dict = dl_train.convert_embed_dict_to_local(dl_train.load_embedding_dict(fname=train_fname))

    '''key: (str) is a string of a specific /path/to/audiofile.wav in the local training(or dev) set folder
    value: (int) is the numeric country/accent label for this specific audio file
    use this as the 'key' to acces the one-hot vector representation of the label'''
    train_label_dict = dl_train.load_label_dict(metadata_file=train_labels_fname)

    '''key: (int) is a number representing a specific country/accent label (arbitrary)
    value: (1D np.array) : one-hot vector representing the label.  Use this to compare to the softmax output '''
    train_onehot_dict = dl_train.load_onehot_dict(label_list_fname=label_list_fname)

    dev_embed_dict = dl_dev.convert_embed_dict_to_local(dl_dev.load_embedding_dict(fname=dev_fname))

    dev_label_dict = dl_dev.load_label_dict(metadata_file=dev_labels_fname)
    dev_onehot_dict = dl_dev.load_onehot_dict(label_list_fname=label_list_fname)

    train_dicts = train_embed_dict, train_label_dict, train_onehot_dict
    dev_dicts = dev_embed_dict, dev_label_dict, dev_onehot_dict

    #test
    #print(len(train_embed_dict))
    #print(len(train_label_dict))
    #print(len(dev_embed_dict))
    #print(len(dev_label_dict))

    return train_dicts, dev_dicts


def setup_data_CNN(data_dicts, input_dims, batch_size):
    '''
    Function that handles all the extra data set up before training / evaluting the CNN
    '''
    embed_dict, label_dict, onehot_dict = data_dicts
    labels_range = {}
    fbins = input_dims[0]
    time_steps = input_dims[1]
    start_frame = 4;  #chop off first 4 frames of each input

    for k,v in onehot_dict.items():
        labels_range[k] = np.where(v==1)[0][0]


    print("...Loading input and labels")
    list_X = []
    list_y = []
#    file_no = 0
    for file, feature_list in embed_dict.items():
#        file_no+=1
#        if file_no == 1000:
#            print('At file 1000')
#            break
        feature_list = feature_list[start_frame:]
        for i, slice in enumerate(feature_list):
            #print(i)
            #if (i%1000==0):
            #    print(slice.shape)
            if slice.shape == (fbins, time_steps):
#                print(file_no, i)
                list_X.append(slice)
                list_y.append(labels_range[label_dict[file]])

    X = np.stack(list_X, axis = 2)
    y = np.stack(list_y, axis = 0)

    # Conver to torch tensors, Batch the data for training and dev set
    X = torch.from_numpy(X).permute(2,0,1).unsqueeze(1).double()
    y = torch.from_numpy(y).long()



    print('Input tensor:')
    print(X.size())
    print('Label tensor:')
    print(y.size())

    dataset = data_utils.TensorDataset(X, y)
    loader = data_utils.DataLoader(dataset, batch_size = batch_size, shuffle=True)
    return loader


def runVanillaCNN(train_dicts, dev_dicts, input_dims, device):

    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    num_classes = len(train_onehot_dict)
    num_samples = len(train_embed_dict)
    num_dev_samples = len(dev_embed_dict) ###

    # Hyper-parameters
    batch_size = 128;
    kernel_size = 3
    in_channels = 1
    num_filters = 64
    dropout_rate = 0.3
    learning_rate = 0.001
    num_epochs = 40
    # best is 0.0001 lr, 0.6 dropout, 8 epochs, up to 62.50% train ac, 14.0526% dev ac

    # Format Data for Train and Eval
    print("Setting up TRAINING data for model...")
    train_loader = setup_data_CNN(train_dicts, input_dims, batch_size)
    print("Setting up VALIDATION data for model...")
    dev_loader = setup_data_CNN(dev_dicts, input_dims, batch_size)

    # Initialize and Train Model
    cnn = VanillaCNN(kernel_size, in_channels, num_filters, num_classes, dropout_rate)
    #cnn = cnn.to(device)
    train_model(cnn, train_loader, num_samples, learning_rate, num_epochs, device)
    torch.save(cnn.state_dict(), "trained_cnn_model_params.bin")

    # Evaluate Model
    eval_model(cnn, dev_loader, device)



def runCRNN(train_dicts, dev_dicts, input_dims, device, nolstm = False):
    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    num_classes = len(train_onehot_dict)
    num_samples = len(train_embed_dict)
    num_dev_samples = len(dev_embed_dict)


    # Hyper parameters
    batch_size = 128
    dropout_rate = 0.3
    embed_size = 64
    hidden_size = 64
    num_layers = 2
    input_size = 51 # input size for the LSTM
    learning_rate = 0.001
    num_epochs = 1

    # Format Data for Train and Eval
    print("Setting up TRAINING data for model...")
    train_loader = setup_data_CNN(train_dicts, input_dims, batch_size)
    print("Setting up VALIDATION data for model...")
    dev_loader = setup_data_CNN(dev_dicts, input_dims, batch_size)

    # Initialize and Train Model
    if (nolstm):
        crnn = CRNNNoLSTM(input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate)
    else:
        crnn = CRNN(input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate)
    
    train_model(crnn, train_loader, num_samples, learning_rate, num_epochs, device)
    torch.save(crnn.state_dict(), "trained_crnn_model_params.bin")

    # Evaluate Model
    eval_model(crnn, dev_loader, device)




# OLD KNN CODE - DOES NOT ADJUST FOR NEW INPUT SIZES!!!
def runKNN_withConcat(train_dicts, dev_dicts, device):
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


if __name__ == '__main__':
    main()
