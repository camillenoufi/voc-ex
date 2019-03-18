from argparse import ArgumentParser
import random
import numpy as np
from models import VanillaCNN, train_model, eval_model, test_model
from load_data import DataLoader
from crnn import CRNN, CRNNNoLSTM
from knn import simpleKNN
import torch.utils.data as data_utils
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

np.set_printoptions(threshold=np.nan)

def main():
    parser = ArgumentParser()
    parser.add_argument("--dir", help="location of local data dir", default=".")
    parser.add_argument("--train_dir", help="name of the train partition folder, must be a subfolder of 'dir' ", default="train")
    parser.add_argument("--dev_dir", help="name of the dev partition folder, must be a subfolder of 'dir' ", default="dev")
    parser.add_argument("--test_dir", help="name of the dev partition folder, must be a subfolder of 'dir' ", default="test")
    parser.add_argument("--model", help="one of  knn, cnn, crnn or nolstm ", default="crnn")
    parser.add_argument("--vm_flag", help="0 = local, 1 = Azure VM", default=0)
    parser.add_argument("--test_flag", help="0 = train model, 1 = test model", default=0)
    args = parser.parse_args()
    dir =  args.dir if args.dir else '/home/group/dataset'  #dataset location in Azure VM
    train_dir =  args.train_dir if args.train_dir else 'train'  #dataset location in Azure VM
    dev_dir =  args.dev_dir if args.dev_dir else 'dev'  #dataset location in Azure VM
    test_dir =  args.test_dir if args.test_dir else 'test'  #dataset location in Azure VM
    vm_flag = args.vm_flag if args.vm_flag else '0'
    test_flag = args.test_flag if args.test_flag else '0'
    print(dir)

    if ( vm_flag=='1' and torch.cuda.is_available() ):
        print("Training on Azure VM GPU...")
        device = torch.device('cuda')
    else:
        print("Training on local datasets...")
        device = torch.device('cpu')

    # load data dictionaries
    train_dicts = None
    test_dicts = None
    dev_dicts = None
    if ( test_flag=='1'):
        print("Test Mode...")
        test_dicts = load_test(dir,test_dir,vm_flag)
    else:
        print("Train Mode...")
        train_dicts, dev_dicts = load_train_and_dev(dir,train_dir,dev_dir,vm_flag)

    model = args.model
    fbins = 80
    time_steps = 204;
    input_dims = (fbins,time_steps)

    if args.model == 'knn':
        print("Running KNN...")
        runKNN(train_dicts, dev_dicts, test_dicts, input_dims, device, test_flag)
        #runKNN_withConcat(train_dicts, dev_dicts)
    if args.model == 'cnn':
        print("Running CNN...")
        runVanillaCNN(train_dicts, dev_dicts, test_dicts, input_dims, device, test_flag)
    if args.model == 'crnn':
        print("Running CRNN...")
        runCRNN(train_dicts, dev_dicts, test_dicts, input_dims, device, test_flag)

    if args.model == 'nolstm':
        print("Running CRNN without LSTM ...")
        runCRNN(train_dicts, dev_dicts, test_dicts, input_dims, device, test_flag, True)



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

    return train_dicts, dev_dicts

def load_test(dir,test_dir,vm_flag):

    print("Test:")
    dl_test = DataLoader(dir, test_dir)
    filepath_list = dl_test.load_filelist()
    '''
    Set label/metadata filenames depending on run version (local or full-azure)
    '''
    if (vm_flag=='1'):
        test_fname = 'test_melfeats_all.pkl'
        test_labels_fname = 'test_labels.csv'
    else:
        test_fname = 'dict_test_feats.pkl'
        test_labels_fname = 'local_test.csv'

    label_list_fname = 'label_nums.csv'

    embed_dict = dl_test.convert_embed_dict_to_local(dl_test.load_embedding_dict(fname=test_fname))

    label_dict = dl_test.load_label_dict(metadata_file=test_labels_fname)
    onehot_dict = dl_test.load_onehot_dict(label_list_fname=label_list_fname)

    dicts = embed_dict, label_dict, onehot_dict
    return dicts


def setup_data_CNN(data_dicts, input_dims, batch_size, train=True):
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
    for file, feature_list in embed_dict.items():
        feature_list = feature_list[start_frame:]
        for i, slice in enumerate(feature_list):
#            print(slice.shape)
            if slice.shape == (fbins, time_steps):
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

    if train:
        dataset = data_utils.TensorDataset(X, y)

        valid_size = 0.2 # Use 20% for validation and early stopping.
        num_train =  len(X)
        split = int(np.floor(valid_size * num_train))
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]


        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = data_utils.DataLoader(dataset,
                                            batch_size = batch_size,
                                            sampler=train_sampler,
                                            num_workers=0)
        valid_loader = data_utils.DataLoader(dataset,
                                            batch_size = batch_size,
                                            sampler=valid_sampler,
                                            num_workers=0)
        return (train_loader, valid_loader)

    else:
        dataset = data_utils.TensorDataset(X, y)
        loader = data_utils.DataLoader(dataset, batch_size = batch_size, shuffle=True)
        return loader, labels_range


def runVanillaCNN(train_dicts, dev_dicts, test_dicts, input_dims, device, test_flag):

    #Architecture hyperparams
    kernel_size = 3
    in_channels = 1
    num_filters = 32
    dropout_rate = 0.3
    model_file = "trained_cnn_model_params.pt"

    if (test_flag=='1'):
        test_embed_dict, test_label_dict, test_onehot_dict = test_dicts
        num_classes = len(test_onehot_dict)
        print("Setting up TEST data...")
        test_loader, label_set = setup_data_CNN(test_dicts, input_dims, batch_size = 128, train=False)
        # Initialize Model
        cnn = VanillaCNN(kernel_size, in_channels, num_filters, num_classes, dropout_rate)
        test_model(cnn, model_file, test_loader, device, label_set)

    else:
        train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
        dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
        train_labels_range, dev_labels_range = {}, {}
        num_classes = len(train_onehot_dict)
        num_samples = len(train_embed_dict)
        num_dev_samples = len(dev_embed_dict) ###


        # Learning Hyper-parameters
        batch_size = 128
        learning_rate = 0.001
        num_epochs = 40

        # Format Data for Train and Eval
        print("Setting up TRAINING data for model...")
        train_loader, valid_loader = setup_data_CNN(train_dicts, input_dims, batch_size)
        print("Setting up VALIDATION data for model...")
        dev_loader, label_set = setup_data_CNN(dev_dicts, input_dims, batch_size, train=False)
        # Initialize Model
        cnn = VanillaCNN(kernel_size, in_channels, num_filters, num_classes, dropout_rate)
        cnn = train_model(cnn, train_loader, valid_loader, num_samples, learning_rate, num_epochs, device, "cnn")
        torch.save(cnn.state_dict(), model_file)

        # Evaluate Model
        eval_model(cnn, dev_loader, device, label_set)



def runCRNN(train_dicts, dev_dicts, test_dicts, input_dims, device, test_flag, nolstm = False):

    #Architecture hyperparams
    dropout_rate = 0.3
    embed_size = 32
    hidden_size = 32
    num_layers = 2
    input_size = 51 # input size for the LSTM
    model_file = "trained_crnn_model_params.pt"

    if (test_flag=='1'):
        test_embed_dict, test_label_dict, test_onehot_dict = test_dicts
        num_classes = len(test_onehot_dict)
        print("Setting up TEST data...")
        test_loader, label_set = setup_data_CNN(test_dicts, input_dims, batch_size = 128, train=False)
        # Initialize Model
        if (nolstm):
            crnn = CRNNNoLSTM(input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate)
        else:
            crnn = CRNN(input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate)
        test_model(crnn, model_file, test_loader, device, label_set)

    else:
        train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
        dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
        train_labels_range, dev_labels_range = {}, {}
        num_classes = len(train_onehot_dict)
        num_samples = len(train_embed_dict)
        num_dev_samples = len(dev_embed_dict)


        # training Hyper parameters
        batch_size = 128
        learning_rate = 0.001
        num_epochs = 40

        # Format Data for Train and Eval
        print("Setting up TRAINING data for model...")
        train_loader, valid_loader = setup_data_CNN(train_dicts, input_dims, batch_size)

        print("Train dataset...", len(train_loader.dataset))
        print("Valid dataset...", len(valid_loader.dataset))


        print("Setting up VALIDATION data for model...")
        dev_loader, label_set = setup_data_CNN(dev_dicts, input_dims, batch_size, train=False)
        model_name = "crnn"
        # Initialize Model
        if (nolstm):
            crnn = CRNNNoLSTM(input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate)
            model_name = "nolstm"
        else:
            crnn = CRNN(input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate)
        crnn = train_model(crnn, train_loader, valid_loader, num_samples, learning_rate, num_epochs, device, model_name)
        torch.save(crnn.state_dict(), model_file)

        # Evaluate Model
        eval_model(crnn, dev_loader, device, label_set)


def runKNN(train_dicts, dev_dicts, test_dicts, input_dims, batch_size, test_flag):
    ''' Runs KNN model with all chunks treated as separate input with their own labels
    '''

    train_embed_dict, train_label_dict, train_onehot_dict = train_dicts
    dev_embed_dict, dev_label_dict, dev_onehot_dict = dev_dicts
    train_labels_range, dev_labels_range = {}, {}
    fbins, time_steps = input_dims
    ftrunc = 40;
    start_frame = 4;  #chop off first 4 frames of each input
    frame_skip = 2;

    # convert labels from one-hot-encoding into range (0,num_class) - should be done in data loader but here for now
    for k,v in train_onehot_dict.items():
        train_labels_range[k] = np.where(v==1)[0][0]

    for k,v in dev_onehot_dict.items():
        dev_labels_range[k] = np.where(v==1)[0][0]


    print("...Loading training input and labels")
    list_X = []
    list_y = []
    for file, feature_list in train_embed_dict.items():
        feature_list = feature_list[start_frame:]
        for slice in feature_list:
            if slice.shape == (fbins, time_steps):
                i = 0
                while i<time_steps:
                    list_X.append(slice[:ftrunc,i])
                    list_y.append(train_labels_range[train_label_dict[file]])
                    i += frame_skip

    X_train = np.stack(list_X, axis = 1)
    y_train = np.stack(list_y, axis = 0)

    print("...Loading dev input and labels")
    list_X = []
    list_y = []
    for file, feature_list in dev_embed_dict.items():
        feature_list = feature_list[start_frame:]
        for slice in feature_list:
            if slice.shape == (fbins, time_steps):
                i = 0
                while i<time_steps:
                    list_X.append(slice[:ftrunc,i])
                    list_y.append(train_labels_range[dev_label_dict[file]])
                    i += frame_skip


    X_dev = np.stack(list_X, axis = 1)
    y_dev = np.stack(list_y, axis = 0)

    X_train =np.transpose(X_train)
    X_dev = np.transpose(X_dev)

    print(X_train.shape)
    print(X_dev.shape)

    num_classes = len(train_onehot_dict)
    weighting = "uniform"
    knn = simpleKNN(num_classes, weighting)

    print("Fitting Simple KNN Model")
    knn.fit(X_train, y_train)

    print("Predicting on the dev set")
    y_pred = knn.predict(X_dev)
    accuracy_ = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred, average='weighted')
    recall = recall_score(y_dev, y_pred, average='weighted')
    f1_micro = f1_score(y_dev, y_pred, average='micro')
    f1_macro = f1_score(y_dev, y_pred, average='macro')
    f1_weighted = f1_score(y_dev, y_pred, average='weighted')
    confusion_mx = confusion_matrix(y_dev,y_pred)

    print("Accuracy: {} ".format(accuracy_))
    print('Precision: {}'.format(precision))
    print('Recall:    {}'.format(recall))
    print('F1 (micro):     {}'.format(f1_micro))
    print('F1 (micro):     {}'.format(f1_macro))
    print('F1 (weighted):  {}'.format(f1_weighted))
    print('Confusion matrix: \n {} \n'.format(confusion_mx))

if __name__ == '__main__':
    main()
