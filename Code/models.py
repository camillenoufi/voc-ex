import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F
from sklearn import neighbors, datasets

class BaselineModel:
    '''
    Abstract class to show what functions
    each model we implement needs to support 
    '''

    def __init__(self):
        '''
        Sets flags for the model to aid in debugging 
        '''

    def fit(self, *args):
        '''
        Trains model parameters and saves them as attributes of this class. 
        Variable numbers of parameters; depends on the class
        '''
        raise NotImplementedError


    def predict(self, *args):
        ''' 
        Uses trained model parameters to predict values for unseen data.
        Variable numbers of parameters; depends on the class.
        Raises ValueError if the model has not yet been trained.
        '''
        if not self.trained:
             raise ValueError("This model has not been trained yet")
        raise NotImplementedError


class VanillaCNN(nn.Module):

    def __init__(self, kernel_size, in_channels, num_filters, num_classes, dropout_rate = 0.3):
        
        super(VanillaCNN, self).__init__()

        self.kernel_size = kernel_size # 3
        self.in_channels = in_channels
        self.out_channels_1 = 32
        self.out_channels_2 = 64 
        self.out_channels_3 =  112 #random for now
        self.num_classes = num_classes
        self.mp_kernel_size = 2
        self.dropout_rate = dropout_rate
        self.fc1_input_size = 100
        self.fc1_out_size = 50
        
        # CNN / Max Pool 1
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_1, self.kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)  # 2x2 MP with padding

        # CNN / Max Pool 2  
        self.conv2 = nn.Conv2d(self.out_channels_1, self.out_channels_2, self.kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool2 =  nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)
 
        # CNN / Max Pool 3 
        self.conv3 = nn.Conv2d(self.out_channels_2, self.out_channels_3, self.kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool3 = nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.fc1_input_size, self.fc1_out_size, bias = True)
        self.fc2 = nn.Linear(self.fc1_out_size , self.num_classes, bias = True)  # fully connected to 10 output channels for 10 classes
                                      
    def forward(self, x_input):
        ''' Forward maps from x_input to x_conv_out 
                                                                                                                                  
        input is x_reshaped of shape (batch  * ? * ?) 
                                                     
        returns: x_out of shape (batch * ?)                                                                                                                                     
        '''
        batch_size, _  = x_input.shape 

        x_conv1 = self.conv1(x_input)     
        x_maxpool1 = self.pool1(F.relu(x_conv1))
        
        x_conv2 = self.conv2(x_maxpool1)
        x_maxpool2 = self.pool2(F.relu(x_conv2))

        x_conv3 = self.conv2(x_maxpool2)
        x_maxpool3 = self.pool2(F.relu(x_conv3))
        
        # reshape (flatten) to be batch size * all other dims
        #x_out = x_maxpool3.view(-1, dim1 * dim2 * dim3)  
        x_out = x_maxpool3.view(batch_size, -1)

        x_out = F.relu(self.fc1(x_maxpool3))  
        x_out = self.fc2(x_out)

        return x_out




def train_model(model, train_data, batch_size, learning_rate, num_epochs):
    ''' 
    Trains a given model
    '''

    model.train()  # set in train mode
    print(train_data)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
    
    for epoch in range(0,num_epochs):
        running_loss = 0.0
        
        for i, batch in enumerate(train_data):

            print(batch)
            inputs, labels = batch
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # compute the forward pass
            outputs = model(inputs)
            
            # compute the loss and optimizee
            loss_ = loss_fn(outputs, labels)
            loss_.backward()
            optimizer.step()
            
            running_loss += loss_





class simpleKNN(BaselineModel):
    
    def __init__(self, num_classes, weighting = "uniform"):

        super(simpleKNN, self).__init__()
        self.num_classes = num_classes
        self.weighting = weighting
        # also pass in country label names to create dict pairing label names to numbers 0-9? 
        
    def fit(self, X_train, y_train):
        ''' 
        Fits a simple knn Model given training matrix X (num_samples * num_features) and 
        class labels y (which is a value in range (0, num_classes-1) )
        '''

        clf = neighbors.KNeighborsClassifier(self.num_classes, self.weighting)
        clf.fit(X_train, y_train)

        self.clf = clf
        self.trained = true


    def predict(self, input):

        ''' Predicts class label for given batch of input ((n_query, n_features)
        for simple KNN model.
 
        Returns predicitions which is a value in range (0, num_classes-1)  
        which is of size (n_samples, n_output) or n_samples ?
        '''

        predictions = self.clf.predict(data) 
        
        return predictions






def outputSize(in_size, kernel_size, stride, padding):
    
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    
    return(output)


