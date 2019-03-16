import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F
from sklearn import neighbors, datasets
from sklearn.metrics import f1_score, precision_score, recall_score

from earlystop import EarlyStopping


class VanillaCNN(nn.Module):

    def __init__(self, kernel_size, in_channels, num_filters, num_classes, dropout_rate = 0.3):
        '''
        Input size is (batch_size, in_channels, height of input planes, width)
        '''
        super(VanillaCNN, self).__init__()

        self.kernel_size = kernel_size # 3
        self.in_channels = in_channels # 1
        self.out_channels_1 = num_filters # random
        self.out_channels_2 = num_filters
        self.out_channels_3 = num_filters  # 18/24
        self.num_classes = num_classes # 10
        self.mp_kernel_size = 2
        self.dropout_rate = dropout_rate
        self.fc1_input_size =  19008  #9504
        #fc1_input_size is dependent on kernel size and num filters, if those change, so will this number
        self.fc1_out_size = 594

        # CNN / Max Pool 1
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_1, self.kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)  # 2x2 MP with padding

        # CNN / Max Pool 2
        self.conv2 = nn.Conv2d(self.out_channels_1, self.out_channels_2, self.kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool2 =  nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)

        # CNN / Max Pool 3
        self.conv3 = nn.Conv2d(self.out_channels_2, self.out_channels_3, self.kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool3 = nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(self.fc1_input_size, self.fc1_out_size, bias = True)
        self.fc2 = nn.Linear(self.fc1_out_size , self.num_classes, bias = True)  # fully connected to 10 output channels for 10 classes

    def forward(self, x_input):
        ''' Forward maps from x_input to x_conv_out

        input x_input is of shape (batch_size, in_channels, height of input planes, width )

        returns: x_out of shape (batch * ?)
        '''

        batch_size, in_channels, height, width  = x_input.shape

        x_conv1 = self.conv1(x_input.float())
        x_maxpool1 = self.pool1(F.relu(x_conv1))

        x_conv2 = self.conv2(x_maxpool1)
        x_maxpool2 = self.pool2(F.relu(x_conv2))

        x_conv3 = self.conv2(x_maxpool2)
        x_maxpool3 = self.pool2(F.relu(x_conv3))

        # reshape (flatten) to be batch size * all other dims
        x_out = x_maxpool3.view(batch_size, -1)
        x_out = self.drop_out(x_out)

        x_out = F.relu(self.fc1(x_out))
        x_out = self.fc2(x_out)

        return x_out




def train_model(model, train_data_loader, valid_loader, batch_size, learning_rate, num_epochs, device):
    '''
    Trains a given model
    '''

    #model = model.to(device).train()  # set in train mode
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    total_steps = len(train_data_loader)
    train_losses = []
    acc_list = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    

    print("Starting Training")

    for epoch in range(0,num_epochs):
        
        ###################
        # train the model #
        ###################

        model = model.to(device).train()  # set in train mode
        running_loss = 0.0
        print("Starting Training ", epoch)
        for i, batch in enumerate(train_data_loader):

            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            #Set the parameter gradients to zero
            optimizer.zero_grad()

            # compute the forward pass
            outputs = model(inputs) #to device

            # compute the loss and optimizee
            loss_ = loss_fn(outputs, labels)
            train_losses.append(loss_.item())
            loss_.backward()
            optimizer.step()

            running_loss += loss_

            # track accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)


            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_steps, loss_.item(),
                              (correct / total) * 100))


        ######################    
        # validate the model #
        ######################

        model = model.to(device).eval()
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = loss_fn(output, target)
            # record validation loss
            valid_losses.append(loss.item())



        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print("Valid loss:", valid_loss)
        print("checking for earlystop criteria")
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    return  model


def eval_model(model, dev_data_loader, device):

    model = model.to(device).eval()

    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    ##
    f1_micro = 0
    f1_macro = 0
    f1_weighted = 0
    precision = 0
    recall = 0
    num_batches = 0
                                
    with torch.no_grad():
        running_eval_loss = 0
        for inputs, labels in dev_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss_ = loss_fn(outputs, labels)
            running_eval_loss += loss_
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            labels = labels.cpu() 
            predicted = predicted.cpu()
            
            f1_micro += f1_score(labels, predicted, average='micro')
            f1_macro += f1_score(labels, predicted, average='macro')
            f1_weighted += f1_score(labels, predicted, average='weighted')
            precision += precision_score(labels, predicted, average='weighted')
            recall += recall_score(labels, predicted, average='weighted')
            num_batches += 1

        print('Test Accuracy of the model on the dev inputs: {} %'.format((correct / total) * 100))
        print('Average f1, precision, and recall metrics over {} batches:'.format(num_batches))
        print('F1 (micro):     {}'.format(f1_micro/num_batches))
        print('F1 (micro):     {}'.format(f1_macro/num_batches))
        print('F1 (weighted):  {}'.format(f1_weighted/num_batches))
        print('Precision: {}'.format(precision/num_batches))
        print('Recall:    {}'.format(recall/num_batches))
