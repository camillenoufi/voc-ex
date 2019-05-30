import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F

from resnext import ResNext



class CRNN (nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate=0.3):

        super(CRNN, self).__init__()
        self.in_channels = input_size;
        self.n_classes = num_classes
        # First plain Vanilla CNN
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.first = nn.Conv2d(1, embed_size, 10, stride=1, padding=1).double()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2,2), (2,2)).double()

        # Next is a ResNext Block
        self.resnext = ResNext(embed_size, 16, 1, dropout_rate)

        # The output from ResNext is then sent through RNN
        self.encoder = nn.LSTM(51, hidden_size, num_layers, batch_first=True, bidirectional=True).double()
        #self.encoder = nn.LSTM(in_channels, input_size=(20,39), hidden_size=hidden_size, bias=True, bidirectional=True).double()
        #self.decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size, bias=True).double()
        self.decoder = nn.Linear(hidden_size*2, num_classes).double()  # 2 for bidirection
        self.dropout = nn.Dropout(dropout_rate).double()

        self.device = device


    def forward(self, input):

        batch_size, in_channels, height, width  = input.shape
        x_conv = self.prepare(input)

        # Apply resnext
        x_conv = self.resnext(x_conv)
        x_conv = x_conv.squeeze(1)

        input = x_conv

        h0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size).to(self.device).double() # 2 for bidirection  
        c0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size).to(self.device).double()
        
        # Forward propagate LSTM
        out, _ = self.encoder(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.decoder(out[:, -1, :])

        out = self.dropout(out)
        out = torch.sigmoid(out)

        return out


    def prepare(self, input):
        x_conv = self.first(input)
        x_conv = self.relu(x_conv)
        x_conv = self.maxpool(x_conv)
        return x_conv




class CRNNNoLSTM(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes, device, dropout_rate=0.3):

        super(CRNNNoLSTM, self).__init__()
        self.in_channels = input_size;
        self.n_classes = num_classes
        # First plain Vanilla CNN
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.first = nn.Conv2d(1, embed_size, 10, stride=1, padding=1).double()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2,2), (2,2)).double()

        # Next is a ResNext Block
        self.resnext = ResNext(embed_size, 16, num_classes, dropout_rate)

        # The output from ResNext is then sent through RNN
        #self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).double()
        #self.encoder = nn.LSTM(in_channels, input_size=(20,39), hidden_size=hidden_size, bias=True, bidirectional=True).double()
        #self.decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size, bias=True).double()
        #self.decoder = nn.Linear(hidden_size*2, num_classes).double()  # 2 for bidirection
        self.dropout = nn.Dropout(dropout_rate).double()

        self.fc1_input_size =  750  #10*19*50
        #fc1_input_size is dependent on kernel size and num filters, if those change, so will this number
        self.fc1_out_size = 594

        self.fc1 = nn.Linear(self.fc1_input_size, self.fc1_out_size, bias = True).double()
        self.fc2 = nn.Linear(self.fc1_out_size , num_classes, bias = True).double()  # fully connected to 10 output channels for 10 classes


        self.device = device

    def forward(self, input):

        batch_size, in_channels, height, width  = input.shape
        x_conv = self.prepare(input)

        # Apply resnext
        x_conv = self.resnext(x_conv)
        #print("resnext output", x_conv.size())
        x_conv = x_conv.squeeze(1)
        #print("resnext output", x_conv.size())

        
        # reshape (flatten) to be batch size * all other dims
        out = x_conv.contiguous().view(batch_size, -1)

        out = self.dropout(out)
        #out = torch.sigmoid(out)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        #print("The fc2", out.size())

        return out

        

    def prepare(self, input):
        #print(input.size())
        x_conv = self.first(input)
        #print("first layer", x_conv.size())
        x_conv = self.relu(x_conv)
        #print("first layer relu", x_conv.size())
        x_conv = self.maxpool(x_conv)
        #print("first layer maxpool", x_conv.size())

        return x_conv



