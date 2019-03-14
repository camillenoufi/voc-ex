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

        self.first = nn.Conv2d(1, embed_size, 5, stride=1, padding=1).double()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2,2), (2,2)).double()

        # Next is a ResNext Block
        self.resnext = ResNext(embed_size, 16, 1, dropout_rate)

        # The output from ResNext is then sent through RNN
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).double()
        #self.encoder = nn.LSTM(in_channels, input_size=(20,39), hidden_size=hidden_size, bias=True, bidirectional=True).double()
        #self.decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size, bias=True).double()
        self.decoder = nn.Linear(hidden_size*2, num_classes).double()  # 2 for bidirection
        self.dropout = nn.Dropout(dropout_rate).double()

        self.device = device


    def forward(self, input):

        x_conv = self.prepare(input)

        # Apply resnext
        x_conv = self.resnext(x_conv)
        #print("resnext output", x_conv.size())
        x_conv = x_conv.squeeze(1)
        #print("resnext output", x_conv.size())


        
        # dec_hiddens = self.dropout(dec_hiddens)
        # output = F.sigmoid(dec_hiddens)
        # output = self.target_projection(output)


        #input = input.reshape(-1, 172, 96).to('cpu')
        #print("encode input", input.size())

        input = x_conv

        h0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size).to(self.device).double() # 2 for bidirection  
        c0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size).to(self.device).double()
        
        # Forward propagate LSTM
        out, _ = self.encoder(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        #print("encode out", out.size())
        # Decode the hidden state of the last time step
        out = self.decoder(out[:, -1, :])
        #print("decode", out.size())

        out = self.dropout(out)
        out = torch.sigmoid(out)
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




