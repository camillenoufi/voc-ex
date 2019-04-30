import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F
from sklearn import neighbors, datasets
import

class VariableWideLayer(nn.Module):

    def __init__(self, in_channels):
        '''
        Input size is (batch_size, in_channels, height of input planes, width)
        '''
        super(VariableFirstLayer, self).__init__()

        self.in_channels = in_channels # 1
        self.out_channels_1 = 18 # random
        self.mp_kernel_size = 2

        # CNN / Max Pool 1
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_1, self.kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(self.mp_kernel_size, stride=2, padding=1)  # 2x2 MP with padding


class VariableKernelSizeCNN(nn.Module):

    def __init__(self, in_channels, num_filters, num_classes, dropout_rate = 0.3):
        '''
        Input size is (batch_size, in_channels, height of input planes, width)
        '''
        super(VanillaCNN, self).__init__()

        self.in_channels = in_channels # 1
        self.out_channels_1 = 18 # random
        self.out_channels_2 = 18
        self.out_channels_3 = 18  # 18/24
        self.num_classes = num_classes # 10
        self.mp_kernel_size = 2
        self.dropout_rate = dropout_rate
        self.fc1_input_size = 5382 #3588 #4784 #4536 #5382
        self.fc1_out_size = 80

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
