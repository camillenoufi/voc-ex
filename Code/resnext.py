import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import torch.nn.functional as F


class ResNext(nn.Module):

	def __init__(self, in_channels, hidden_size, n_classes, dropout_rate=0.3):


		super(ResNext, self).__init__()
		self.layer1 = nn.Conv2d(in_channels, hidden_size, (1,1), stride=1, padding=1).double()
		self.layer2 = nn.Conv2d(hidden_size, hidden_size, (5,5), stride=1, groups=16, padding=1).double()
		self.layer3 = nn.Conv2d(hidden_size,in_channels, (1,1), stride=1, padding=1).double()

		# Linear Layers for feed forward 
		self.projection = nn.Linear(in_channels, in_channels, bias=True)
		self.gate = nn.Linear(in_channels, in_channels, bias=True)

		self.maxpool = nn.MaxPool2d((2,2), (2,2)).double()

		self.output_layer = nn.Linear(in_channels, n_classes, bias = True).double()
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, input):

		#print(input.size())
		# Apply the layers sequentially 
		x_conv = self.layer1(input)
		#print("resNext layer 1", x_conv.size())
		x_conv = self.layer2(x_conv)
		#print("resNext layer 2", x_conv.size())
		x_conv = self.layer3(x_conv)
		#print("resNext layer 3", x_conv.size())

		x_proj =  self.maxpool(F.relu(x_conv))
		#print("resNext maxpool", x_proj.size())
		# Need to permute before applying output layer
		x_proj = x_proj.permute(0,2,3,1)
		#print("resNext maxpool", x_proj.size())
		x_out = self.output_layer(x_proj)
		#print("resNext output", x_out.size())

		# permute back to get original shape
		x_out = x_out.permute(0,3,1,2)
		x_out = self.dropout(x_out)
		#print("resNext output", x_out.size())

		return x_out



