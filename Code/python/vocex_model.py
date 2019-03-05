from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ResNext
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

INPUT_SIZE = 172
INPUT_LENGTH = 96
OUT_SIZE = 172

class Vocex(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, dropout_rate=0.1):
        super(Vocex, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Random Embedding sent to siamese? Simply for dimensionality reduction.
        # TODO(vidya): How big should this embedding be? 
        # In the case of the 
        self.embedding = ResNext(embed_size=INPUT_LENGTH, output_size = OUT_SIZE)

        # initialize the encoder
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bias=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embed_size+hidden_size, hidden_size, bias=True)
        self.att_projection = nn.Linear(2*hidden_size, hidden_size, bias=False) #Attention Layer?
        self.target_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(input):

    	# Step 1: Get the embedding from ResNext
    	# Step 2: Encode the embedding
    	# Step 3: decode the output from Step 2
    	# Step 4: Compute the score. 

    	return 0.0



        
        






