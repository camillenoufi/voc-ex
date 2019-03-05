#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

class ResNext(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, output_size, window_size = 5, max_word_len=21):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """

        # The input is a 80 X 21 tensor
        # Should we move it to a 1D convolution? Most likely or may be not.

        super(ResNext, self).__init__()
        
        

    def forward(self, input):
        
        """
            Steps:
            1. Apply the Siamese arch
            2. Apply ResNext 
        """
        #(x0, x1) = self.cnn(input)
