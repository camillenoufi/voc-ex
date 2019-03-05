#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import numpy as np
import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.
        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character char_embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """

        
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character char_embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse char_embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()
        char_vocab_size = len(target_vocab.char2id)
        self.n_chars = char_vocab_size

        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size,
                                   bidirectional=False,
                                   bias=True)
        self.char_output_projection = nn.Linear(hidden_size, char_vocab_size, bias=True)
        self.decoderCharEmb = nn.Embedding(char_vocab_size, char_embedding_size,
                                           padding_idx=0)
        self.target_vocab = target_vocab

        # Added for 2c.
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum',ignore_index=0)

        # Added for 2d
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.
        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        embedded = self.decoderCharEmb(input)
        h_t, c_t = self.charDecoder(embedded, dec_hidden)
        s_t = self.char_output_projection(h_t)
        return s_t, c_t

    

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.
        @param char_sequence: tensor of integers, shape (length, batch).
            Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder.
            A tuple of two tensors of shape (1, batch, hidden_size)
        @returns The cross-entropy loss, computed as the *sum* of cross-entropy
        losses of all the words in the batch.
        """
        # what if different than length expected by forward?

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###    - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout
        #  (e.g., <START>,m,u,s,i,c,<END>).

        target = char_sequence[1:]
        x_input = char_sequence[:-1]
        s_t, c_t = self.forward(x_input, dec_hidden)  
        s_t = s_t.contiguous().view(-1, self.n_chars)
        target = target.contiguous().view(-1)
        return self.cross_entropy_loss(s_t, target)

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode
        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        output_words = []
        batch_size = initialStates[0].shape[1]
        start_seed = np.array([self.target_vocab.start_of_word] * batch_size).reshape(1, batch_size)
        current_char = torch.tensor(start_seed, dtype=torch.long).to(device)
        output_word = []
        c = initialStates

        for t in range(max_length):
            s_t, c = self.forward(current_char, c)
            p_t = self.softmax(s_t)

            current_char = torch.argmax(p_t, dim=-1)
            loop_over = current_char.squeeze(dim=0).cpu().numpy()
            word =[]
            for x in loop_over:
                word.append(self.target_vocab.id2char[x])
            output_words.append(word)

        words =[]
        for w in np.array(output_words).T:
            curr_word = []
            for char in w:
                if self.target_vocab.char2id[char] == self.target_vocab.end_of_word:
                    break
                curr_word.append(char)
            words.append(curr_word) 
        
        return [''.join(w) for w in words]

        ### END YOUR CODE