#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1i
class CNN(nn.Module):

    def __init__(self, char_embedding_size: int = 50, word_embedding_size: int=50, word_size: int=21):
        super(CNN, self).__init__()
        self.k = 5
        self.char_embedding_size = char_embedding_size
        self.f = word_embedding_size
        self.word_size = word_size
        self.h_conv = nn.Conv1d(self.char_embedding_size,
                                self.f,
                                self.k)
        self.max_pool = nn.MaxPool1d(word_size - self.k + 1)

    def forward(self, x_reshaped):
        """ Implement forward propagation for CNN
        Take mini-batch of character embeddings and convolve them, to encode words into more compact representation,
        to then be max-pooled and compressed further.

        @param x_reshaped (Tensor): (   max_sentence_length_char,
                                        character_embedding_size,
                                        max_word_length )
        @returns x_conv_out (Tensor): (max_sentence_length_char,
                                        word_embed_size)
        """
        #print ("x_reshaped.shape={}".format(str(x_reshaped.shape)))
        x_conv = self.h_conv(x_reshaped)
        #print ("x_conv.shape={}".format(str(x_conv.shape)))
        x_conv_relu = nn.functional.relu(x_conv)
        #print ("x_conv_relu.shape={}".format(str(x_conv_relu.shape)))
        x_conv_out = self.max_pool(x_conv_relu).squeeze(dim=2)
        #print ("x_conv_out.shape={}".format(str(x_conv_out.shape)))
        return x_conv_out

### END YOUR CODE

