#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        self.char_embed_size = 50
        self.char_embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, pad_token_idx)
        self.cnn = CNN(self.char_embed_size, self.embed_size)
        self.highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x_char_emb = self.char_embeddings(input)

        #print("input.size = " + str(input.size()))
        #print("x_char_emb.size = " + str(x_char_emb.size()))

        x_reshaped = x_char_emb.reshape(-1, x_char_emb.size()[3],x_char_emb.size()[2])
        #print("x_reshaped.size = " + str(x_reshaped.size()))
        x_conv_out = self.cnn(x_reshaped)
        #print("x_conv_out.size = " + str(x_conv_out.size()))

        x_conv_out_reshaped = x_conv_out.reshape(input.size()[0], input.size()[1], self.embed_size)
        #print("x_conv_out_reshaped.size = " + str(x_conv_out_reshaped.size()))

        x_highway = self.highway(x_conv_out_reshaped)
        output = self.dropout(x_highway)
        #print("output.size = " + str(output.size()))

        return output
        ### END YOUR CODE

