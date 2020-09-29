#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size):
        """ Init Highway.
        @param (param_name, TBC): (param_description, tbc)
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size

        self.w_proj = nn.Linear(embed_size, embed_size, bias=True)
        self.w_gate = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, x_conv_out):
        """ Implement forward propagation for Highway network
        Given final output of CNN, x_conv_out, calculate and return x_highway.

        @param x_conv_out(Tensor): (max_sentence_length, batch_size, embed_size)
        @returns x_highway (Tensor): (max_sentence_length, batch_size, embed_size)
        """
        h_proj = self.w_proj(x_conv_out)
        x_proj =  nn.functional.relu(h_proj)

        h_gate = self.w_gate(x_conv_out)
        x_gate = torch.sigmoid(h_gate)

        x_highway = x_gate * x_proj + (1 - x_gate) * (x_conv_out)
        return x_highway

### END YOUR CODE

