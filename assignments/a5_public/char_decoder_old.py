#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.target_vocab = target_vocab

        V_char = 0
        if target_vocab is not None:
            V_char = len(target_vocab.char2id)

        self.char_output_projection = nn.Linear(hidden_size, V_char, bias=True)
        end_token_word_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(V_char, char_embedding_size, end_token_word_idx)

        self.softmax = nn.Softmax(dim=2)
        self.padding_idx = self.target_vocab.char2id['<pad>']
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx,
                                   reduction='sum')
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_input = self.decoderCharEmb(input.long()).float()
        output, (h_t, c_t) = self.charDecoder(char_input, (dec_hidden[0].float(), dec_hidden[1].float()))
        s_t = self.char_output_projection(output)
        return s_t, (h_t, c_t)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        (score, dec_hidden) = self.forward(char_sequence[:-1], dec_hidden)
        target = char_sequence[1:].contiguous().view(-1)  # not get first character
        score = score.view(-1, score.shape[-1])
        loss = self.loss(score, target)
        return loss

        ### END YOUR CODE

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
        dec_hidden = initialStates # tuple of two tensors of size (1, batch, hidden_size)
        _, batch_size, hidden_size = initialStates[0].shape
        current_chars = [self.target_vocab.start_of_word for _ in range(batch_size)] # size batch_size, 1
        current_char_tensor = torch.Tensor(current_chars, device=device)
        current_char_input = current_char_tensor.view(1,-1) # size (1, batch_size)

        output_batched_words = None
        for t in range(max_length):
            scores, dec_hidden = self.forward(current_char_input, dec_hidden) #tuple of two tensors of shape (1, batch, vocab_size), (1, batch, hidden_size)
            p_hat = self.softmax(scores) # shape (1, batch, vocab_size)
            current_char_input = p_hat.argmax(dim=2) # shape (1, batch)
            if output_batched_words is None:
                output_batched_words = current_char_input
            else:
                output_batched_words = torch.cat((output_batched_words, current_char_input)) # will have shape (max_length, batch)

        output_batched_words = output_batched_words.t() # shape (batch, max_length)

        decodedWords = []

        for batched_words in output_batched_words:
            word = ''
            for batched_char_tensor in batched_words:
                char_item = batched_char_tensor.item()
                if char_item != self.target_vocab.end_of_word:
                    word += self.target_vocab.id2char[char_item]
                else:
                    break
            decodedWords.append(word)
        return decodedWords

        ### END YOUR CODE

