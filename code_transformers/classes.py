

# Functions
import numpy as np
def create_encoded_vector(input, dictionary,maxTokenLength):
    # Input is list of strings on the form ['a','b','c'] where the string are tokens
    # Dictionary is the dictionary containign all possible tokens, and an index for them
    # MaxTokenLength is the max amount of tokens any input creates
    encoded_tensor = np.zeros(maxTokenLength,dtype=np.int32)
    # Change value in right place to one
    keyCount = 0
    for key in input:
        encoded_tensor[keyCount] = dictionary[key]
        keyCount+=1
    
    # encoded_tensor = np.expand_dims(encoded_tensor,axis=1)
    return encoded_tensor






## CLASSES AND LAYERS

import math
import torch
import torch.nn as nn

# From attention is all you need
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len,dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x = x + torch.tensor(self.pe[:, :x.size(1)], 
        #                  requires_grad=False)
        x = x + self.pe[:x.size(0), :].detach()
        return self.dropout(x)