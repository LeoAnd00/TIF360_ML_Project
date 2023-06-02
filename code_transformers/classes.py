

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






## CLASSES AND LAYERS #################################

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


class TransformerLayer(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels,num_heads, dropout_rate):
        super().__init__()
        self.Attention = torch.nn.MultiheadAttention(embedding_dim,num_heads=num_heads,dropout=dropout_rate)
        self.Norm1 = torch.nn.LayerNorm(embedding_dim)
        self.Dense1 = torch.nn.Linear(embedding_dim,hidden_channels)
        self.relu = torch.nn.ReLU()
        self.Dense2 = torch.nn.Linear(hidden_channels,embedding_dim)
        

        self.Norm2 = torch.nn.LayerNorm(embedding_dim)
        

    def forward(self, x):
        addNormX = x
        x, _ = self.Attention(x,x,x)
        x = self.Norm1(x + addNormX)
        addNormX = x
        x = self.Dense1(x)
        x = self.relu(x)
        x = self.Dense2(x)
        x = self.Norm2(x + addNormX)

 
        return x
    


class TransformerNetwork(torch.nn.Module):
    def __init__(self,hidden_channels,output_dim, vocab_size, embedding_dim, num_heads, maxTokenLength,
                  nDescriptors,nFingerprints, useDescriptors, useFingerprints):
        super().__init__()
        # Embedd and add pos encoding to input
        self.dropout_rate = 0.15
        self.EmbeddingLayer = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim = embedding_dim , max_norm=True)
        self.PositionalEncoding = PositionalEncoding(embedding_dim, maxTokenLength, dropout = self.dropout_rate)

        self.TransEnc1 = TransformerLayer(embedding_dim,hidden_channels, num_heads, self.dropout_rate)
        self.TransEnc2 = TransformerLayer(embedding_dim,hidden_channels, num_heads, self.dropout_rate)
        self.TransEnc3 = TransformerLayer(embedding_dim,hidden_channels, num_heads, self.dropout_rate)
        self.Pooling = torch.nn.AvgPool1d(kernel_size = maxTokenLength)

        self.DenseOut1 = torch.nn.Linear(embedding_dim+nDescriptors+nFingerprints,hidden_channels)
        self.DenseOut2 = torch.nn.Linear(hidden_channels,output_dim)
        self.relu = torch.nn.ReLU()

        self.useDescriptors = useDescriptors
        self.useFingerprints = useFingerprints


    def forward(self,x,descriptors,fingerprints):
        x = self.EmbeddingLayer(x)
        x = self.PositionalEncoding(x)
        x = self.TransEnc1(x)
        x = self.TransEnc2(x)
        x = self.TransEnc3(x)
        # Permute so 22 dimension is compressed
        x = self.Pooling(x.permute((0,2,1))).permute((0,2,1))
        x = torch.squeeze(x,axis=1)
        if self.useDescriptors and self.useFingerprints:
            x = torch.cat((x,descriptors,fingerprints),1)
        elif self.useDescriptors:
            x = torch.cat((x,descriptors),1)
        elif self.useFingerprints:
            x = torch.cat((x,fingerprints),1)
        
        #x = x[:,-1,:]


        x = self.DenseOut1(x)
        x = self.relu(x)
        x = self.DenseOut2(x)
        
        return x