import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size)
        
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        
        frequency_term = torch.exp(torch.arrange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position*frequency_term)
        pe[:, 1::2] = torch.cos(position*frequency_term)
        
        pe = pe.unsqueeze(0) # add batch dimention
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad(False)
        return self.dropout(x)
        

        
        
