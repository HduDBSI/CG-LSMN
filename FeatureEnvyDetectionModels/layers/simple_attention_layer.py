
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
class SimpleAttention(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout=0.1, concat=True):
        super(SimpleAttention, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)).to(device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_dim, 1)).to(device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def forward(self, input): 
        
        e = torch.tanh(torch.matmul(input, self.W))
        #print("e", e.shape)
        att = F.softmax(torch.matmul(e, self.a),dim=0)
        #print("att", att.shape)
        ouput = torch.matmul(att.reshape(1,-1), e).squeeze(1)
        #print("simgleAttOut", ouput.shape)
        return ouput  
