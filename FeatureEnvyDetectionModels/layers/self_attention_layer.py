
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class SelfAttentionLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, alpha=0.2, dropout=0.1, concat=True):
        super(SelfAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)).to('cuda'))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_dim, 1)).to('cuda'))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        

    def forward(self, input): # features是一个二维列表
        
        w_input = torch.mm(input, self.W)
        input_dot = self._prepare_attentional_mechanism_input(w_input)
        e = torch.matmul(input_dot, self.a).squeeze(2)
        attention = F.softmax(torch.sum(e, dim=1), dim=-1).reshape(-1,1)  # 每行e加和为1，（1*N）
        #print("attention",attention)
        #output = self.leakyrelu(torch.matmul(attention, w_input).view(1,-1)) #（1*N · N*f = 1*f）
        #print("w_input",w_input)
        output = w_input * attention
        #print("selfAttOut", output.shape)
        return output  # 返回所有结点的单一向量组成的二维张量特征数据
        
    def _prepare_attentional_mechanism_input(self, w_input): #[2708, 8]
        N = w_input.size()[0] # number of node features

        Wh_repeated_in_chunks = w_input.repeat_interleave(N, dim=0)   #torch.Size([7333264, 8])
        
        Wh_repeated_alternating = w_input.repeat(N, 1) #torch.Size([7333264, 8])

        all_combinations_matrix = Wh_repeated_in_chunks * Wh_repeated_alternating
        # all_combinations_matrix.shape == (N * N, out_dim)
        #print("all_combinations_matrix",all_combinations_matrix.shape)
        dotMatrix = all_combinations_matrix.view(N, N, self.out_dim)
        #print("dotMatrix", dotMatrix.shape)
        return dotMatrix


