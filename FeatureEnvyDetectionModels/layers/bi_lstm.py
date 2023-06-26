import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bidirectional = True, dropout = 0):
        super(BiLSTM, self).__init__()
        # Hidden dimensions
        self.dropout =dropout
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        
        # Readout layer
        self.bilstm = nn.LSTM(input_dim, hidden_dim, layer_dim, bidirectional = self.bidirectional, dropout = self.dropout)
    
    def forward(self, x):
        #print("x",x.shape)
        #(num_layers * num_directions, batch, hidden_size)
        #h0 = Variable(torch.zeros(self.layer_dim*2, 1, self.hidden_dim).to(device))
        #c0 = Variable(torch.zeros(self.layer_dim*2, 1, self.hidden_dim).to(device))
        
        # One time step
        #out, (hn, cn) = self.bilstm(x, (h0, c0))
        out, (hn, cn) = self.bilstm(x,)
        #print("out",out.shape)
        output = out[:, -1, :]
        #print("output",output.shape)
        return output

