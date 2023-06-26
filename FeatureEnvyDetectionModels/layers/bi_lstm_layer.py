import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, bidirectional = True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    
    def forward(self, x):
        #print("x",x.shape)
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        
        # h0, c0 shape: (num_layers * num_directions, batch, hidden_size)
        if torch.cuda.is_available():
            #(num_layers * num_directions, batch, hidden_size)
            h0 = Variable(torch.zeros(self.layer_dim*2, 1, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim*2, 1, self.hidden_dim))
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim*2, 1, self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim*2, 1, self.hidden_dim))
        
        # One time step
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #print("out",out.shape)
        #output = out.reshape(out.size(0),1,self.layer_dim,self.hidden_dim)
        #forward_output = output[:, :, 0, :]  # 正向LSTM的输出，形状为(L, N, h)
        #backward_output = output[:, :, 1, :]  # 反向LSTM的输出，形状为(L, N, h)
        forward_h_n = hn[::2]  # 正向LSTM的h_n，形状为(n, N, h)
        backward_h_n = hn[1::2]  # 反向LSTM的h_n，形状为(n, N, h)

        # 因为是正向LSTM，所以时间方向是从左向右，因此forward_output[-1]代表
        # 最后一个时间步上的最后一层的输出
        #print(forward_output[-1] == forward_h_n[-1])

        # 因为是反向LSTM，所以时间方向是从右向左，因此backward_output[0]代表
        # 最后一个时间步上的最后一层的输出
        #print(backward_output[0] == backward_h_n[-1])

        #final_out = torch.cat([forward_output[-1],backward_output[0]],dim=-1)
        final_out = torch.cat([forward_h_n[-1],backward_h_n[-1]],dim=-1)
        #print('final_out',final_out.shape,final_out)
        #exit()
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        #out = self.fc(out[:, -1, :])
        final_out = self.fc(final_out)
        # out.size() --> 100, 10
        #print("BiLSTM out",out.shape)
        return final_out