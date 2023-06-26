# Design a cross graph local info matching model.

# Input: graphA & graphB  (H and edge_index)
# Output: match or mismatch (0 or 1)

from turtle import forward
import torch
from torch import nn
import math
from torch.nn import *
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# from layers.cross_graph_attention_layer import crossGraphAttentionLayer
from layers.bi_lstm_layer import LSTMModel

class crossGraphLocalMatch(nn.Module):
    def __init__(self, vocabSize, hidden, in_features, out_features, class_num, dropout, alpha):
        super(crossGraphLocalMatch, self).__init__()

        # word embedding layer
        self.embed = nn.Embedding(vocabSize,hidden)

        # two gcn layers
        self.GCN1 = GCNConv(hidden, hidden)
        self.GCN2 = GCNConv(hidden, hidden)

        # three group of readout op
        # self.readOut1 = crossGraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
        # self.readOut2 = crossGraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
        # self.readOut3 = crossGraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)

        # three group of BiLSTM
        self.bi_lstm1 = LSTMModel(hidden, 100, 2, hidden)
        self.bi_lstm2 = LSTMModel(hidden, 100, 2, hidden)
        self.bi_lstm3 = LSTMModel(hidden, 100, 2, hidden)

        # one fusion BiLSTM layer
        self.fusionBiLSTM = LSTMModel(2*hidden, 100, 2, hidden)

        # full connect layer
        self.fc = Linear(hidden, 2)

    def forward(self, h1_index, edge_index1, h2_index, edge_index2):

        #print("h1_index",h1_index.shape,h1_index)
        #print("h2_index",h2_index.shape,h2_index)
        h1 = self.embed(h1_index)
        h2 = self.embed(h2_index)
        #print("h1",h1.shape)
        #print("h2",h2.shape)

        # readOut weighted h1, h2 at text level
        text_h1, text_h2 = h1, h2
        #print("text_h1",text_h1.shape)
        #print("text_h2",text_h2.shape)

        # first gcn layer
        h1 = F.relu(self.GCN1(h1, edge_index1))
        h2 = F.relu(self.GCN1(h2, edge_index2))
        # readOut weighted h1, h2 at type level
        type_h1, type_h2 = h1, h2

        # second gcn layer
        h1 = F.relu(self.GCN2(h1, edge_index1))
        h2 = F.relu(self.GCN2(h2, edge_index2))
        # readOut weighted h1, h2 at call level
        call_h1, call_h2 = h1, h2

        textInfo = torch.cat([self.bi_lstm1(text_h1.unsqueeze(1)), self.bi_lstm1(text_h2.unsqueeze(1))], dim=-1)
        typeInfo = torch.cat([self.bi_lstm2(type_h1.unsqueeze(1)), self.bi_lstm2(type_h2.unsqueeze(1))], dim=-1)
        callInfo = torch.cat([self.bi_lstm3(call_h1.unsqueeze(1)), self.bi_lstm3(call_h2.unsqueeze(1))], dim=-1)
        #print("textInfo",textInfo.shape)
        #print("typeInfo",typeInfo.shape)
        #print("callInfo",callInfo.shape)
        # fusion three info
        #print('torch.stack([textInfo,typeInfo,callInfo], dim=0)',torch.stack([textInfo,typeInfo,callInfo], dim=0).shape)
        fusionOut = self.fusionBiLSTM(torch.stack([textInfo,typeInfo,callInfo], dim=0))
        #print("fusionOut",fusionOut.shape)
        out = self.fc(fusionOut.reshape(1,-1))
        #print("out",out.shape)
        out = F.softmax(out, dim=-1)

        return out