# Design a cross graph attention mechanism layer. (readout data from gcn)

# Input: nodes in graphA & nodes in graphB
# Output: weighted nodes in A & weighted nodes in B

import torch
import torch.nn.functional as F
from torch import nn
from layers.multi_head_att_topk import MultiheadAttention

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
        
        e = torch.matmul(input, self.W)
        #print("e", e.shape)
        att = F.softmax(torch.matmul(e, self.a),dim=0)
        #print("att", att.shape)
        #print("e",e.shape)
        ouput = att * e
        #print("simgleAttOut", ouput.shape)
        #exit()
        return ouput 

class crossGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(crossGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        #self.att = SimpleAttention(in_features, out_features)
        #input_dim, output_dim, num_heads, top_k
        #self.multi_head_att_topk = MultiheadAttention(in_features, out_features, num_heads = 2, top_k = 1000)

        self.W = nn.Parameter(torch.empty(size=(in_features, 2*out_features)).to(device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps) + eps * (d <= eps)
        return n / d
    
    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.mm(v1, v2.permute(1, 0))
        
        v1_norm = v1.norm(p=2, dim=1, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=1, keepdim=True).permute(1, 0)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)


    def forward(self, h1, h2):
        
        # Add trainable W
        wh1 = torch.relu(torch.mm(h1, self.W))
        wh2 = torch.relu(torch.mm(h2, self.W))

        # Calculate cosine similarity between two nodes in two graphs
        cosineSimilarity = torch.relu(self.cosine_attention(wh1, wh2))
        #weightA = torch.max(cosineSimilarity, dim=1).values
        #weightB = torch.max(cosineSimilarity, dim=0).values
        #weightA = torch.sum(cosineSimilarity, dim=1)
        #weightB = torch.sum(cosineSimilarity, dim=0)
        weightA = torch.mean(cosineSimilarity, dim=1)
        weightB = torch.mean(cosineSimilarity, dim=0)
        '''
        weightA = torch.zeros(h1.size()[0]).to(device)
        weightB = torch.zeros(h2.size()[0]).to(device)
        for i,v in enumerate(h1):
            weightA[i] = torch.max(self.cosine_attention(v.reshape(1,-1),h2), dim=1).values[0]
        for i,v in enumerate(h2):
            weightB[i] = torch.max(self.cosine_attention(v.reshape(1,-1),h1), dim=1).values[0]
        '''
        #print('weightA',weightA)
        #print('weightB',weightB)
        #exit()
        # Calculate attention mechanisms based on cosine similarity
        
        Att1 = weightA.reshape(-1,1)
        Att2 = weightB.reshape(-1,1)
        #print('Att1',Att1)
        # #print('Att2',Att2)
        # readOut1 = wh1 * Att1
        # readOut2 = wh2 * Att2
        readOut1 = h1 * Att1
        readOut2 = h2 * Att2
        # readOut1 += h1 * Att1
        # readOut2 += h2 * Att2
        #print('readOut1',readOut1)
        #print('readOut2',readOut2)
        
        return readOut1, readOut2

if __name__ == '__main__':
    """
    a = torch.Tensor(3,5)
    print("a",a.shape,a)
    h=torch.Tensor([[0.1],[0.2],[0.3]])
    H=torch.Tensor([[5,4,3,2,1],[0,1,2,3,4],[7,8,9,10,11]])
    hh = h.repeat_interleave(a.size(1),1)
    HH = H.repeat(h.size(0),1)
    print("hh",hh.shape,hh)
    print(torch.mul(a,hh))
    '''
    hh tensor([[1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.]])
    '''
    #print("HH",HH)
    """
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps) + eps * (d <= eps)
        return n / d
    
    def cosine_attention(v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.mm(v1, v2.permute(1, 0))
        #print('a',a.shape,a)
        v1_norm = v1.norm(p=2, dim=1, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=1, keepdim=True).permute(1, 0)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        #print('d',d.shape,d)
        #exit()
        return div_with_small_value(a, d)

    model = crossGraphAttentionLayer(5,10,0.1,0.2).to(device)

    embedding = nn.Embedding(1000000, 128) # 假定字典中只有100个词，词向量维度为128
    word1 = [1, 2, 3, 4]
    word2 = [1, 2, 3, 4]      
    word3 = [1, 2, 30, 40]
    embed1 = embedding(torch.LongTensor(word1)).to(device)
    embed2 = embedding(torch.LongTensor(word2)).to(device)
    embed3 = embedding(torch.LongTensor(word3)).to(device)

    print('embed1',embed1)
    print('embed2',embed2.shape)


    h1,h2 = model(embed1,embed2)

    print("h1",h1)
    print("h2",h2.shape)
    
    print(embedding.weight.requires_grad)


    print('cosine_attention11', cosine_attention(embed1,embed1))
    print('cosine_attention12', cosine_attention(embed1,embed2))
    print('cosine_attention13', cosine_attention(embed1,embed3))
    print('cosine_attention12+13', cosine_attention(embed1,embed2) + cosine_attention(embed1,embed3))
    print('cosine_attention1(2+3)', cosine_attention(embed1,torch.sum(torch.stack([embed2,embed3],dim=0),dim=0)))


    word4 = [i for i in range(6)]
    word5 = [i for i in range(3,8)]
    embed4 = embedding(torch.LongTensor(word4)).to(device)
    embed5 = embedding(torch.LongTensor(word5)).to(device)
    print('embed4',embed4.shape)
    print('embed5',embed5.shape)

    cosine_attention45 = torch.relu(cosine_attention(embed4,embed5))

    print('cosine_attention45', cosine_attention45)

    weightA = torch.max(cosine_attention45, dim=0).values
    weightB = torch.max(cosine_attention45, dim=1).values

    print('weightA',weightA)
    print('weightB',weightB)

    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    b = torch.tensor([[0.1],[0.2],[0.3],[0.1]])
    print('a',a.shape)
    print('b',b.shape)
    c = a*b
    c[1][1] = 10
    print(c)

    weightAA = torch.zeros(embed4.size()[0])
    weightBB = torch.zeros(embed5.size()[0])
    for i,v in enumerate(embed4):
        print('cosine_attention(v,embed5)',cosine_attention(v.reshape(1,-1),embed5))
        weightAA[i] = torch.max(cosine_attention(v.reshape(1,-1),embed5), dim=1).values[0]
    for i,v in enumerate(embed5):
        print('cosine_attention(v,embed4)',cosine_attention(v.reshape(1,-1),embed4))
        weightBB[i] = torch.max(cosine_attention(v.reshape(1,-1),embed4), dim=1).values[0]
    a = torch.ones([5,1])
    print(a,torch.max(a,dim=0))
    print('weightAA',weightAA)
    print('weightBB',weightBB)

