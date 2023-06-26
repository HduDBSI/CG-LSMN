import torch
import torch.nn.functional as F

class MultiheadAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, top_k):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.top_k = top_k
        
        self.query_fc = torch.nn.Linear(input_dim, output_dim * self.num_heads)
        self.key_fc = torch.nn.Linear(input_dim, output_dim * self.num_heads)
        self.value_fc = torch.nn.Linear(input_dim, output_dim * self.num_heads)
        self.output_fc = torch.nn.Linear(output_dim, output_dim * self.num_heads)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        seq_len, input_dim = x.size()
        batch_size = 1
        assert input_dim == self.input_dim
        self.top_k = seq_len if seq_len < self.top_k else self.top_k
        
        # Linear projections
        Q = self.query_fc(x)  # Q: (batch_size, seq_len, output_dim)
        K = self.key_fc(x)    # K: (batch_size, seq_len, output_dim)
        V = self.value_fc(x)  # V: (batch_size, seq_len, output_dim)

        # Split heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.output_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.output_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.output_dim)
        # print('Q',Q.shape)
        # print('K',K.shape)
        # print('V',V.shape)
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.output_dim) ** 0.5
        #print('scores',scores.shape)
        scores = F.softmax(scores, dim=-1)  # scores: (batch_size, num_heads, seq_len, seq_len)
        #print('scores',scores.shape)
        # Apply attention weights to values
        attended = torch.matmul(scores, V)  # attended: (batch_size, num_heads, seq_len, output_dim // num_heads)
        #print('attended',attended.shape)
        # Concatenate heads and reshape
        attended = attended.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len * self.num_heads, self.output_dim)
        #print('attended',attended.shape)
        # Apply output projection
        output = self.output_fc(attended)  # output: (batch_size, seq_len, input_dim)

        # Select top K features from each head
        topk_outputs = []
        for i in range(self.num_heads):
            head_output = output[:, :, i * (self.output_dim):(i + 1) * (self.output_dim)]
            topk_indices = torch.topk(head_output, k=self.top_k, dim=1).indices
            topk_features = torch.gather(head_output, 1, topk_indices)
            #print('topk_features',topk_features.shape)
            topk_outputs.append(topk_features)
            
        topk_outputs = torch.stack(topk_outputs, dim=1).view(-1, self.output_dim)
        #print('topk_outputs',topk_outputs.shape)
        return topk_outputs




if __name__ == '__main__':
    
    batch_size = 1
    seq_len = 10000
    input_dim = 128
    output_dim = 256
    num_heads = 8
    top_k = 100

    x = torch.randn(batch_size, seq_len, input_dim)
    attention = MultiheadAttention(input_dim, output_dim, num_heads, top_k)
    print(x.size())
    output = attention(x)
    print(output.size())  # Output: (32, 10, 24) (assuming K = 3 and num_heads = 8)

    '''
    Input: torch.Size([1, 10000, 128])
    Q torch.Size([1, 10000, 8, 32])
    K torch.Size([1, 10000, 8, 32])
    V torch.Size([1, 10000, 8, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    topk_features torch.Size([1, 100, 32])
    Output: torch.Size([1, 100, 256])
    '''