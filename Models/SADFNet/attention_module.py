import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(AttentionFusion, self).__init__()
        self.query_layer = nn.Linear(feature_dim, attention_dim)
        self.key_layer = nn.Linear(feature_dim, attention_dim)
        self.value_layer = nn.Linear(feature_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        attention_scores = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5))
        context = torch.bmm(attention_scores, V)
        return context
