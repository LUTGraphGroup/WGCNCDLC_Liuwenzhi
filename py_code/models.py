import torch.nn as nn
import torch.nn.functional as F
from layers import *



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_r, device):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout).to(device)  # 构建第一层 GCN
        self.gc2 = GraphConvolution(nhid, nclass, dropout).to(device)  # 构建第二层 GCN
        self.gc3 = GraphConvolution(nclass, nclass, dropout).to(device)  # 构建第三层 GCN
        self.decoder = InnerProductDecoder(nclass, dropout, num_r).to(device)  # 解码器
        self.dropout = dropout
        self.num_r = num_r
        self.device = device

    def forward(self, x, adj):
        x1 = self.gc1(x, adj).to(self.device)
        x2 = self.gc2(x1, adj).to(self.device)
        # x3 = self.gc3(x2, adj).to(self.device)
        embeddings = x2
        x4 = self.decoder(embeddings).to(self.device)
        return x4

