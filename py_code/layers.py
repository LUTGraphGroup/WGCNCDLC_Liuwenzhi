import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input_1 = F.dropout(input, self.dropout, training=self.training)
        support = torch.mm(input_1, self.weight)  # GraphConvolution forward。input*weight
        output = torch.spmm(adj, support)  # 稀疏矩阵的相乘，和mm一样的效果
        if self.bias is not None:
            return F.relu(output + self.bias)
        else:
            return F.relu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class InnerProductDecoder(Module):
    """
    decoder 解码器
    """
    def __init__(self, input_dim, dropout, num_r):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(input_dim, input_dim)))  # 建立一个w权重，用于对特征数进行线性变化
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)  # 对权重矩阵进行初始化
        self.dropout = dropout
        self.num_r = num_r

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout)
        M = inputs[0:self.num_r, :]
        D = inputs[self.num_r:, :]
        M = torch.mm(M, self.weight)
        D = torch.t(D)  # 转置
        x = torch.mm(M, D)
        # x = torch.reshape(x, [-1])  # 转化为行向量
        outputs = torch.sigmoid(x)
        return outputs


