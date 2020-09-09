#coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import pickle
from torch import nn

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, n_time_step,bias=True):
        super(GraphConvolution, self).__init__()
        self.n_time_step = n_time_step
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.out_features))
        if bias:
            self.bias1 = Parameter(torch.FloatTensor(self.n_time_step, 1, out_features))
            self.bias2 = Parameter(torch.FloatTensor(self.n_time_step, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.actication = nn.ReLU()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)

        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias2 is not None:
            self.bias2.data.uniform_(-stdv2, stdv2)


    def forward(self, Feature_tensor, adj):

        # support = torch.mm(input, self.weight)
        support1 = Feature_tensor.matmul(self.weight1)
        # output = torch.spmm(adj, support)
        output = adj.matmul(support1)

        # if self.bias1 is not None:
        #     return nn.RelU(output + self.bias1)
        # else:
        #     return nn.RelU(output)


        support2 = self.actication(output).matmul(self.weight2)
        output = adj.matmul(support2)


        if self.bias2 is not None:
            return self.actication(output + self.bias2)
        else:
            return self.actication(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


