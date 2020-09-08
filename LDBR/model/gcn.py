#coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import pickle
from torch import nn

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    in this program different time slice share the same weight parameter
    maybe this is a place for improvment
    """

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


        # 如果是两层
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



if __name__ == '__main__':

    '''上面定义的GCN是单个网络上适用的。下面测试一下，在多个时间片，也就是一个tensor的情况下，能不能运行
    尤其是，当一个节点是孤立节点的时候
    事实证明，因为是矩阵运算。如果点是孤立点，就不发生什么就好了
    下一个问题： 不同时间片都用同样的GCN的参数么？？ 最相近的那篇文章，是这样的。但是鉴于W是一个权重，不同时间片，
    这个W应该是不一样的才对。这个如何衡量呢？
    可以先只初始化一个。后续再优化
    '''

    file = np.load('../Data/DBLP3.npz')
    Labels = file['labels']  # (n_node, num_classes)
    Graphs_ori = file['adjs']
    print (Graphs_ori.shape )
    fout = open('../Data/normalized_feature.pkl', 'rb')
    Features = pickle.load(fout) # (n_node, n_time, att_dim)
    fout.close()
    print (Features.shape)
    (n_node, n_time, att_dim) = Features.shape


    Features = torch.Tensor(Features)
    Features = Features.transpose(0,1)
    att_dim = Features.shape[2]
    print (Features.shape)


    adj = torch.Tensor(Graphs_ori)
    # label = Labels[t_index,:,:]
    # print (feature.shape)
    model  = GraphConvolution(att_dim, 64, n_time)
    representation = model(Features, adj)

    #Features  n_time, n_node, feature dimension
    # weight feature_dimension, out dimension
    # adj n_time, n_node, n_node

    # support = feature * weight   [(n_time, n_node,n_dimension) mul (n_dimension, out_dimension)] = [n_time, n_node, out_dimension]
    #adj * support  (n_time, n_node, n_node) matmum (n_time,n_node, out_dimension) = [n_time, n_node, out_dimension]



