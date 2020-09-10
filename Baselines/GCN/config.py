#coding=utf-8
 

import numpy as np
import argparse



ld_reg = 0.01

parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('--filedpath', type=str, default='./Data')
parser.add_argument('--filename', type=str, default='/DBLP3.npz')
parser.add_argument('--n_hop', type=int, default=2)
parser.add_argument('--n_sample', type=int, default=4)







#attention 
parser.add_argument('--h', type=int, default=4) #  attention multi-head
parser.add_argument('--attention_dimension', type=int, default=64)  # feature dimension
parser.add_argument('--rnn_h_dim', type=int, default=48)  # feature dimension


# training 
parser.add_argument('--num_false', type=int, default=2)
parser.add_argument('--sample_sizes', type=list, default=[4,4])
parser.add_argument('--n_nbor', type=int, default=40)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_stacked_layers', type=int, default=1)
parser.add_argument('--M', type=int, default= 10)  # equals n_hidden_units =10
parser.add_argument('--n_h_units', type=int, default=10)  # feature dimension
parser.add_argument('--ld_reg', type=float, default=ld_reg)  # feature dimension
parser.add_argument('--lambda_l2_reg', type=float, default=1e-5)  # feature dimension
parser.add_argument('--lambda_reg_att', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=0.0025)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--domain', type=int, default=2 ** 31 - 1)
parser.add_argument('--smooth_rate', type=float, default=1e-5)
parser.add_argument('--att_regularization', type=float, default=1e-5)




#testing 
parser.add_argument('--topk', type=int, default=100)  # if real neighbor not in topk
parser.add_argument('--train', type=float, default=0.7)  # if real neighbor not in topk
parser.add_argument('--validation', type=float, default=0.1)  # if real neighbor not in topk
parser.add_argument('--test', type=float, default=0.2)  # if real neighbor not in topk



config = parser.parse_args()
config.training_sample_time = int(6000/config.batch_size)
