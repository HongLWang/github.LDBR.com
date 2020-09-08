#coding=utf-8

import psutil, os, sys
import config
from time import time
from tools import *
from torch import nn
import  torch
import torch.optim as optim
from model.gcn import GraphConvolution
from model.self_attention import MultiHeadedAttention

import torch.nn.functional as F

import warnings

warnings.filterwarnings('ignore')

class Behavior_embedding(torch.nn.Module):

    def __init__(self, feature_dimension, gcn_layer,
                 attention_layer, n_time):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.gcn = gcn_layer
        self.attention = attention_layer
        self.n_time = n_time
        self.lstm = torch.nn.LSTM(config.attention_dimension, config.rnn_h_dim,
                                  bidirectional=False, batch_first=True, dropout=0)

        self.Final_proj1 = torch.nn.Linear(config.rnn_h_dim, 16)
        self.Final_proj2 = torch.nn.Linear(16, 4)
        self.Final_proj3 = torch.nn.Linear(4, 1)
        self.get_prediction_pro = torch.nn.Sequential(
            self.Final_proj1,
            nn.SELU(),
            self.Final_proj2,
            nn.SELU(),
            self.Final_proj3,
        )


    def forward(self, Feature_tensor, adj):
        '''-*/
        :param Feature_tensor: initial feature [n_time, n_node, n_dimensi                                                                                                                                                                              on]
        :param adj: [ n_time, n_node, n_node]
        :return: Dynamic node embedding, [n_node, n_time, attention_dimension]
        '''
        Feature_tensor = self.gcn(Feature_tensor, adj)
        Feature_tensor = Feature_tensor.transpose(0, 1)

        return Feature_tensor


    def link_prediction_training(self, behavior_embedding, task_data):

        fromnode = torch.Tensor([a[0] for a in task_data]).long().cuda()
        tonode = torch.Tensor([a[1] for a in task_data]).long().cuda()
        label = torch.Tensor([a[2] for a in task_data]).float().cuda()
        from_embedding = torch.index_select(behavior_embedding, 0, fromnode)
        to_embedding = torch.index_select(behavior_embedding, 0, tonode)
        element_wise_product = from_embedding.mul(to_embedding)
        prediction = self.get_prediction_pro(element_wise_product)

        return prediction, label


    def link_prediction_test(self, behavior_embedding, from_node, to_node):

        fromnode = torch.Tensor(from_node).long().cuda()
        tonode = torch.Tensor(to_node).long().cuda()
        from_embedding = torch.index_select(behavior_embedding, 0, fromnode)
        to_embedding = torch.index_select(behavior_embedding, 0, tonode)
        element_wise_product = from_embedding.mul(to_embedding)
        prediction = self.get_prediction_pro(element_wise_product)

        return prediction

    def eval(self):
        print ('\n')

def Tongji(model, Data, epoch):
    '''
    :param input_feature: initial feature [n_time, n_node, n_dimension]
            of size [n_time, n_node, attention_dimension]
    :return:
    '''
    input_feature = Data['Feature']
    adj_cuda = Data['adj_cuda']
    adj_cpu = Data['adj_cpu']

    node_feature = model(input_feature,adj_cuda) #[n_node, n_time, attention_dimension]
    attention_feature = model.attention(node_feature, node_feature,node_feature)
    distance_arr = distance_statistic(attention_feature)

    # print ('90% embedding distance at epoch ', epoch)
    sorted_distance = sorted(distance_arr)
    threshold = sorted_distance[int(len(distance_arr)*0.9)][0]
    # print (threshold)
    return threshold


def sample_training(model, Data, epoch, optimizer, threshold):
    '''
    :param input_feature: initial feature [n_time, n_node, n_dimension]
            of size [n_time, n_node, attention_dimension]
    :return:
    '''
    input_feature = Data['Feature']
    adj_cuda = Data['adj_cuda']
    adj_cpu = Data['adj_cpu']
    cum_table = Data['cum_table']

    node_feature = model(input_feature, adj_cuda)  # [n_node, n_time, attention_dimension]
    function = random_sampling_t_n_node(adj_cpu, Data['train'][-1])

    total_loss = 0
    total_instance = 0

    n_time = adj_cpu.shape[0]

    while True:
        try:
            start_time, end_time, chosen_node = function.__next__()
        except:
            break

        attention_mask = torch.ones((1, n_time)).cuda()
        attention_mask[:, end_time + 1:] = 0
        node_feature_after_attention = model.attention(node_feature,
                                                       node_feature, node_feature, attention_mask)

        valid_node = chose_valid(chosen_node, node_feature_after_attention, start_time, end_time, threshold)
        # valid_node = chosen_node
        LinkPre_training_data = get_LP_train_data \
            (adj_cpu, end_time + 1, config.num_false, valid_node, cum_table)
        Attr_training_data = {}
        for t_index in range(start_time, end_time + 1):
            training_data = get_attribute_training_data \
                (adj_cpu, t_index, config.num_false, valid_node, cum_table)
            Attr_training_data[t_index] = training_data

        if len(chosen_node) > 0:
            optimizer.zero_grad()
            chosen_node_embedding = \
                node_feature_after_attention[:, start_time:end_time + 1, :]  # (num_node, num_time, att_dim)
            behavior_embedding = model.lstm(chosen_node_embedding)[0][:, -1, :]
            prediction, label = model.link_prediction_training(behavior_embedding, LinkPre_training_data)
            prediction = torch.sigmoid(prediction)
            label = torch.unsqueeze(label, -1)
            task_loss = F.binary_cross_entropy(prediction, label, reduce=False)

            smooth_loss = get_smooth_loss(chosen_node_embedding[chosen_node, :, :])
            attribute_loss = get_attribute_loss(node_feature_after_attention, Attr_training_data)
            sum_loss = torch.sum(
                task_loss) + config.smooth_rate * smooth_loss + config.att_regularization * attribute_loss

            sum_loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += sum_loss.item()
            total_instance += len(LinkPre_training_data)

        #     if epoch % 10 == 0:
        #         tr_acc, auc = eval_single_label(label, prediction)
        #         print('epoch %d, training auc = %g' % (epoch, auc))
        #         loss_arr = task_loss.cpu().data.numpy()
        #         print('training loss = ', np.min(loss_arr), ' ', np.mean(loss_arr), ' ',
        #               np.median(loss_arr), ' ', np.max(loss_arr))
        # if epoch % 20 == 0:
        #     print('epoch %d, training loss = %g' % (epoch, total_loss))
        #     print('epoch %d, training loss mean= %g' % (epoch, total_loss / total_instance))


# @profile(precision=10)
def behavior_division_v1(model, Data):

    pid = os.getpid()
    p = psutil.Process(pid)

    input_feature = Data['Feature']
    adj_cuda = Data['adj_cuda']
    adj_cpu = Data['adj_cpu'] # (13, 6000, 6000)

    (n_time, n_node, feature_dimension) = input_feature.shape
    node_feature = model(input_feature, adj_cuda)  # [n_node, n_time, attention_dimension]

    task_time_data = {}
    for j in range(n_time-1):
        task_data_vocab = pos_task_data(adj_cpu, j+1)
        task_time_data[j] = task_data_vocab

    for t_index in range(n_time-1):

        attention_mask = torch.ones((1, n_time)).cuda()
        attention_mask[:, t_index+1:] = 0
        node_feature_after_attention = model.attention(node_feature, node_feature, node_feature, attention_mask)

        chosen_node_embedding = \
            node_feature_after_attention[:, 0:t_index+1, :]  # (num_node, num_time, att_dim)
        behavior_embedding = model.lstm(chosen_node_embedding)[0][:, -1, :]


        for node in range(n_node):
            task_data = task_time_data[t_index][node]
            if task_data:
                prediction, label = model.link_prediction_training(behavior_embedding, task_data)
                label = torch.unsqueeze(label, -1)

                prediction = torch.sigmoid(prediction)  # 然后softmax 这样防止所有数据都在正半轴或者负半轴，最后结果更好。
                loss = F.binary_cross_entropy(prediction, label, reduce=False)

                ## 用 loss 决定
                mean_loss = torch.mean(loss).item()
                if mean_loss >= 4:
                    print("node %d, after t_index %d" % (node, t_index))



def link_prediction_testing(model, Data, mode):
    '''

    :param model: Behavior_embedding
    :param Data: model forward data
    :return:
    '''
    input_feature = Data['Feature']
    adj_cuda = Data['adj_cuda']
    adj_cpu = Data['adj_cpu'] # (13, 6000, 6000)
    (n_time, n_node, feature_dimension) = input_feature.shape
    node_feature = model(input_feature, adj_cuda)  # [node ,time ,dimension]

    time_range = Data[mode]
    task_time_label = {}
    for j in range(time_range[0], time_range[1] + 1):
        label_vocab = adj_cpu[j]
        task_time_label[j] = label_vocab


    res = []
    label = []

    for t_index in range(time_range[0], time_range[1] + 1):
        attention_mask = torch.ones((1, n_time)).cuda()
        attention_mask[:, t_index:] = 0
        node_feature_after_attention = model.attention(node_feature, node_feature, node_feature, attention_mask)

        chosen_node_embedding = \
            node_feature_after_attention[:, 0:t_index, :]  # (num_node, num_time, att_dim)
        behavior_embedding = model.lstm(chosen_node_embedding)[0][:, -1, :]



        for node in range(n_node):
            from_node = np.array([node]*n_node) # this node to test
            to_node = np.arange(n_node)  #candidate all node
            prediction = model.link_prediction_test(behavior_embedding, from_node, to_node)
            prediction[node] = torch.Tensor([-(10**9)])
            prediction = torch.sigmoid(prediction)

            prediction_np = prediction.cpu().data.numpy()
            prediction_np=np.squeeze(prediction_np, -1)


            y_true = task_time_label[t_index][node].data.numpy()
            y_true = y_true.reshape((len(y_true),1))
            y_true=np.squeeze(y_true, -1)
            y_true = y_true.astype(bool)

            try:
                single_auc = roc_auc_score(y_true, prediction_np)
                label.append(y_true)
                res.append(prediction_np)

                # if node %100 ==0:
                #     print('node : ', node, 'auc : ', single_auc)
            except ValueError:
                pass

    res = np.array(res)
    label = np.array(label)
    y_pred = res.reshape((res.shape[0]*res.shape[1], 1))
    y_true = label.reshape((label.shape[0]*label.shape[1], 1))

    link_prediiction_test_auc = roc_auc_score(y_true, y_pred)
    # print (mode, 'total_link_prediiction_test_auc: ', link_prediiction_test_auc)

    return link_prediiction_test_auc




# @profile(precision=10)
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fp = '../Data/reddit.npz'
    files = np.load(fp)
    print(list(files.keys()))

    Graphs_ori = files['adjs']  # (13, 6000, 6000) [n_time, n_node, n_node]
    print(Graphs_ori.shape)

    Features = files['attmats']  # (n_node, n_time, feature_dimension)
    Features = np.transpose(Features, (1, 0, 2))  # (n_time, n_node, feature_dimension)
    print(Features.shape)


    (n_time, n_node, feature_dimension) = Features.shape

    gcn_layer = GraphConvolution(feature_dimension, config.attention_dimension, n_time)
    attention_layer = MultiHeadedAttention(config.h, config.attention_dimension)

    regulazation_model = {}
    regulazation_model['reg_att_loss_model'] = torch.nn.MSELoss()
    regulazation_model['prediction_loss_model'] = torch.nn.CrossEntropyLoss()
    prediction_loss = torch.nn.CrossEntropyLoss\
        (weight=None, size_average=None, reduce=None, reduction='none')
    regulazation_model['prediction_loss_model'] = prediction_loss


    sample_cum_table = {}
    for t_index in range(n_time):
        graph_t = Graphs_ori[t_index, :, :] + 1
        node_degree = np.sum(graph_t, 0)
        cum_table = make_cum_table(index2freq=node_degree)
        sample_cum_table[t_index] = cum_table


    I = np.eye(n_node)
    I = np.array([I for k in range(n_time)])
    A =  Graphs_ori + I
    for index in range(n_time):
        a = A[index]
        d = np.sum(a, 0)
        D = np.diag(d ** (-1./2))
        normalized_A = np.dot(np.dot(D, a), D)
        A[index] = normalized_A


    Data = {}
    Data['Feature'] = torch.Tensor(Features).float().cuda()
    Data['adj_cuda'] = torch.Tensor(A).cuda()
    Data['adj_cpu'] = torch.Tensor(Graphs_ori)
    Data['cum_table'] = sample_cum_table



    dynamic_model =  Behavior_embedding(feature_dimension, gcn_layer,
                                        attention_layer, n_time).cuda()

    optimizer = optim.Adam(dynamic_model.parameters(), lr=config.lr, weight_decay=config.lambda_l2_reg)

    max_auc, max_iteration = train_epoch(dynamic_model, Data, optimizer)


def train_epoch(dynamic_model, Data, optimizer):

    num_ts = Data['adj_cpu'].shape[0]
    train_ts = [0, round(num_ts * config.train)]
    valid_ts = [train_ts[-1] + 1,  round(num_ts * (config.train + config.validation))]
    test_ts = [valid_ts[-1]+1, num_ts-1]
    Data['train'] = train_ts
    Data['valid'] = valid_ts
    Data['test'] = test_ts


    max_auc = 0
    max_iteration = 0
    for epoch in range(config.epoch):
        threshold = Tongji(dynamic_model, Data, epoch)
        sample_training(dynamic_model, Data, epoch, optimizer, threshold)    #
        if epoch % 10 ==0:
            torch.save(dynamic_model.state_dict(),
    './save_model/reddit-best-' + str(epoch)  + '.pkl')
            total_auc = link_prediction_testing(dynamic_model, Data, 'test')

            if total_auc > max_auc:
                max_auc = total_auc
                max_iteration = epoch

    print ('max auc for all epochs', max_auc, ' at epoch ' , max_iteration)

    return max_auc, max_iteration

if __name__ == '__main__':
    st = time()
    main()
    print ('total_time is , ',time()-st)
