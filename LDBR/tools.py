#coding=utf-8


import numpy as np
from sklearn import preprocessing
from config import config
import torch
from sklearn.metrics import f1_score, roc_auc_score
import random
from six import iteritems
import line_profiler
def meanstd_normalization_tensor(tensor): #numpy fucntion

    n_node, n_steps, n_dim = tensor.shape
    tensor_reshape = preprocessing.scale(np.reshape(tensor, [n_node, n_steps * n_dim]), axis=1)
    tensor_norm = np.reshape(tensor_reshape, [n_node, n_steps, n_dim])

    return tensor_norm


def eval(y_true, y_pred):

    res_pred = torch.argmax(y_pred, 1)
    res_real = torch.argmax(y_true, 1)
    correct_pred = res_pred.eq(res_real).float()
    accuracy = torch.mean(correct_pred)

    auc_mac = roc_auc_score(res_real.cpu(), res_pred.cpu(), average='macro')
    auc_mic = roc_auc_score(res_real.cpu(), res_pred.cpu(), average='micro')

    # F1
    f1_mac = f1_score(res_real.cpu(),  res_pred.cpu(), average='macro')
    f1_mic = f1_score(res_real.cpu(),  res_pred.cpu(),  average='micro')



    return accuracy, auc_mac, auc_mic, f1_mac, f1_mic



def eval_single_label(y_true, y_pred):

    num_data = len(y_true)

    replace_0 = torch.zeros((num_data, 1)).cuda()
    replace_1 = torch.ones((num_data, 1)).cuda()
    res_pred = torch.where(y_pred > 0.5, replace_1, replace_0)
    correct_pred = (res_pred == y_true).float()
    accuracy = torch.mean(correct_pred)

    y_true_ = y_true.cpu().data.numpy()
    y_pred_ = y_pred.cpu().data.numpy()
    auc =  roc_auc_score(y_true_, y_pred_)

    return accuracy, auc


def eval_po(y_true, y_pred):

    res_pred = torch.argmax(y_pred, 1)
    res_real = torch.argmax(y_true, 1)
    correct_pred = res_pred.eq(res_real).float()
    accuracy = torch.mean(correct_pred)

    return accuracy

def load_link_prediction_task_fast(graph, t_index, num_false):

    pos_node_pairs = []
    neg_node_pairs = []

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in range(n_node):
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        neg_col = np.where(nodelink == -1)[0]

        pos_node_pairs += [(row, a, [0,1]) for a in pos_col]
        neg_col_selected = np.random.choice(neg_col, num_false, replace= True)


        if row in set(neg_col_selected):
            exsit_row_index = np.where(neg_col_selected == row)[0]
            neg_col_selected = np.delete(neg_col_selected, exsit_row_index)

        neg_node_pairs += [(row, a, [1,0]) for a in neg_col_selected]

    pos_node_pairs += neg_node_pairs

    random.shuffle(pos_node_pairs)

    return pos_node_pairs


def load_link_prediction_task_singlelabel(graph, t_index, num_false):

    pos_node_pairs = []
    neg_node_pairs = []

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in range(n_node):
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        neg_col = np.where(nodelink == -1)[0]

        pos_node_pairs += [(row, a, 1) for a in pos_col]
        neg_col_selected = np.random.choice(neg_col, num_false, replace= True)


        if row in set(neg_col_selected):
            exsit_row_index = np.where(neg_col_selected == row)[0]
            neg_col_selected = np.delete(neg_col_selected, exsit_row_index)

        neg_node_pairs += [(row, a, 0) for a in neg_col_selected]

    pos_node_pairs += neg_node_pairs

    random.shuffle(pos_node_pairs)

    return pos_node_pairs

def load_link_prediction_task_posonly_fast(graph, t_index, num_false): 

    pos_node_pairs = []

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1, graph_t, -1)

    for row in range(n_node):
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]

        pos_node_pairs += [(row, a, [0, 1]) for a in pos_col]


    random.shuffle(pos_node_pairs)

    return pos_node_pairs


def load_link_prediction_chosen_node(graph, t_index, num_false, chosen_node): 

    pos_node_pairs = []
    neg_node_pairs = []

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in chosen_node:
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        neg_col = np.where(nodelink == -1)[0]

        pos_node_pairs += [(row, a, [0,1]) for a in pos_col]
        neg_col_selected = np.random.choice(neg_col, num_false, replace= True)


        if row in set(neg_col_selected):
            exsit_row_index = np.where(neg_col_selected == row)[0]
            neg_col_selected = np.delete(neg_col_selected, exsit_row_index)

        neg_node_pairs += [(row, a, [1,0]) for a in neg_col_selected]

    pos_node_pairs += neg_node_pairs

    random.shuffle(pos_node_pairs)

    return pos_node_pairs


def find_task_data_vocab_positive(graph, t_index, num_false): 

    vocab = {}

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in range(n_node):
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]

        pos_node_pairs = [(row, a, [0,1]) for a in pos_col]
        vocab[row] = pos_node_pairs

    return vocab


def pos_task_data(graph, t_index):

    vocab = {}

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in range(n_node):
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]

        pos_node_pairs = [(row, a, 1) for a in pos_col]
        vocab[row] = pos_node_pairs

    return vocab

def find_task_data_vocab_allInstance(graph, t_index, num_false): 

    real_neighbor = {}

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1, graph_t, -1)

    if t_index == 1:
        print ('real neighbor for node 0 in time slice 0')
        print (np.where(graph_t[0] >= 1))

    for row in range(n_node):

        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        real_neighbor[row] = pos_col

    return  real_neighbor



def find_task_data_vocab_allInstance(graph, t_index, num_false): 

    real_neighbor = {}

    graph_t = graph[t_index, :, :]
    n_node = graph_t.shape[0]
    link = np.where(graph_t >= 1, graph_t, -1)

    if t_index == 1:
        print ('real neighbor for node 0 in time slice 0')
        print (np.where(graph_t[0] >= 1))

    for row in range(n_node):

        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        real_neighbor[row] = pos_col

    return  real_neighbor


def get_smooth_loss(node_embedding):

    n_time = node_embedding.shape[1]
    smoothing_loss = 0
    for time_index in range(n_time-1):
        pre_emb = node_embedding[:, time_index, :]
        pro_emb = node_embedding[:, time_index+1, :]
        distance = torch.norm(pre_emb-pro_emb, dim = 1)
        smoothing_loss += torch.sum(distance)
    return smoothing_loss


def get_attribute_loss(node_feature, attribute_task_data):
    attribute_loss = 0

    for t_index, task_data in iteritems(attribute_task_data):
        node_embedding = node_feature[:,t_index,:]
        fromnode = torch.Tensor([a[0] for a in task_data]).long().cuda()
        tonode = torch.Tensor([a[1] for a in task_data]).long().cuda()
        label = torch.Tensor([a[2] for a in task_data]).float().cuda()
        from_embedding = torch.index_select(node_embedding, 0, fromnode)
        to_embedding = torch.index_select(node_embedding, 0, tonode)
        dot_product = torch.matmul(from_embedding, torch.transpose(to_embedding, -1, -2))
        dot_product = dot_product.mul(label).float()
        att_loss = torch.sigmoid(dot_product)
        sum_att_loss = torch.sum(att_loss)

        attribute_loss += sum_att_loss

    return attribute_loss




def random_sampling(adj, node_embedding, cum_table, uptime):

    [n_time, n_node, n_node] = adj.shape
    sample_times = config.training_sample_time
    for k in range(sample_times):
        start_time = np.random.randint(uptime-1) 
        end_time = np.random.randint(start_time+1, uptime) 
        chosen_node = np.random.randint(n_node, size = config.batch_size)
        valid_node = chose_valid(chosen_node, node_embedding,start_time, end_time)
        #link prediction training data -> link condition in end_time + 1
        LinkPre_training_data = get_LP_train_data\
            (adj, end_time + 1, config.num_false, valid_node, cum_table)

        # attribute training data during start time and end time. return the link condition and unlink condition
        Attr_training_data = {}
        for t_index in range(start_time, end_time + 1):
            training_data = get_attribute_training_data\
                (adj, t_index, config.num_false, valid_node, cum_table)
            Attr_training_data[t_index] = training_data

        yield start_time, end_time, LinkPre_training_data, Attr_training_data, valid_node



def random_sampling_t_n_node(adj,uptime):

    [n_time, n_node, n_node] = adj.shape
    sample_times = config.training_sample_time
    for k in range(sample_times):
        start_time = np.random.randint(uptime-1)
        end_time = np.random.randint(start_time+1, uptime)
        chosen_node = np.random.randint(n_node, size = config.batch_size)
        yield start_time, end_time, chosen_node


def chose_valid(chosen_node, node_embedding, start_time, end_time,threshold):
    Distance = torch.zeros((len(chosen_node), 1)).cuda()
    for t in range(start_time,  end_time- 1):
        PreEmbed = node_embedding[chosen_node, t, :]
        ProEmbed = node_embedding[chosen_node, t + 1, :]
        distance = torch.norm(PreEmbed - ProEmbed, dim=1)
        distance = torch.unsqueeze(distance, -1)
        Distance += distance
    Distance = (Distance/ (end_time-start_time*1.0) ).cpu().data.numpy()
    Distance = np.squeeze(Distance, -1)
    valid = Distance < threshold
    valid_node = chosen_node[valid]

    return valid_node

def distance_statistic( node_embedding):

    all_dist = []
    (num_node, n_time, att_dim) = node_embedding.shape
    for t in range(n_time-1):
        PreEmbed = node_embedding[:,t,:]
        ProEmbed = node_embedding[:,t+1,:]
        distance = torch.norm(PreEmbed-ProEmbed, dim=1)
        all_dist.append(distance.cpu().data.numpy())

    all_dist = np.array(all_dist)
    all_dist = all_dist.reshape((-1,1))



    return all_dist



def get_LP_train_data(graph, t_index, num_false, chosen_node, cum_table): 

    pos_node_pairs = []
    neg_node_pairs = []

    graph_t = graph[t_index, :, :]
    cum_table_t = cum_table[t_index]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in chosen_node:
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        pos_node_pairs += [(row, a, 1) for a in pos_col]

        neg_col = []
        # while len(neg_col) < num_false:
        while len(neg_col) < num_false*len(pos_col):
            w = cum_table_t.searchsorted(np.random.randint(cum_table_t[-1]))
            if not w in pos_col:
                neg_col.append(w)
        neg_node_pairs += [(row, a, 0) for a in neg_col]

    pos_node_pairs += neg_node_pairs

    random.shuffle(pos_node_pairs)

    return pos_node_pairs


def get_attribute_training_data(graph, t_index, num_false, chosen_node, cum_table): 

    pos_node_pairs = []
    neg_node_pairs = []

    graph_t = graph[t_index, :, :]
    cum_table_t = cum_table[t_index]
    link = np.where(graph_t >= 1,graph_t,-1)

    for row in chosen_node:
        nodelink = link[row, :]
        pos_col = np.where(nodelink >= 1)[0]
        pos_node_pairs += [(row, a, 1) for a in pos_col]

        neg_col = []
        while len(neg_col) < num_false *config.num_false:
            w = cum_table_t.searchsorted(np.random.randint(cum_table_t[-1]))
            if not w in pos_col:
                neg_col.append(w)

        neg_node_pairs += [(row, a, -1) for a in neg_col]

    pos_node_pairs += neg_node_pairs

    random.shuffle(pos_node_pairs)

    return pos_node_pairs

def make_cum_table(domain=2 ** 31 - 1,  index2freq = []):

    ns_exponent = 0.75
    vocab_size = len(index2freq)
    cum_table = np.zeros(vocab_size, dtype=np.uint32)
    # compute sum of all power (Z in paper)
    train_words_pow = 0.0
    for word_index in range(vocab_size):
        train_words_pow += index2freq[word_index] ** ns_exponent

    cumulative = 0.0
    for word_index in range(vocab_size):
        cumulative += index2freq[word_index] ** ns_exponent
        cum_table[word_index] = round(cumulative / train_words_pow * domain)
    if len(cum_table) > 0:
        assert cum_table[-1] == domain

    return cum_table





























