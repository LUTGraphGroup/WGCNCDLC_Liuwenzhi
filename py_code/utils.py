import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import interp
import networkx as nx
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def constructHNet(met_dis_matrix, met_matrix, dis_matrix):
    mat1 = np.hstack((met_matrix, met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    adj = sp.csc_matrix(adj)
    # # 保存结果
    # result = pd.DataFrame(adj)
    # result.to_excel('../output/adj.xlsx', index=False)  # index=False设置不生成序号列
    return adj


def constructNet(met_dis_matrix):  # 构造网络G，分块形式，对角线分块为0, 初始化特征矩阵
    met_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[0], met_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[1], met_dis_matrix.shape[1]), dtype=np.int8))
    mat1 = np.hstack((met_matrix, met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T, dis_matrix))
    adj_0 = np.vstack((mat1, mat2))
    adj_0 = sp.csc_matrix(adj_0)
    # # 保存结果
    # result = pd.DataFrame(adj_0)
    # result.to_excel('../output/adj_0.xlsx', index=False)  # index=False设置不生成序号列
    return adj_0


def normalize_adj(mx):
    """normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 矩阵行求和
    r_inv = np.power(rowsum, -0.5).flatten()  # 求和的-1/2次方
    r_inv[np.isinf(r_inv)] = 0.  # 如果是inf(无穷大)，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角矩阵
    #  用r_mat_inv.A查看元素
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)  # 构造D-1/2*A*D-1/2
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(seed):
    Adj = pd.read_csv('../data/A_new_greedy.csv', header=0)
    Or_Adj = pd.read_csv('../data/association_matrix.csv', header=0)  # 未补全的邻接矩阵
    non_zero_count = np.count_nonzero(Adj)
    count_ones = np.count_nonzero(Adj == 1)
    print("非零元素个数：", non_zero_count)
    print("元素为1的个数：", count_ones)
    print('Adj', Adj)

    Dis_simi = pd.read_csv('../data/diease_network_simi.csv', header=0)
    print('Dis_simi', Dis_simi)
    Meta_simi = pd.read_csv('../data/metabolite_ntework_simi.csv', header=0)
    print('Meta_simi', Meta_simi)
    Dis_simi_tensor = torch.tensor(Dis_simi.to_numpy()).to(device)
    Meta_simi_tensor = torch.tensor(Meta_simi.to_numpy()).to(device)

    # 使用全连接层将特征映射到256维
    linear_disease = nn.Linear(265, 256).to(device)  # 疾病特征映射层
    linear_metabolite = nn.Linear(2315, 256).to(device)  # 代谢物特征映射层
    # 特征映射
    mapped_disease_feature = linear_disease(Dis_simi_tensor.float()).to(device)  # 映射后的疾病特征，大小为1x256
    mapped_metabolite_feature = linear_metabolite(Meta_simi_tensor.float()).to(device)  # 映射后的代谢物特征，大小为1x256

    feature = torch.cat((mapped_metabolite_feature, mapped_disease_feature), dim=0).to(device)

    # 训练，验证，测试的样本
    index_matrix = np.mat(np.where(Adj == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)  # random.seed(): 设定随机种子，使得random.shuffle随机打乱的顺序一致
    random.shuffle(random_index)  # random.shuffle将random_index列表中的元素打乱顺序
    k_folds = 5

    CV_size = int(association_nam / k_folds)  # 每折的个数
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()  # %取余
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]  # 将余下的元素加到最后一折里面
    random_index = temp

    return Adj, Or_Adj, Dis_simi, Meta_simi, feature, random_index, k_folds



def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f' % mean_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AUPR: %.4f' % mean_prc)  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()



def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1

    return link_labels
















