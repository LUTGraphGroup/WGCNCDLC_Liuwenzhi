from __future__ import division
from __future__ import print_function
import time
import argparse
import torch.optim as optim
from utils import *
from models import *
from metric import *
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')  # seed=0
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')  # epochs=200
parser.add_argument('--lr', type=float, default=0.005,  # lr=0.01
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')  # weight_decay=5e-4
parser.add_argument('--hidden', type=int, default=64,  # hidden=16
                    help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=512,  # hidden=16
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,  # dropout=0.5
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

print('layer', 2)
print('hidden', args.hidden)
print('nclass', args.nclass)
print('dropout', args.dropout)
Adj, Or_Adj, Dis_simi, Meta_simi, feature, random_index, k_folds = load_data(args.seed)

auc_result = []
acc_result = []
pre_result = []
recall_result = []
f1_result = []
prc_result = []
fprs = []
tprs = []
precisions = []
recalls = []
print("seed=%d, evaluating metabolite-disease...." % args.seed)
for k in range(k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    Or_Adj_1 = np.matrix(Or_Adj, copy=True)
    Or_train = np.matrix(Or_Adj, copy=True)

    val_pos_edge_index = np.array(random_index[k]).T  # 验证集边索引，正样本
    val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long).to(device)  # tensor格式，验证集正样本
    # 验证集负采样，采集与正样本相同数量的负样本
    val_neg_edge_index = np.mat(np.where(Or_train < 1)).T.tolist()
    random.seed(args.seed)
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = val_neg_edge_index[:val_pos_edge_index.shape[1]]
    val_neg_edge_index = np.array(val_neg_edge_index).T
    val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).to(device)   # tensor格式，验证集负样本

    Or_train[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0
    train_pos_edge_index = np.mat(np.where(Or_train > 0))  # 训练集边索引，正样本
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)   # tensor格式，训练集正样本


    train_matrix = np.matrix(Adj, copy=True)
    train_matrix[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0



    # adj = constructNet(train_matrix)  # 仅拼接原始邻接矩阵，不加入相似性
    adj = constructHNet(train_matrix, Meta_simi, Dis_simi)
    adj_normal = normalize_adj(adj)
    adj_normal = sparse_mx_to_torch_sparse_tensor(adj_normal).to(device)
    feature = feature


    # Model and optimizer
    model = GCN(nfeat=feature.shape[1],
                nhid=args.hidden,  # nhid=64
                nclass=args.nclass,  # nhid=64
                dropout=args.dropout,  # dropout=0.5
                num_r=Meta_simi.shape[0],  # 解码器分割代谢物和疾病矩阵的维数
                device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = F.binary_cross_entropy  # _with_logits  # 交叉熵损失函数
    best_auc = 0
    best_prc = 0
    best_epoch = 0
    best_tpr = 0
    best_fpr = 0
    best_recall = 0
    best_precision = 0
    for epoch in range(args.epochs):
        start = time.time()  # 返回当前时间
        model.train()
        optimizer.zero_grad()  # 梯度清零
        # 训练集负采样，采集与正样本相同数量的负样本. 此部分应在train模块中，因为每次训练取不同的负样本可以提高训练效果。
        train_neg_edge_index = np.mat(np.where(train_matrix < 1)).T.tolist()
        # validate = [x for x in train_neg_edge_index if x not in validate_pos_index]

        random.shuffle(train_neg_edge_index)
        train_neg_edge_index = train_neg_edge_index[:train_pos_edge_index.shape[1]]
        train_neg_edge_index = np.array(train_neg_edge_index).T
        train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)  # tensor格式训练集负样本

        met_len = Adj.shape[0]
        dis_len = Adj.shape[1]
        output = model(feature, adj_normal).to(device)
        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], 1)
        trian_scores = output[edge_index[0], edge_index[1]].to(device)

        trian_labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(device)  # 训练集中正样本标签

        # 运行模型，输入参数 (features, adj)
        loss_train = criterion(trian_scores, trian_labels).to(device)
        loss_train.backward(retain_graph=True)  # 反向传播
        optimizer.step()  # 参数更新
        model.eval()
        with torch.no_grad():  # 禁用梯度计算，以避免跟踪计算图中的梯度
            score_train_cpu = np.squeeze(trian_scores.detach().cpu().numpy())
            label_train_cpu = np.squeeze(trian_labels.detach().cpu().numpy())
            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)

            predict_y_proba = output.reshape(met_len, dis_len).to(device)
            score_val, label_val, metric_tmp = cv_model_evaluate(predict_y_proba, val_pos_edge_index, val_neg_edge_index)

            fpr, tpr, thresholds = metrics.roc_curve(label_val, score_val)
            precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
            val_auc = metrics.auc(fpr, tpr)
            val_prc = metrics.auc(recall, precision)

            end = time.time()
            # if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                  'Acc: %.4f' % metric_tmp[0], 'Pre: %.4f' % metric_tmp[1], 'Recall: %.4f' % metric_tmp[2],
                  'F1: %.4f' % metric_tmp[3],
                  'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Val PRC: %.4f' % val_prc,
                  'Time: %.2f' % (end - start))
            if val_auc > best_auc:
                metric_tmp_best = metric_tmp
                best_auc = val_auc
                best_prc = val_prc
                best_epoch = epoch + 1
                best_tpr = tpr
                best_fpr = fpr
                best_recall = recall
                best_precision = precision

    print('Fold:', k + 1, 'Best Epoch:', best_epoch, 'Val acc: %.4f' % metric_tmp_best[0],
              'Val Pre: %.4f' % metric_tmp_best[1],
              'Val Recall: %.4f' % metric_tmp_best[2], 'Val F1: %.4f' % metric_tmp_best[3], 'Val AUC: %.4f' % best_auc,
              'Val PRC: %.4f' % best_prc,
              )

    acc_result.append(metric_tmp_best[0])
    pre_result.append(metric_tmp_best[1])
    recall_result.append(metric_tmp_best[2])
    f1_result.append(metric_tmp_best[3])
    auc_result.append(best_auc)
    prc_result.append(best_prc)

    fprs.append(best_fpr)
    tprs.append(best_tpr)
    recalls.append(best_recall)
    precisions.append(best_precision)

print('## Training Finished !')
print('-----------------------------------------------------------------------------------------------')
print('Acc', acc_result)
print('Pre', pre_result)
print('Recall', recall_result)
print('F1', f1_result)
print('Auc', auc_result)
print('Prc', prc_result)
print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
        'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
        'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
        'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
        'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
        'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

pd.DataFrame(recalls).to_csv('../result/recalls.csv', index=False)
pd.DataFrame(precisions).to_csv('../result/precisions.csv', index=False)
print('fprs', fprs)
print('tprs', tprs)
print('recalls', recalls)
print('precisions', precisions)
# 画五折AUC和PR曲线
plot_auc_curves(fprs, tprs, auc_result, directory='../result', name='test_auc')
plot_prc_curves(precisions, recalls, prc_result, directory='../result', name='test_prc')