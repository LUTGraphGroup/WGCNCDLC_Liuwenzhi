import networkx as nx
import pandas as pd
import numpy as np

SimDis = pd.read_csv('data/diease_network_simi.csv', header=0).values
SimMet = pd.read_csv('data/metabolite_ntework_simi.csv', header=0).values
A = pd.read_csv('data/association_matrix.csv', header=0).values

INDEX = []
for i in range(len(A)):     # len(A) = 2260, 是矩阵的行，代谢物的索引
    for j in range(len(A[i])):
        index = (i, j)
        value = A[i][j]
        if value == 1.0:
            INDEX.append(index)

# 构建疾病和代谢物相似性网络
G_dis = nx.Graph()
for i in range(len(SimDis)):
    for j in range(i+1, len(SimDis[i])):
        weight = SimDis[i][j]
        G_dis.add_edge(i, j, weight=weight)

G_met = nx.Graph()
for i in range(len(SimMet)):
    for j in range(i+1, len(SimMet[i])):
        weight = SimMet[i][j]
        G_met.add_edge(i, j, weight=weight)

# 使用greedy_modularity_communities进行社团检测
def greedy_modularity(G,n):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight', resolution=n))
    print('communities', communities)
    # 将社团划分结果存储为列表嵌套的形式
    nested_communities = [list(community) for community in communities]
    print('nested_communities', nested_communities)
    for i, community in enumerate(nested_communities):
        print(f'Community {i + 1}: {community}')
    return nested_communities

def Strength(C1, C2):  # 前面填代谢物的社团，后面填疾病的社团
    count = 0
    for i in range(0, len(C1)):
        for j in range(0, len(C2)):
            if (C1[i], C2[j]) in INDEX:
                count = count+1
    return count / (len(C1)*len(C2))


def CDS(C1, C2):       # C1是所有划分完成的代谢物社团，C2是所有划分完成的疾病社团，n是输出前多少个关联社团
    COMMU = []
    VALUE = []
    for i in range(0, len(C1)):
        INDEX = []
        S = []
        for j in range(0, len(C2)):
            s = Strength(C1[i], C2[j])
            index = (i, j)
            S.append(s)
            INDEX.append(index)
        COMMU.append(INDEX)
        VALUE.append(S)
    return COMMU, VALUE


def Relationship_Complete(C1, C2, A = A):
    com, val = CDS(C1, C2)
    A_new = A.astype(float)
    for i in range(0, len(C1)):
        for j in range(0, len(C2)):
            pairs = [[x, y] for x in C1[i] for y in C2[j]]
            entry = val[i][j]
            for p in pairs:
                m = p[0]
                d = p[1]
                if A_new[m, d] != 1:
                    A_new[m, d] = entry
    return A_new


def normalized(Adj):
    # 找出位于0和1之间的数（不包括0和1）
    Adj = pd.DataFrame(Adj)
    filtered_data = Adj[(Adj > 0) & (Adj < 1)]

    # 将DataFrame转换为Series，然后找到最大值和最小值
    max_value = filtered_data.stack().max()
    min_value = filtered_data.stack().min()
    print('最大值', max_value)
    print('最小值', min_value)

    # 定义最小值和最大值
    a = 0.1
    b = 0.5

    # 最大最小归一化
    normalized_data = (filtered_data - min_value) / (max_value - min_value) * (b - a) + a
    # 使用combine_first()将归一化后的数据填充到Adj中的相应位置,而不影响其他位置的值
    Adj.loc[filtered_data.index] = normalized_data.combine_first(Adj)
    return Adj


communities_dis = greedy_modularity(G_dis, 1.12)
print(len(communities_dis))
communities_met = greedy_modularity(G_met, 1.5)
print(len(communities_met))
A_new = Relationship_Complete(communities_met, communities_dis)
Adj = normalized(A_new)

Adj.to_csv('output/A_new_greedy.csv', index=False)
non_zero_count = np.count_nonzero(Adj)
count_ones = np.count_nonzero(Adj == 1)
print("非零元素个数：", non_zero_count)
print("元素为1的个数：", count_ones)
# 输出归一化后的DataFrame
print(Adj)


