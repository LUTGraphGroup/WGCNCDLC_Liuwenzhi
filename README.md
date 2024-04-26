# WGCNCDLC
Enhancing Disease-Related Metabolite Prediction with Weighted Graph Convolutional Networks Based on Community-Driven Link Completion

## 🏠 Overview
![image](https://github.com/LUTGraphGroup/WGCNCDLC_Liuwenzhi/assets/109469869/26127fb7-2ed2-483b-9713-12c44f72b946)


## 🛠️ Dependecies
- Python == 3.9
- pytorch == 1.12.1
- numpy == 1.22.4+mkl
- pandas == 1.4.4
- scikit-learn == 1.2.2


## 🗓️ Dataset
```
disease-metabolite associations: association_matrix.csv and disease-metabolite.xlsx
Disease similarity network: diease_network_simi.csv
Metabolite similarity network: metabolite_ntework_simi.csv
community complete result: A_new_greedy.csv
```

## 🛠️ Model options
```
--epochs           int     Number of training epochs.                 Default is 500.
--input dim        int     initial feature dimention.                 Default is 256.
--hidden dim       int     GCN Layer1 output dimention.               Default is 64.
--nclass dim       int     GCN Layer2 output dimention                Default is 512.
--dropout          float   Dropout rate                               Default is 0.1.
--lr               float   Learning rate                              Default is 0.005.
--wd               float   weight decay                               Default is 5e-4.

```

## 🎯 How to run?
```
1. The data folder stores associations and similarities. 
2、Run greedy_modularity_communities.py to get the completed association matrix.
3、Run train.py in the py_code folder to get the experimental results.

```
