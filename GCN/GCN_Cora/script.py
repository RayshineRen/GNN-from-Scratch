from data import CoraData
import pprint
import numpy as np
import scipy.sparse as sp
import itertools

data = CoraData("/home/ray/code/python/GNN-from-Scratch/GCN/GCN_Cora/data/Cora").data
# pprint.pprint(data.adjacency_dict)
# print(data.adjacency_dict)
adj_dict = data.adjacency_dict
edge_index = []
num_nodes = len(adj_dict)
# print(num_nodes)
for src, dst in adj_dict.items():
    edge_index.extend([src, v] for v in dst)
    edge_index.extend([v, src] for v in dst)
l = itertools.groupby(sorted(edge_index))
edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
edge_index = np.asarray(edge_index)
# print(edge_index)
adj = sp.coo_matrix(
    (np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
    shape=(num_nodes, num_nodes),
    dtype="float32",
)
pprint.pprint(adj)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
pprint.pprint(adj)
adj = adj + sp.eye(adj.shape[0])
rowsum = np.array(adj.sum(1))
print(rowsum)
r_inv_sqrt = np.power(rowsum, -0.5).flatten()
print(r_inv_sqrt)
r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
adj = adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
print(adj)
