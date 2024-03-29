# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        # # 从节点的邻居中进行有放回地进行采样
        # res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        # results.append(res)
        if len(neighbor_table[sid]) >= sample_num:
            res = np.random.choice(
                neighbor_table[sid], size=(sample_num,), replace=False
            )
        else:
            res = np.random.choice(
                neighbor_table[sid], size=(sample_num,), replace=True
            )
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射
    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    # print("sampling result = ", sampling_result)
    # print("sample_nums = ", sample_nums)
    for k, hopk_num in enumerate(sample_nums):
        # print("sampling_result[k] = ", sampling_result[k])
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError(
                "Unknown aggr type, expected sum, max, or mean, but got {}".format(
                    self.aggr_method
                )
            )

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, aggr_method={}".format(
            self.input_dim, self.output_dim, self.aggr_method
        )


class SageGCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation=F.relu,
        aggr_neighbor_method="mean",
        aggr_hidden_method="sum",
    ):
        """SageGCN层定义

        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(
            input_dim, hidden_dim, aggr_method=aggr_neighbor_method
        )
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = (
            self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        )
        return "in_features={}, out_features={}, aggr_hidden_method={}".format(
            self.input_dim, output_dim, self.aggr_hidden_method
        )


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list

        for l in range(self.num_layers):
            # print(f"========= 第 {l} 层 =========")
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                # print("self.num_layers - l " , self.num_layers - l-1)
                # for hop in range(self.num_layers - l-1,l,-1):
                #     print(f"======== hop {hop} ============ " )
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                # print(" src_node_num = ", src_node_features.shape)

                neighbor_node_features = hidden[hop + 1].view(
                    (src_node_num, self.num_neighbors_list[hop], -1)
                )
                # print(" neighbor_node_features = ", neighbor_node_features.shape)

                h = gcn(src_node_features, neighbor_node_features)
                # print(" after gcn h = ", h.shape)

                next_hidden.append(h)
            hidden = next_hidden
            # print("hidden shape = ",len(hidden))
        return hidden[0]

    def extra_repr(self):
        return "in_features={}, num_neighbors_list={}".format(
            self.input_dim, self.num_neighbors_list
        )
