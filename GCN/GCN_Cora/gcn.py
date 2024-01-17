import torch.nn as nn
import torch


# GCNConv 简单的实现方式
class GCNConv_s(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNConv_s, self).__init__()
        self.input_dim = input_dim  # 输入维度
        self.output_dim = output_dim  # 输出维度
        self.use_bias = use_bias  # 偏置
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))  # 初始权重
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))  # 偏置
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # 重新设置参数
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            # 偏置先全给0
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature, l=1):
        """
        :param adjacency: 邻接矩阵
        :param input_feature: 输入特征
        :param l: lambda 影响自环权重值
        :return:
        """
        # 公式: (D^-0.5) A' (D^-0.5) X W
        size = adjacency.shape[0]
        # X W
        support = torch.mm(input_feature, self.weight)
        # A' = A + \lambda I
        A = adjacency + l * torch.eye(size)
        # D: degree
        SUM = A.sum(dim=1)
        D = torch.diag_embed(SUM)
        # D'=D^(-0.5)
        D = D.__pow__(-0.5)
        # 让inf值变成0
        D[D == float("inf")] = 0
        # (D^-0.5) A' (D^-0.5)
        adjacency = torch.sparse.mm(D, adjacency)
        adjacency = torch.sparse.mm(adjacency, D)
        # (D^-0.5) A' (D^-0.5) X W
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            # 使用偏置
            output += self.bias
        return output

    def __repr__(self):
        # 打印的时候内存信息属性
        return (
            self.__class__.__name__
            + " ("
            + str(self.input_dim)
            + " -> "
            + str(self.output_dim)
            + ")"
        )
