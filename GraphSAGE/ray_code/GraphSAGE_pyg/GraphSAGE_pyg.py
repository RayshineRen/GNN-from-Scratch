import os.path as osp
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Planetoid(
    root="/home/ray/code/python/python_data_course/机器学习与深度学习导论/data", name="Cora"
)
data = dataset[0]
num_nodes_list = torch.arange(data.num_nodes)
train_idx = num_nodes_list[data["train_mask"]]
# print(data)

train_loader = NeighborSampler(
    data.edge_index,
    node_idx=train_idx,
    sizes=[15, 10, 5],
    batch_size=256,
    shuffle=True,
    num_workers=12,
)

subgraph_loader = NeighborSampler(
    data.edge_index,
    node_idx=None,
    sizes=[-1],
    batch_size=256,
    shuffle=False,
    num_workers=12,
)

# print(train_loader)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            # 对每一层的bipartite图都有x_target = x[:size[1]]
            x_target = x[
                : size[1]
            ]  # Target nodes are always placed first.目标节点放在最前面，一共有size[1]个目标节点
            # 实现了对一层bipartite图的卷积。可以把卷积就理解为聚合操作，这里就是逐层聚合，从第L层到第1层
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:  # 不是最后一层就执行下面的操作
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description("Evaluating")

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):  # 一共有l层
            xs = []
            # 一个batchsize中的目标节点采样L=1层涉及到的所有节点
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


model = SAGE(dataset.num_features, 64, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)  # x[n_id]这个batchsize中的目标节点采样L层涉及到的所有节点
        loss = criterion(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)
    correct = (y_pred == y_true).sum().item()
    test_acc = correct / data.num_nodes
    return test_acc


test_accs = []
for run in range(1, 11):
    print("")
    print(f"Run {run:02d}:")
    print("")

    model.reset_parameters()

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 51):
        loss, acc = train(epoch)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")

    test_acc = test()
    print(f"Test: {test_acc:.4f}")
    test_accs.append(test_acc)

test_acc = torch.tensor(test_accs)
print("============================")
print(f"Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}")
