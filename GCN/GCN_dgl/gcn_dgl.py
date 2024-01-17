import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import CoraGraphDataset


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(GraphConv(in_feats, n_hidden, activation))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation))
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0 and i != self.n_layers:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        accuracy = torch.sum(indices == labels)
        return accuracy.item() * 1.0 / len(labels)


def train(
    n_epochs=100,
    lr=1e-2,
    weight_decay=5e-4,
    n_hidden=16,
    n_layers=2,
    activation=F.relu,
    dropout=0.5,
):
    data = CoraGraphDataset(raw_dir="/home/ray/.dgl/cora_v2")
    g = data[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    in_feats = features.shape[1]
    n_classes = data.num_labels

    model = GCN(g, in_feats, n_hidden, n_classes, n_layers, activation, dropout)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(model, features, labels, val_mask)
        print(
            "Epoch {} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc)
        )
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == "__main__":
    train()
