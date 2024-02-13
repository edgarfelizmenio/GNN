# Tutorial URL
# https://docs.dgl.ai/tutorials/blitz/1_introduction.html#sphx-glr-tutorials-blitz-1-introduction-py

import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Loading Cora Dataset
dataset = dgl.data.CoraGraphDataset()

print(f"Number of categories: {dataset.num_classes}")

# the single graph from the Cora dataset
g = dataset[0]

# g.ndata = node features
# g.edata = edge features

# ndata
# - train_mask: indicates if node is in the training set
# - val_mask: indicates if node is in the validation set
# - test_mask: indicates if node is in the test set
# - label: ground truth
# - feat: node features
print("Node features")
print(g.ndata)
print(g.ndata["feat"].shape)
print("Edge features")
print(g.edata)

# defining a GCN
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Training the GCN
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # train for 100 epochs
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and  the corresponding test accuracy
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss: .3f}, val acc: {val_acc:.3f} (best {best_test_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )

# Create the model with given dimensions
model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
# train the model
train(g, model)


# Training to GPU
print("\nTraining to GPU")
g = g.to('cuda')
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
train(g, model)