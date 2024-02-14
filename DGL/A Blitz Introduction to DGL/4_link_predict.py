# Link Prediction using Graph Neural Networks

# Objective
# 1. Build a GNN-based link prediction model.
# 2. Train and evaluate t he model on a small DGL-provided dataset.

import itertools
import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score

# Applications
# 1. social recommendation
# 2. item recommendation
# 3. knowledge graph completion, etc.

# This problem: predict  whether a citation relationship, either citing or being cited, between two papers exists in a citation network.

# Link prediction - binary classification problem
# - Edges: positive examples
# - Sample a number of nonexistent edges as negative examples
# - Divide the positive examples and negative examples into a training set and a test set.
# - Evaluate the model with any binary classification metric such as Area Under the Curve (AUC)

# load the dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Prepare training and testing sets

# Split edge set for training and testing
u, v = g.edges()

eids = np.arange(g.num_edges())
eids = np.random.permutation(eids) # randomize the ids to make training and test set randomized

test_size = int(len(eids) * 0.1) # test set is 10%, training set is 90%
train_size = g.num_edges() - test_size

test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# create the adjacency matrix using the edge information (u,v)
adj = sp.coo_matrix((np.ones(len(u)),(u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes()) # remove the diagonal 1's as well
neg_u, neg_v = np.where(adj_neg != 0) # neg_u and neg_v are negative edges

# sample negative edges
neg_eids = np.random.choice(len(neg_u), g.num_edges())
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]]
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]]
)

# When training, remove the edges in the test set from the original graph
train_g = dgl.remove_edges(g, eids[:test_size])

# Define a GraphSAGE model

# 2. Create model
# Build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# The model then predicts the probability of existence of an edge by computing a score between the representations of both incident nodes 
# (e.g. an MLP or a dot product)

# Positive graph, negative graph, and apply edges
# Node classification -> compute node representations
# Link prediction -> compute representation of pairs of nodes
        
# Treat pairs of nodes as another graph, since graph edges can describe pairs of nodes.
# positive graph - graph consisting of positive examples
# negative graph - graph consisting of negative examples

# construct positive graph and negative graph
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes = g.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes = g.num_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes = g.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes = g.num_nodes())

# This way, we can use ```DGLGraph.apply_edges``` method, which conveniently computes new edge features based on the incident nodes' features and the original edge features (if applicable).
# DGL provides a set of optimized builtin functions to compute new edge features based on the original node/edge features. 
# For example, ```dgl.function.u_dot_v``` computes a dot product of the incident nodes' representations for each edge.
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:,0]
        

# You can also write your own function if it is complex. For instance, the following module produces a scalar score on each edge by concatenating the incident nodes's features and passing it to an MLP.
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.
        
        Parameters
        ----------
        edges :
            Has three memebers ``src``, ``dst`` and ``data``, each of which is a dictionary representing the featurs of the source nodes, the destination nodes, and the edges themselves.
        
        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}
        
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
        
# Training loop
# After you defined the node representation computation and the edge score computation, you can go ahead and define the overall model, loss function, and evaluation metric.
        
# The loss function is simply binary cross entropy loss.
# The evaluation metric is AUC.
        
model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
# You can replace DotPredictor with MLPPredictor.
# pred = MLPPredictor(16)
pred = DotPredictor()

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)

# 3. set up loss and optimizer
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.01
)

# 4. Training
all_logits = []
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata["feat"])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# 5. Check results

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))

