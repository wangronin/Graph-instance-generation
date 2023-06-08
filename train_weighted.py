from __future__ import division, print_function

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# import pyro
# import pyro.distributions as dist
# import scipy.sparse as sp
import torch

# from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from models import GAE
from preprocessing import mask_test_edges, preprocess_graph, preprocess_graph_weighted

# from utils import dotdict, eval_gae, load_data, make_sparse, plot_results


def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=100, with_labels=True)
    plt.show()


# create the data set first
# generate an Erdos-Renyi graph with edge weights
epochs = 200
N = 100
n_nodes = 20
n_feature = 5
p = 8 / n_nodes
G = np.random.rand(N, n_nodes, n_nodes) < p
G = np.triu(G, 1)
G = 1.0 * (G + np.transpose(G, axes=(0, 2, 1)))
nonzero_idx = np.nonzero(G)
G[nonzero_idx] = np.random.rand(len(nonzero_idx[0]))
features = torch.ones(N, n_nodes, n_feature)
# signed normalized Laplacian
G_norm = G.copy()
for i in range(N):
    G_norm[i] = preprocess_graph(G[i]).toarray()

train_data = TensorDataset(features, torch.Tensor(G_norm), torch.Tensor(G))
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

model = GAE(n_feature=n_feature, n_hidden=5, n_latent=2)
# model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(x, adj_norm, adj):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, adj_norm)
    loss = model.reconstruction_loss(z, adj)
    loss.backward()
    optimizer.step()
    return float(loss)


# @torch.no_grad()
# def test(data):
#     model.eval()
#     z = model.encode(data.x, data.edge_index)
#     return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

n_step = len(train_loader)
for epoch in range(1, epochs + 1):
    loss = 0
    for batch_idx, samples in enumerate(train_loader):
        loss_ = train(*samples)
        loss += loss_
        # auc, ap = test(test_data)
        # print(f"Step {batch_idx + 1}/{n_step}: LOSS: {loss_:.4f}")
    print(f"Epoch: {epoch:03d}, LOSS: {loss / n_step:.4f}")
    # print(f"Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}")

# F, adj = train_data[0]
# show_graph_with_labels(adj)

# adj_ = model.decode(model.encode(F, adj))
# show_graph_with_labels(adj_)
