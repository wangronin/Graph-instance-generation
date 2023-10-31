import argparse
import random
import sys
import time

import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "./")
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset

from model.gae import GAE
from utils import show_graph_with_labels

__author__ = "Hao Wang"

# TODO:
# 1. measure the empirical edge probability and compare the original graphs
# 2. visualize the latent space and check if we sample a set of latent points therein, what
# graph will be generated/decoded, and what is the empirical edge prob. we will get?
# 3. Now, if we encode a graph with different edge prob., then where it lands in the latent space?
# 4. Do we see any structures in the generated graphs?


def get_normalized_laplacian(A: np.ndarray) -> np.ndarray:
    adj_ = sp.coo_matrix(A + sp.eye(A.shape[0]))
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    normalized_laplacian = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return normalized_laplacian.toarray()


def get_ER_instances(
    N: int, p: float, n_features: int, n_instances: int = 1000
) -> Tuple[List[sp.csr_array], List[np.ndarray]]:
    adjacencies, features = [], []
    for _ in range(n_instances):
        adj = np.random.rand(N, N) < p
        adj = np.triu(adj, 1)
        adj = 1 * (adj + adj.T)
        row, col = np.nonzero(adj)
        adj = sp.csr_array((np.ones(len(row)), (row, col)), shape=(N, N))
        feature = np.random.randn(N, n_features)
        adjacencies += [adj]
        features += [feature]
    return adjacencies, features


class GraphDataset(Dataset):
    def __init__(self, adjacency, features, device):
        self.device = device
        # the normalized graph laplacian
        self._A = torch.tensor(np.array([A.toarray() for A in adjacency])).to(self.device)
        self._L = torch.tensor(np.array([get_normalized_laplacian(A) for A in adjacency])).to(self.device)
        # features of each node
        self._X = torch.tensor(np.array([F for F in features])).to(self.device)
        self._N = len(self._L)

    def __len__(self):
        return self._N

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (self._X[idx], self._L[idx], self._A[idx])


def seed_everything(seed: int, cuda: bool = False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def train(
    model: nn.Module,
    data: DataLoader,
    optimizer,
    args,
) -> Tuple[float, Dict]:
    model.train()
    t = time.time()
    total_loss = 0
    N = len(data)
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, L, adj = batch
        adj_ = model(x.double(), L.double())
        loss = model.reconstruction_loss(adj_, adj)
        total_loss += float(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        # print(f"Batch {batch_num + 1}/{N} - loss: {loss:.5f} - {time.time() - t:.5f} seconds")
        t = time.time()
    return total_loss / N


def evaluate(
    model: nn.Module,
    data: DataLoader,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0
    N = len(data)
    with torch.no_grad():
        for _, batch in enumerate(data):
            x, L, adj = batch
            adj_ = model(x.double(), L.double())
            loss = model.reconstruction_loss(adj_, adj)
            total_loss += float(loss)
    return total_loss / N


def main():
    args = make_parser().parse_args()
    print("[Model hyperparams]: {}".format(str(args)))

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    print(f"running on device {device}")
    seed_everything(seed=42, cuda=cuda)

    # prepare the graph data set
    n_features = 20
    n_nodes = 10
    adjacencies, features = get_ER_instances(N=n_nodes, p=0.2, n_features=n_features, n_instances=100)
    dataset = GraphDataset(adjacencies, features, device)
    N = len(dataset)
    train_size = int(0.75 * N)
    train_idx = random.sample(range(N), train_size)
    test_idx = list(set(range(N)) - set(train_idx))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = GAE(n_feature=n_features, n_hidden=20, n_latent=20, dropout=args.dropout)
    model.to(device)
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=int(args.epochs / 10), gamma=0.8)

    try:
        best_valid_loss = None
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, args)
            test_loss = evaluate(model, test_dataloader)
            scheduler.step()
            print(f"epoch {epoch}/{args.epochs} - train loss: {train_loss:.5f} - test loss: {test_loss:.5f}")
            if not best_valid_loss or test_loss < best_valid_loss:
                best_valid_loss = test_loss
                # TODO: save the model here

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    # visualize the latent space
    data = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=False)
    adjacencies, features = get_ER_instances(N=n_nodes, p=0.5, n_features=n_features, n_instances=100)
    dataset2 = GraphDataset(adjacencies, features, device)
    N = len(data)
    embeddings = []
    with torch.no_grad():
        for k, batch in enumerate(data):
            x, L, adjacency = batch
            z = model.encode(x.double(), L.double())[0].numpy()
            z = np.mean(z, axis=1)
            embeddings.append(z)

    data2 = DataLoader(dataset2, batch_size=1, shuffle=False)
    with torch.no_grad():
        for k, batch in enumerate(data2):
            x, L, adjacency = batch
            z = model.encode(x.double(), L.double())[0].numpy()
            z = np.mean(z, axis=1)
            embeddings.append(z)

    embeddings = np.vstack(embeddings)

    reducer = umap.UMAP()
    embeddings_scaled = StandardScaler().fit_transform(embeddings)
    embeddings = reducer.fit_transform(embeddings_scaled)
    fig, ax = plt.subplots(1, 1, sharey=True)
    line = []
    line += ax.plot(embeddings[:N, 0], embeddings[:N, 1], "k.")
    line += ax.plot(embeddings[N:, 0], embeddings[N:, 1], "r.")
    ax.set_xlabel("feature 1")
    ax.set_xlabel("feature 2")
    ax.legend(line, ["p=0.2", "p=0.6"])
    plt.savefig(f"figures/latent_space.pdf")

    # plot the reconstructed graphs
    edge_prob_pred = []
    edge_prob_input = []
    data = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=False)
    with torch.no_grad():
        for k, batch in enumerate(data):
            x, L, adjacency = batch
            adjacency = adjacency[0].numpy()
            prob_matrix = model(x.double(), L.double())[0].numpy()
            np.fill_diagonal(prob_matrix, 0)
            edge_prob_input.append(np.mean(adjacency))
            edge_prob_pred.append(np.mean(prob_matrix))
            # remove the self-loops
            adjacency_pred = prob_matrix > 0.5
            # compute the reconstruction loss
            print(np.mean(np.abs(adjacency - adjacency_pred)))
            # plot_reconstructed_graph(prob_matrix, adjacency, k)
            plot_probability_matrix(prob_matrix, adjacency, k)

    # plot the empirical edge probility of the reconstructed graph
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Edge prob.")
    data = pd.DataFrame(
        data={
            "kind": ["reconstructed"] * len(edge_prob_pred) + ["input"] * len(edge_prob_input),
            "density": np.array(edge_prob_pred + edge_prob_input),
        }
    )
    sns.violinplot(data=data, x="kind", y="density", ax=ax)
    plt.savefig(f"figures/estimated_edge_prob.pdf")
    plt.close(fig)


def plot_reconstructed_graph(adjacency_pred, adjacency, figure_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    pos = show_graph_with_labels(adjacency_pred, ax1)
    show_graph_with_labels(adjacency, ax2, pos)
    ax1.set_title("Reconstructed")
    ax2.set_title("Input graph")
    plt.savefig(f"reconstruction{figure_name}.pdf")
    plt.close(fig)


def plot_probability_matrix(probability_matrix, adjacency, figure_name):
    np.fill_diagonal(probability_matrix, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 6), subplot_kw={"aspect": "equal"})
    sns.heatmap(probability_matrix, vmin=0, vmax=1, cmap="Blues", ax=ax1)
    sns.heatmap(adjacency, vmin=0, vmax=1, cmap="Blues", ax=ax2)
    ax1.set_title("predicted edge prob.")
    ax2.set_title("target adjacency matrix")
    plt.savefig(f"figures/heatmap{figure_name}.pdf")
    plt.close(fig)


def make_parser():
    parser = argparse.ArgumentParser(description="PyTorch GCN Autoencoder")
    parser.add_argument("--hidden", type=int, default=300, help="number of hidden units for the RNN encoder")
    parser.add_argument("--n_layers", type=int, default=4, help="number of layers of the RNN encoder")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=1, metavar="N", help="batch size")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--cuda", action="store_true", help="[USE] CUDA")
    return parser


if __name__ == "__main__":
    main()
