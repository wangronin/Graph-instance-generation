import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gae
# ------------------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae(edges_pos, edges_neg, emb, adj_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    emb = emb.data.numpy()
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []

    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []

    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    accuracy = accuracy_score((preds_all > 0.5).astype(float), labels_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return accuracy, roc_score, ap_score


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def plot_results(results, test_freq, path="results.png"):
    # Init
    plt.close("all")
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results["train_elbo"]))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results["train_elbo"])
    ax.set_ylabel("Loss (ELBO)")
    ax.set_title("Loss (ELBO)")
    ax.legend(["Train"], loc="upper right")

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results["accuracy_train"])
    ax.plot(x_axis_test, results["accuracy_test"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend(["Train", "Test"], loc="lower right")

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results["roc_train"])
    ax.plot(x_axis_test, results["roc_test"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC AUC")
    ax.set_title("ROC AUC")
    ax.legend(["Train", "Test"], loc="lower right")

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results["ap_train"])
    ax.plot(x_axis_test, results["ap_test"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Precision")
    ax.legend(["Train", "Test"], loc="lower right")

    # Save
    fig.tight_layout()
    fig.savefig(path)


def show_graph_with_labels(adjacency_matrix, ax, pos=None):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    if pos is None:
        pos = nx.spring_layout(gr)
    nx.draw(gr, ax=ax, pos=pos, node_size=130, with_labels=True)
    return pos
