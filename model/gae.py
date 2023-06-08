import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        if self.bias is not None:
            self.bias.data.uniform_(-init_range, init_range)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class GCNEncoder(nn.Module):
    """Encoder using GCN layers"""

    def __init__(self, n_feature: int, n_hidden: int, n_latent: int, dropout: float):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(n_feature, n_hidden)
        self.gc2_mu = GraphConvolution(n_hidden, n_latent)
        self.gc2_sig = GraphConvolution(n_hidden, n_latent)
        self.dropout = dropout

    def forward(self, x: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        log_sig = self.gc2_sig(x, adj)
        return mu, torch.exp(log_sig)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout: float):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.fudge = 1e-7

    def forward(self, z: Tensor) -> Tensor:
        z = F.dropout(z, self.dropout, training=self.training)
        # z = torch.nn.functional.normalize(z, dim=2)
        adj_prob = torch.matmul(z, z.transpose(1, 2))
        # adj = (self.sigmoid(adj_prob) + self.fudge) * (1 - 2 * self.fudge)
        adj_prob = F.sigmoid(adj_prob)
        # mask = torch.eye(z.shape[1]).repeat(z.shape[0], 1, 1).bool()
        # adj_prob[mask] = 0
        return adj_prob


class GAE(nn.Module):
    """Graph Auto Encoder"""

    def __init__(self, n_feature: int, n_hidden: int, n_latent: int, dropout: float = 0.0):
        super(GAE, self).__init__()
        self.input_dim = n_feature
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        # Layers
        self.dropout = dropout
        self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
        self.decoder = InnerProductDecoder(self.dropout)
        self.loss = nn.BCELoss(reduction="none")

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)[0]

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def forward(self, features, adjacency) -> Tensor:
        z = self.encode(features, adjacency)
        adjacency_ = self.decode(z)
        return adjacency_

    def reconstruction_loss(self, adj_: Tensor, adj: Tensor) -> Tensor:
        r"""

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        # adj_ = self.decoder(z)
        # A = torch.flatten(adj, start_dim=1)
        # A_ = torch.flatten(adj_, start_dim=1)
        # indices = torch.nonzero(A == 1, as_tuple=True)
        # indices_ = torch.nonzero(A == 0, as_tuple=True)
        # loss = self.loss(adj_, adj + torch.eye(adj.shape[1]))
        # weights = torch.exp(loss)
        # indices = torch.nonzero(torch.eye(adj_.shape[1]).repeat(adj_.shape[0], 1, 1) == 0, as_tuple=True)
        # return torch.sum(loss)
        # return self.loss(torch.concat([A[indices], A[indices_]]), torch.concat([A_[indices], A_[indices_]]))
        # fro_norm = torch.linalg.matrix_norm(adj - adj_)
        # return fro_norm.sum() / (adj.shape[0] ** 2)
        return torch.sum(torch.abs(adj + torch.eye(adj.shape[1]) - adj_) ** 2)
