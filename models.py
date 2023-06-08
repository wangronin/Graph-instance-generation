from typing import Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dist import WeightedBernoulli
from layers import GraphConvolution

pyro.enable_validation(True)


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
        self.sigmoid = nn.Tanh()
        self.fudge = 1e-7

    def forward(self, z: Tensor) -> Tensor:
        z = F.dropout(z, self.dropout, training=self.training)
        z_ = torch.nn.functional.normalize(z, dim=2)
        adj = torch.matmul(z, z.transpose(1, 2))
        # adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj


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

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)[0]

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def reconstruction_loss(self, z: Tensor, adj: Tensor) -> Tensor:
        r"""

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        adj_ = self.decoder(z)
        fro_norm = torch.linalg.matrix_norm(adj - adj_)
        return fro_norm.sum() / z.shape[0]

    # def test(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
    #     r"""Given latent variables :obj:`z`, positive edges
    #     :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
    #     computes area under the ROC curve (AUC) and average precision (AP)
    #     scores.

    #     Args:
    #         z (Tensor): The latent space :math:`\mathbf{Z}`.
    #         pos_edge_index (LongTensor): The positive edges to evaluate
    #             against.
    #         neg_edge_index (LongTensor): The negative edges to evaluate
    #             against.
    #     """
    #     from sklearn.metrics import average_precision_score, roc_auc_score

    #     pos_y = z.new_ones(pos_edge_index.size(1))
    #     neg_y = z.new_zeros(neg_edge_index.size(1))
    #     y = torch.cat([pos_y, neg_y], dim=0)

    #     pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
    #     neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
    #     pred = torch.cat([pos_pred, neg_pred], dim=0)

    #     y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    #     return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(nn.Module):
    """Variational Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308)"""

    def __init__(self, data, n_hidden, n_latent, dropout, subsampling=False):
        super(VGAE, self).__init__()

        # Data
        self.x = data["features"]
        self.adj_norm = data["adj_norm"]
        self.adj_labels = data["adj_labels"]
        self.obs = self.adj_labels.view(1, -1)

        # Dimensions
        N, D = data["features"].shape
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.n_subsample = 2 * self.n_edges
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)
        self.subsampling = subsampling

        # Layers
        self.dropout = dropout
        self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
        self.decoder = InnerProductDecoder(self.dropout)

    def model(self):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # Setup hyperparameters for prior p(z)
        z_mu = torch.zeros([self.n_samples, self.n_latent])
        z_sigma = torch.ones([self.n_samples, self.n_latent])

        # sample from prior
        z = pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))

        # decode the latent code z
        z_adj = self.decoder(z).view(1, -1)

        # Score against data
        pyro.sample("obs", WeightedBernoulli(z_adj, weight=self.pos_weight).to_event(2), obs=self.obs)

    def guide(self):
        # register PyTorch model 'encoder' w/ pyro
        pyro.module("encoder", self.encoder)

        # Use the encoder to get the parameters use to define q(z|x)
        z_mu, z_sigma = self.encoder(self.x, self.adj_norm)

        # Sample the latent code z
        pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))

    def get_embeddings(self):
        z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
        # Put encoder back into training mode
        self.encoder.train()
        return z_mu

    def generate(self):
        z_mu, z_sigma = self.encoder.eval()(self.x, self.adj_norm)
        z = pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))
        z_adj = self.decoder(z).view(1, -1)
        res = pyro.sample("obs", WeightedBernoulli(z_adj, weight=self.pos_weight).to_event(2), obs=self.obs)
        graph = res.view(self.n_samples, -1).cpu().detach().numpy()
        return graph - np.eye(self.n_samples)
