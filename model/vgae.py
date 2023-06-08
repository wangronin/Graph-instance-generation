from typing import Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from torch import Tensor

from .dist import WeightedBernoulli
from .gae import GCNEncoder, InnerProductDecoder

pyro.enable_validation(True)


class VGAE(nn.Module):
    """Variational Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308)"""

    def __init__(self, data: dict, n_hidden: int, n_latent: int, dropout: float, subsampling: bool = False):
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

    def get_embeddings(self) -> Tensor:
        z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
        # Put encoder back into training mode
        self.encoder.train()
        return z_mu

    def generate(self) -> Tensor:
        z_mu, z_sigma = self.encoder.eval()(self.x, self.adj_norm)
        z = pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))
        z_adj = self.decoder(z).view(1, -1)
        res = pyro.sample("obs", WeightedBernoulli(z_adj, weight=self.pos_weight).to_event(2), obs=self.obs)
        graph = res.view(self.n_samples, -1).cpu().detach().numpy()
        return graph - np.eye(self.n_samples)
