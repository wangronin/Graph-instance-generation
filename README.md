# PyTorch Implementation of GAE

Graph Auto-Encoder in PyTorch

This is a PyTorch/Pyro implementation of the Variational Graph Auto-Encoder model described in the paper:

T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308).

## Requirements

Tested on:

- Python 3.11
- pyro-ppl==0.3.0
- torch==2.0.1
- networkx==3.1

## Training

To construct instances of the Erdős–Rényi model:

```bash
python train_gcn.py
```

<!-- ### Notes
- This implementation uses Pyro's blackbox SVI function with the default ELBO loss. This is slower than the TensorFlow implementation which uses a custom loss function with an analytic solution to the KL divergence term.
- Currently the code is not set up to use a GPU, but the code should be easy to extend to improve running speed -->
