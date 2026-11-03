import torch
from .layers import build_nsf

class MarginalFlow(torch.nn.Module):
    def __init__(self, y_dim, hidden=1024, layers=12, bins=10, use_actnorm=True, use_inv1x1=True):
        super().__init__()
        self.flow = build_nsf(y_dim, hidden, layers, bins, use_actnorm, use_inv1x1)
    def log_prob(self, y):
        return self.flow.log_prob(y)
    def forward(self, y):
        z, logabsdet = self.flow._transform.forward(y)
        logpz = self.flow._distribution.log_prob(z)
        return z, logabsdet, logpz
