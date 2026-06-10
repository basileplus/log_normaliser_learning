import torch
from torch import nn
import torch.nn.functional as F

class PositiveLinear(nn.Module):
    """Linear layer with W >= 0 for ICNN layers"""
    def __init__(self, in_dim, out_dim, eps=0.001):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1) 
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, eps=0.001):
        return x @ (F.softplus(self.W) + eps) +self.b # ICNN should use softplus, no clamping otherwise gradient is null if W<0. Classic issue


class ICNN(nn.Module):
    """
    Input-Convex Neural Network  f : R^n_in -> R^n_out.
    Recurrence: z_{t+1} = σ(W_t z_t + A_t θ),  W_t ≥ 0,  σ = softplus.
    A final PositiveLinear layer outputs a scalar.
    """
    def __init__(self, n_in, n_out, h=16, depth=4):
        super().__init__()
        self.Ws  = nn.ModuleList([
            PositiveLinear(h, h) for _ in range(depth - 1)
        ])
        self.As  = nn.ModuleList([
            nn.Linear(n_in, h) for _ in range(depth)
        ])
        self.outLayer = nn.Linear(h, n_out)

    def forward(self, theta):
        z = F.softplus(self.As[0](theta))
        for W, A in zip(self.Ws, self.As[1:]):
            z = F.softplus(W(z) + A(theta))
        return self.outLayer(z)