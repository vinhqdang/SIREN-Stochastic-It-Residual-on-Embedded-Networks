import torch
import torch.nn as nn
import math

class DriftNetwork(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d)
        )
        
    def forward(self, x):
        return self.net(x)

class DiffusionNetwork(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        out_dim = d * (d + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )
        
        # Indices for lower triangular matrix
        self.tril_indices = torch.tril_indices(row=d, col=d, offset=0)
        
    def forward(self, x):
        out = self.net(x)  # shape: (B, d*(d+1)/2)
        B = x.shape[0]
        
        L = torch.zeros(B, self.d, self.d, device=x.device)
        L[:, self.tril_indices[0], self.tril_indices[1]] = out
        
        # For simulation, we only need L since L*L^T + eps*I gives the covariance
        # and L is the Cholesky factor. We can add eps to the diagonal of L.
        eps = 1e-4
        diag_idx = torch.arange(self.d)
        L[:, diag_idx, diag_idx] = L[:, diag_idx, diag_idx] + eps
        return L

class ResidualBlock(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.linear1 = nn.Linear(H, H)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(H, H)
        self.norm = nn.LayerNorm(H)
        
    def forward(self, x):
        h = self.linear1(x)
        h = self.gelu(h)
        h = self.linear2(h)
        return self.norm(x + h)

class ScoreNetwork(nn.Module):
    def __init__(self, d, H=256):
        super().__init__()
        # Input is x concatenated with graph context, so 2*d
        self.input_proj = nn.Sequential(
            nn.Linear(2 * d, H),
            nn.LayerNorm(H)
        )
        
        self.blocks = nn.ModuleList([ResidualBlock(H) for _ in range(4)])
        
        self.sigma_proj = nn.Linear(1, H)
        
        self.output_layer = nn.Linear(H, d)
        
    def forward(self, x, ctx, sigma):
        # x: (B, d), ctx: (B, d), sigma: (B, 1) or scalar
        if isinstance(sigma, float) or sigma.dim() == 0:
            sigma = torch.tensor([sigma], device=x.device).expand(x.shape[0], 1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)
            
        inp = torch.cat([x, ctx], dim=-1)
        h = self.input_proj(inp)
        
        # Add noise conditioning
        # Spec says h <- h + sigma_emb
        sigma_emb = self.sigma_proj(torch.log(sigma + 1e-8))
        h = h + sigma_emb
        
        for block in self.blocks:
            h = block(h)
            
        s = self.output_layer(h)
        # Noise-scaled output
        s = s / (sigma + 1e-8)
        return s

class SIRENModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drift = DriftNetwork(config.d)
        self.diffusion = DiffusionNetwork(config.d)
        self.score = ScoreNetwork(config.d, H=config.H)
