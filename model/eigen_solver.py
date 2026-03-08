import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh
import numpy as np

class EigenSolver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def solve(self, L, v_init=None, device='cuda'):
        raise NotImplementedError

class PowerMethodEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_iter = cfg.get('max_iter', 100)
        self.n_nodes = cfg['n_nodes']  # Requirement
        self.register_buffer('v0', torch.randn(self.n_nodes, 1))

    def solve(self, L, v_init=None, device='cuda'):
        """
        Find the smallest eigenpair using shifted power iteration.
        """
        n_total = L.shape[0]
        B = n_total // self.n_nodes
        N = self.n_nodes
        
        # Initialize starting vector
        if v_init is not None:
            # v_init expected shape: (B, N) or (B*N, 1)
            x = v_init.reshape(n_total, 1)
        else:
            # Fallback to random initialization
            x = self.v0.repeat(B, 1)
        
        # Helper for per-batch normalization
        def normalize_v(v):
            v_reshaped = v.view(B, N)
            norms = torch.norm(v_reshaped, dim=1, keepdim=True) + 1e-9
            return (v_reshaped / norms).view(B * N, 1)

        x = normalize_v(x)
        
        for _ in range(self.max_iter):
            # y = U*x - L@x
            Lx = torch.sparse.mm(L, x)
            y = U * x - Lx
            x = normalize_v(y)
        
        # Rayleigh quotient per batch item
        Lx = torch.sparse.mm(L, x)
        # lam shape: (B, 1)
        x_reshaped = x.view(B, N)
        Lx_reshaped = Lx.view(B, N)
        lam = torch.sum(x_reshaped * Lx_reshaped, dim=1, keepdim=True)
        
        return lam, x

class RayleighMinimizerEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.K = cfg.get('iterations', 10)
        # Learnable step sizes for each iteration.
        # Initialized such that softplus(param) is approximately 0.1
        # log(exp(0.1) - 1) approx -2.25
        self.raw_step_sizes = nn.Parameter(torch.ones(self.K) * -2.25)
        
        self.n_nodes = cfg['n_nodes']  # Requirement
        self.register_buffer('v0', torch.randn(self.n_nodes, 1))

    def solve(self, L, v_init=None, device='cuda'):
        """
        Minimize Rayleigh quotient starting from v_init (if provided).
        """
        n_total = L.shape[0]
        B = n_total // self.n_nodes
        N = self.n_nodes
        
        # Ensure step sizes are positive
        step_sizes = torch.nn.functional.softplus(self.raw_step_sizes)
        
        # Initialize starting vector
        if v_init is not None:
            # v_init expected shape: (B, N) or (B*N, 1)
            v = v_init.reshape(n_total, 1)
        else:
            # Fallback to random initialization
            v = self.v0.repeat(B, 1)
        
        def normalize_v(vec):
            vec_reshaped = vec.view(B, N)
            norms = torch.norm(vec_reshaped, dim=1, keepdim=True) + 1e-9
            return (vec_reshaped / norms).view(B * N, 1)

        v = normalize_v(v)

        for k in range(self.K):
            Lv = torch.sparse.mm(L, v)
            v = v - step_sizes[k] * Lv
            v = normalize_v(v)
        
        # Rayleigh quotient per batch item
        Lv = torch.sparse.mm(L, v)
        v_reshaped = v.view(B, N)
        Lv_reshaped = Lv.view(B, N)
        lam = torch.sum(v_reshaped * Lv_reshaped, dim=1, keepdim=True)
        
        return lam, v

class LanczosEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tol = cfg.get('tol', 1e-7)

    def solve(self, L, v_init=None, device='cuda'):
        """
        Non-differentiable solver using scipy's eigsh (Lanczos).
        Useful for validation where gradients aren't needed.
        """
        # If L is a torch tensor, convert to scipy for eigsh
        if torch.is_tensor(L):
            L_cpu = L.detach().cpu().coalesce()
            indices = L_cpu.indices().numpy()
            values = L_cpu.values().numpy()
            from scipy.sparse import coo_matrix
            L_src = coo_matrix((values, (indices[0], indices[1])), shape=L.shape)
        else:
            L_src = L
            
        lam, x = eigsh(L_src, k=1, which='SA', tol=self.tol)
        return torch.tensor(lam, device=device), torch.tensor(x, device=device)

def build_eigen_solver(cfg):
    solver_type = cfg.get("type", "power")
    if solver_type == "lanczos":
        return LanczosEigen(cfg)
    elif solver_type == "power":
        return PowerMethodEigen(cfg)
    elif solver_type == "rayleigh":
        return RayleighMinimizerEigen(cfg)
    else:
        raise ValueError(f"Unknown eigen solver type: {solver_type}")
