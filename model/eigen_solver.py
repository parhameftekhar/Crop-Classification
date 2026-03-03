import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh
import numpy as np

class EigenSolver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def solve(self, L, device='cuda'):
        raise NotImplementedError

class PowerMethodEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_iter = cfg.get('max_iter', 100)

    def solve(self, L, device='cuda'):
        """
        Find the smallest eigenpair of symmetric matrix L using shifted power iteration.
        This implementation is fully differentiable.
        """
        n = L.shape[0]
        
        # Estimate U (upper bound on lambda_max)
        # Since L is a Laplacian, we can use the max row sum of abs values
        # For a sparse COO tensor L
        abs_L_values = torch.abs(L.values())
        row_indices = L.indices()[0]
        row_sums = torch.zeros(n, device=device)
        row_sums.index_add_(0, row_indices, abs_L_values)
        U = torch.max(row_sums)
        
        # Initialize random vector
        x = torch.randn(n, 1, device=device)
        x = x / torch.norm(x)
        
        for _ in range(self.max_iter):
            # y = (U*I - L)x = U*x - L@x
            # Sparse matrix multiplication in PyTorch
            Lx = torch.sparse.mm(L, x)
            y = U * x - Lx
            x = y / torch.norm(y)
        
        # Rayleigh quotient for the smallest eigenvalue of L
        Lx = torch.sparse.mm(L, x)
        lam = torch.mm(x.t(), Lx)
        
        return lam, x

class RayleighMinimizerEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.K = cfg.get('iterations', 10)
        # Learnable step sizes for each iteration.
        # Initialized such that softplus(param) is approximately 0.1
        # log(exp(0.1) - 1) approx -2.25
        self.raw_step_sizes = nn.Parameter(torch.ones(self.K) * -2.25)

    def solve(self, L, device='cuda'):
        """
        Find the smallest eigenpair by minimizing the Rayleigh quotient 
        using gradient descent with learnable step sizes.
        This implementation is fully differentiable.
        """
        n = L.shape[0]
        
        # Ensure step sizes are positive using softplus
        step_sizes = torch.nn.functional.softplus(self.raw_step_sizes)
        
        # Initialize random vector
        v = torch.randn(n, 1, device=device)
        v = v / torch.norm(v)

        for k in range(self.K):
            # Gradient of v^T L v is 2Lv
            # We absorb the factor of 2 into the learnable step_size
            Lv = torch.sparse.mm(L, v)
            v = v - step_sizes[k] * Lv
            # Projection back to the unit sphere
            v = v / torch.norm(v)
        
        # Rayleigh quotient for the smallest eigenvalue of L
        Lv = torch.sparse.mm(L, v)
        lam = torch.mm(v.t(), Lv)
        
        return lam, v

class LanczosEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tol = cfg.get('tol', 1e-7)

    def solve(self, L, device='cuda'):
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
