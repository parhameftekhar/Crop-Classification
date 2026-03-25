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

class SimpleLanczosEigen(EigenSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_nodes = cfg['n_nodes']
        self.m = cfg.get('iterations', cfg.get('m', 20))
        self.tol = cfg.get('tol', 1e-7)
        self.register_buffer('v0', torch.randn(self.n_nodes, 1))
        
        # Sub-solver for the small tridiagonal problem (subspace dimension m)
        sub_cfg = {
            'n_nodes': self.m,
            'iterations': cfg.get('rayleigh_iterations', cfg.get('inner_iterations', 100))
        }
        self.sub_solver = RayleighMinimizerEigen(sub_cfg)

    def solve(self, L, v_init=None, device='cuda'):
        """
        Differentiable batched Lanczos implementation.
        Uses an internal RayleighMinimizerEigen to solve the tridiagonal sub-problem.
        """
        n_total = L.shape[0]
        B = n_total // self.n_nodes
        N = self.n_nodes
        m = self.m

        # 1. --- Initialize ---
        if v_init is not None:
            v = v_init.reshape(B, N)
        else:
            v = self.v0.repeat(B, 1).reshape(B, N)
        
        v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
        
        V = [v]
        alpha = []
        beta = []
        
        # 2. --- Lanczos iterations ---
        for j in range(m):
            v_giant = V[j].reshape(B * N, 1)
            w_giant = torch.sparse.mm(L, v_giant)
            w = w_giant.reshape(B, N)
            
            a = (V[j] * w).sum(dim=1, keepdim=True)
            alpha.append(a)
            
            # Standard Lanczos step (recurrence)
            w = w - a * V[j]
            if j > 0:
                w = w - beta[j-1] * V[j-1]
            
            # --- FULL RE-ORTHOGONALIZATION (Gram-Schmidt) ---
            # Corrects loss of orthogonality in finite-precision and prevents ill-conditioning of T
            if j > 0:
                V_basis_j = torch.stack(V, dim=2) # (B, N, j+1)
                # Compute projections of w onto all previous vectors: h = V^T * w
                h = torch.bmm(V_basis_j.transpose(1, 2), w.unsqueeze(-1)) # (B, j+1, 1)
                # Subtract projections: w = w - V * h
                w = w - torch.bmm(V_basis_j, h).squeeze(-1)
                
            b = w.norm(dim=1, keepdim=True)
            beta.append(b)
            
            if j < m - 1:
                V.append(w / (b + 1e-9))

        # 3. --- Assemble batched tridiagonal matrix T as a Giant Sparse Matrix ---
        alpha_stack = torch.stack(alpha, dim=1).squeeze(-1) # (B, m)
        beta_stack = torch.stack(beta[:-1], dim=1).squeeze(-1) # (B, m-1)
        
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * m
        
        # Diag indices
        global_diag = (torch.arange(m, device=device).unsqueeze(0) + batch_offsets).flatten()
        idx_diag = torch.stack([global_diag, global_diag], dim=0)
        
        # Off-diag indices
        i_off = torch.arange(m - 1, device=device)
        global_i = (i_off.unsqueeze(0) + batch_offsets).flatten()
        global_j = ((i_off + 1).unsqueeze(0) + batch_offsets).flatten()
        
        idx_off = torch.cat([
            torch.stack([global_i, global_j], dim=0),
            torch.stack([global_j, global_i], dim=0)
        ], dim=1)
        
        T_indices = torch.cat([idx_diag, idx_off], dim=1)
        T_values = torch.cat([alpha_stack.flatten(), beta_stack.flatten(), beta_stack.flatten()], dim=0)
        
        T_sparse = torch.sparse_coo_tensor(T_indices, T_values, (B*m, B*m), device=device).coalesce()
            
        # 4. --- Solve small eigenproblem on T using the sub-solver ---
        # Project our initial guess v into the Lanczos subspace to get a hot start y0
        V_basis = torch.stack(V, dim=2) # (B, N, m)
        y0 = torch.bmm(V_basis.transpose(1, 2), v.unsqueeze(-1)) # (B, m, 1)
        y0 = y0 / (y0.norm(dim=1, keepdim=True) + 1e-9)
        y0_giant = y0.reshape(B * m, 1)
        
        # The sub_solver returns lam (B, 1) and y_global (B*m, 1)
        lam, y_global = self.sub_solver.solve(T_sparse, v_init=y0_giant, device=device)
        y = y_global.reshape(B, m, 1)
        
        # 5. --- Map back to original space: v_approx = V_basis @ y ---
        v_approx = torch.bmm(V_basis, y) # (B, N, 1)
        
        return lam, v_approx.reshape(B * N, 1)

def build_eigen_solver(cfg):
    solver_type = cfg.get("type", "power")
    if solver_type == "lanczos":
        return LanczosEigen(cfg)
    elif solver_type == "simple_lanczos":
        return SimpleLanczosEigen(cfg)
    elif solver_type == "power":
        return PowerMethodEigen(cfg)
    elif solver_type == "rayleigh":
        return RayleighMinimizerEigen(cfg)
    else:
        raise ValueError(f"Unknown eigen solver type: {solver_type}")
