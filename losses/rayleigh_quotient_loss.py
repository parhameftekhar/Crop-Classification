import torch
import torch.nn as nn

class RayleighQuotientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, eigen_vector, L):
        """
        eigen_vector: (B, N) - The current predicted eigenvectors
        L: (BN, BN) - The Giant Sparse Block-Diagonal Laplacian returned by the model
        """
        # Ensure eigen_vector is a flat column vector (BN, 1) to match Sparse matrix
        v_flat = eigen_vector.reshape(-1, 1)
        
        # 2. Compute Numerator: v^T L v
        # torch.sparse.mm(L, v) returns (BN, 1)
        # Using dot product between v_flat and (L @ v_flat) 
        Lv = torch.sparse.mm(L, v_flat)
        numerator = torch.sum(v_flat * Lv)
        
        # 3. Compute Denominator: v^T v
        denominator = torch.sum(v_flat ** 2) + 1e-9
        
        # 4. Rayleigh Quotient
        loss = numerator / denominator
   
        return loss
