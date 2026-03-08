import torch
import torch.nn as nn
from model.graph_learning import modified_sigmoid
from model.eigen_solver import build_eigen_solver
import os

class GraphSpectralNet(nn.Module):
    """
    End-to-end Spectral Graph Network for crop classification.
    Integrates feature extraction, graph construction, and differentiable eigen-solving.
    """
    def __init__(self, feature_extractor_checkpoint, solver_cfg, order=None, edge_i=None, edge_j=None, d_star=1.0, device='cuda'):
        super(GraphSpectralNet, self).__init__()
        self.device = device
        
        # Step 1: Initialize and load the Feature Extractor
        self.feature_extractor = self._load_feature_extractor(feature_extractor_checkpoint)
        
        # Step 2: Initialize the Eigen Solver
        self.solver = build_eigen_solver(solver_cfg)
        self.solver.to(self.device)
        
        # Graph structure parameters (optional, can be passed to forward)
        if order is not None:
            self.register_buffer('order', order)
        else:
            self.order = None
            
        if edge_i is not None:
            self.register_buffer('edge_i', edge_i)
        else:
            self.edge_i = None
            
        if edge_j is not None:
            self.register_buffer('edge_j', edge_j)
        else:
            self.edge_j = None
            
        self.d_star = d_star
        
    def _load_feature_extractor(self, checkpoint_path):
        """
        Loads the pre-trained FeatureExtractor model from a checkpoint.
        Uses weights_only=False because the checkpoints currently save the full object.
        """
        if os.path.exists(checkpoint_path):
            # Load the full model object
            model = torch.load(checkpoint_path, weights_only=False)
            model.to(self.device)
            print(f"GraphSpectralNet: Loaded feature extractor from {checkpoint_path}")
            return model
        else:
            raise FileNotFoundError(f"Feature extractor checkpoint not found at: {checkpoint_path}")

    def forward(self, x, order=None, edge_i=None, edge_j=None):
        """
        Forward pass through the pipeline:
        1. Feature extraction for the whole batch
        2. Batched Graph construction (Block-Diagonal Laplacian)
        3. Differentiable Eigen-solving across the batch
        """
        # x shape: (B, H, W, C)
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        B, H, W, C = features.shape
        num_nodes_per_patch = H * W
        
        # Use provided graph structure or falls back to buffers
        curr_order = order if order is not None else self.order
        curr_edge_i = edge_i if edge_i is not None else self.edge_i
        curr_edge_j = edge_j if edge_j is not None else self.edge_j
        
        if curr_order is None or curr_edge_i is None or curr_edge_j is None:
            raise ValueError("Graph structure (order, edge_i, edge_j) must be provided")
        
        # Flatten and reorder features for the whole batch (B, N, C)
        features_flat = features.reshape(B, num_nodes_per_patch, -1)[:, curr_order, :]
        
        # Calculate edge weights for the whole batch
        # features_flat_i shape: (B, E, C)
        features_flat_i = features_flat[:, curr_edge_i, :]
        features_flat_j = features_flat[:, curr_edge_j, :]
        
        # distances shape: (B, E)
        distances = ((features_flat_i - features_flat_j) ** 2).sum(dim=2)
        weights = modified_sigmoid(distances, self.d_star, scale=1.0)
        
        # Build batched degree matrix
        degree = torch.zeros(B, num_nodes_per_patch, device=self.device)
        degree.scatter_add_(1, curr_edge_i.expand(B, -1), weights)
        degree.scatter_add_(1, curr_edge_j.expand(B, -1), weights)
        
        # Construct Giant Block-Diagonal Laplacian
        # Node offsets for each batch item
        batch_offsets = torch.arange(B, device=self.device).unsqueeze(1) * num_nodes_per_patch
        
        # Shifted indices for the global sparse matrix
        global_edge_i = (curr_edge_i.unsqueeze(0) + batch_offsets).flatten()
        global_edge_j = (curr_edge_j.unsqueeze(0) + batch_offsets).flatten()
        global_diag = (torch.arange(num_nodes_per_patch, device=self.device).unsqueeze(0) + batch_offsets).flatten()
        
        # Giant Laplacian Indices
        diag_indices = torch.stack([global_diag, global_diag], dim=0)
        off_diag_indices = torch.cat([
            torch.stack([global_edge_i, global_edge_j], dim=0),
            torch.stack([global_edge_j, global_edge_i], dim=0)
        ], dim=1)
        
        L_indices = torch.cat([diag_indices, off_diag_indices], dim=1)
        
        # Giant Laplacian Values
        global_weights = weights.flatten()
        global_degree = degree.flatten()
        L_values = torch.cat([global_degree, -global_weights, -global_weights], dim=0)
        
        giant_n = B * num_nodes_per_patch
        L = torch.sparse_coo_tensor(L_indices, L_values, (giant_n, giant_n)).coalesce()
        
        # 3. Differentiable Eigen-solving
        # solve returns lam (B, 1) and vector (BN, 1)
        eigen_val, eigen_vector = self.solver.solve(L, device=self.device)
        
        # Reshape vector back to (B, N)
        eigen_vector = eigen_vector.view(B, num_nodes_per_patch)
        
        return eigen_val, eigen_vector, L, features_flat
