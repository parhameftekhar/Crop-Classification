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
        1. Feature extraction
        2. Graph construction (Weights -> Laplacian)
        3. Differentiable Eigen-solving
        """
        # x is assumed to be a single subpatch (H, W, C) or (1, H, W, C)
        if x.dim() == 4:
            features = self.feature_extractor(x).squeeze(0)
        else:
            features = self.feature_extractor(x.unsqueeze(0)).squeeze(0)
        
        # Use provided graph structure or falls back to buffers
        curr_order = order if order is not None else self.order
        curr_edge_i = edge_i if edge_i is not None else self.edge_i
        curr_edge_j = edge_j if edge_j is not None else self.edge_j
        
        if curr_order is None or curr_edge_i is None or curr_edge_j is None:
            raise ValueError("Graph structure (order, edge_i, edge_j) must be provided either in __init__ or forward")
        
        # Flatten and reorder features based on the sparse structure
        num_pixels = features.shape[0] * features.shape[1]
        features_flat = features.reshape(num_pixels, -1)[curr_order, :]
        
        # Calculate edge weights
        features_i = features_flat[curr_edge_i]
        features_j = features_flat[curr_edge_j]
        distances = ((features_i - features_j) ** 2).sum(dim=1)
        weights = modified_sigmoid(distances, self.d_star, scale=1.0)
        
        # Build the degree matrix (diagonal) for Signed Laplacian
        # D_ii = sum(|W_ij|) ensures the matrix is positive semi-definite and handles repulsion
        num_nodes = features_flat.shape[0]
        degree = torch.zeros(num_nodes, device=self.device)
        abs_weights = torch.abs(weights)
        degree.index_add_(0, curr_edge_i, abs_weights)
        degree.index_add_(0, curr_edge_j, abs_weights)
        
        # Build Sparse Laplacian matrix indices
        diag_indices = torch.stack([torch.arange(num_nodes, device=self.device)] * 2, dim=0)
        off_diag_indices = torch.cat([
            torch.stack([curr_edge_i, curr_edge_j], dim=0),
            torch.stack([curr_edge_j, curr_edge_i], dim=0)
        ], dim=1)
        
        L_indices = torch.cat([diag_indices, off_diag_indices], dim=1)
        # Build Sparse Laplacian matrix values: L = D - W
        L_values = torch.cat([degree, -weights, -weights], dim=0)
        
        L = torch.sparse_coo_tensor(L_indices, L_values, (num_nodes, num_nodes)).coalesce()
        
        # Step 3: Differentiable Eigen-solving
        # Find the smallest eigenpair of Laplacian L
        eigen_val, eigen_vector = self.solver.solve(L, device=self.device)
        
        return eigen_val, eigen_vector, L, features_flat
