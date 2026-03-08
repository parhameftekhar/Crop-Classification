import torch
from .base_loss import BaseSparseLoss

class SignedLaplacianLoss(BaseSparseLoss):
    def forward(self, preds, ground_truth):
        # preds: (batch_size, num_pixels) - already flattened and ordered
        # ground_truth: (batch_size, img_height, img_width)
        batch_size = ground_truth.shape[0]

        ground_truth = ground_truth.reshape(batch_size, -1)
        ground_truth = ground_truth[:, self.order]

        
        edges = self.edges
        edge_i = edges[:, 0]
        edge_j = edges[:, 1]

        ground_truth_i = ground_truth[:, edge_i]
        ground_truth_j = ground_truth[:, edge_j]

        W_ij = (ground_truth_i * ground_truth_j)
        
        # Formula: |W_ij| * (x_i - sign(W_ij) * x_j)**2
        # Since ground_truth is -1 or 1, |W_ij| is always 1 and sign(W_ij) is just W_ij
        abs_W_ij = torch.abs(W_ij)
        sign_W_ij = torch.sign(W_ij)

        preds_i = preds[:, edge_i]
        preds_j = preds[:, edge_j]

        # (x_i - sign(W_ij) * x_j)^2
        # This pushes xi and xj to same sign if W_ij > 0, and opposite sign if W_ij < 0
        term = (preds_i - sign_W_ij * preds_j) ** 2
        
        loss = torch.mean(abs_W_ij * term)
   
        return loss
