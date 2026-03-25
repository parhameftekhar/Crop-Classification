import torch
import torch.nn.functional as F
from .base_loss import BaseSparseLoss

class MSELoss(BaseSparseLoss):
    """
    Standard MSE loss wrapper that handles the Morton/pixel ordering 
    required for Spectral Net predictions.
    """
    def forward(self, preds, ground_truth):
        # preds: (batch_size * num_pixels, 1) or (batch_size, num_pixels)
        # ground_truth: (batch_size, img_height, img_width)
        batch_size = ground_truth.shape[0]
        preds = preds.view(batch_size, -1)

        # Flatten ground truth and reorder to match the graph's pixel order (Morton)
        ground_truth_flat = ground_truth.reshape(batch_size, -1)
        ground_truth_reordered = ground_truth_flat[:, self.order]

        # SCALE NORMALIZATION:
        # Since 'preds' is a unit vector (norm=1) from the solver, 
        # we must normalize the ground truth to unit norm as well.
        # Otherwise, the scale difference (+-1 vs +-1/sqrt(N)) would dominate the loss.
        ground_truth_reordered = ground_truth_reordered / (torch.norm(ground_truth_reordered, dim=1, keepdim=True) + 1e-9)

        # Compute point-wise MSE
        return F.mse_loss(preds, ground_truth_reordered)
