import torch
from .base_loss import BaseSparseLoss

class NormalizedCorrelationLoss(BaseSparseLoss):
    def forward(self, preds, ground_truth):
        # preds: (batch_size, num_pixels) - already flattened and ordered
        # ground_truth: (batch_size, img_height, img_width)
        batch_size = ground_truth.shape[0]

        ground_truth = ground_truth.reshape(batch_size, -1)
        ground_truth = ground_truth[:, self.order]
        ground_truth = ground_truth.float()

        # Normalize along the pixel dimension
        # Using a small epsilon to avoid division by zero
        eps = 1e-8
        preds_norm = torch.norm(preds, p=2, dim=1, keepdim=True)
        gt_norm = torch.norm(ground_truth, p=2, dim=1, keepdim=True)
        
        preds_normalized = preds / (preds_norm + eps)
        gt_normalized = ground_truth / (gt_norm + eps)

        # Batch dot product
        dot_product = torch.sum(preds_normalized * gt_normalized, dim=1)
        
        loss = 1 - torch.abs(dot_product)
        
        return torch.mean(loss)
