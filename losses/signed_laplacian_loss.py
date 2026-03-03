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

        ground_edge_weights = (ground_truth_i * ground_truth_j)

        preds_i = preds[:, edge_i]
        preds_j = preds[:, edge_j]

        preds_square_diff = (preds_i - preds_j) ** 2

        loss = torch.mean(ground_edge_weights * preds_square_diff)
   
        return loss
