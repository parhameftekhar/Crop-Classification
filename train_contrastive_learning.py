from model import FeatureExtractor
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model import create_feature_pairs, modified_sigmoid, create_coo_sparse_matrix
import torch.optim as optim
import torch
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, confusion_matrix
from utils import correct_pred_sign
import logging
import os
from datetime import datetime
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Global configuration
TARGET_CROP = -1  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp and target crop
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_crop{TARGET_CROP}_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def setup_data_loaders():
    # Setup training loader
    train_loader = setup_training_loader(
        path_to_train_data='./training_data/train_patches.npy',
        unchanged_crops=UNCHANGED_CROPS,
        target_crops=[TARGET_CROP],
        train_batch_size=16,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )

    # Setup validation loader
    val_loader = setup_training_loader(
        path_to_train_data='./training_data/val_patches.npy',
        unchanged_crops=UNCHANGED_CROPS,
        target_crops=[TARGET_CROP],
        train_batch_size=1,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )

    return train_loader, val_loader

def setup_model(logger):
    features_extractor = FeatureExtractor(
        num_block=4,
        kernel_size=9,
        stride=1,
        padding=4,
        num_channel_in=18,
        num_channel_internal=36,
        num_channel_out=18,
        matrix_size=18,
        device='cuda'
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in features_extractor.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Training for target crop: {TARGET_CROP}")

    return features_extractor

def validate_model(features_extractor, val_loader, positive_center, negative_center, d_star, order, edges, edge_i, edge_j, logger):
    features_extractor.eval()
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    # Initialize overall confusion matrix
    overall_confusion = np.zeros((2, 2))

    with torch.no_grad():
        for bands, label in tqdm(val_loader, desc="Validation"):
            features = features_extractor(bands).squeeze(0)
            
            # Process each 112x112 quadrant
            for i in range(2):
                for j in range(2):
                    # Extract the 112x112 quadrant
                    start_h = i * 112
                    start_w = j * 112
                    quadrant_features = features[start_h:start_h+112, start_w:start_w+112, :]
                    quadrant_label = label.squeeze(0)[start_h:start_h+112, start_w:start_w+112]
                    
                    # Reshape and reorder
                    quadrant_features = quadrant_features.reshape(-1, quadrant_features.shape[-1])[order, :]
                    quadrant_label = quadrant_label.reshape(-1)[order]
                    
                    # Calculate distances and weights
                    features_i, features_j = quadrant_features[edge_i], quadrant_features[edge_j]
                    distances = ((features_i - features_j) ** 2).sum(dim=1)
                    weights = modified_sigmoid(distances, d_star, scale=1)
                    
                    # Create sparse matrix and compute Laplacian
                    coo_mat = create_coo_sparse_matrix(edges, weights.cpu().numpy())
                    sparse_adjacency = coo_mat + coo_mat.T
                    
                    degree = sparse_adjacency.sum(axis=1).A1
                    D = diags(degree)
                    L = D - sparse_adjacency
                    
                    # Compute eigenvector and prediction
                    _, eigen_vector = eigsh(L, k=1, which='SA', tol=1e-7)
                    pred = np.sign(eigen_vector).flatten()
                    sign_correct = correct_pred_sign(pred, quadrant_features, positive_center, negative_center)
                    pred = sign_correct * pred
                    y = quadrant_label.cpu().numpy()

                    # Convert predictions and labels to binary (0 and 1)
                    y_binary = (y == 1).astype(np.int32)
                    pred_binary = (pred == 1).astype(np.int32)
                    
                    # Compute confusion matrix for this quadrant
                    quadrant_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
                    overall_confusion += quadrant_confusion

                    valid_accuracy_list.append(np.sum(y == pred) / len(pred))
                    valid_f1_score_list.append(f1_score(y, pred, pos_label=1))

                    logger.info(f"Quadrant ({i},{j}) accuracy: {valid_accuracy_list[-1]} f1_score: {valid_f1_score_list[-1]}")
    
    # Calculate metrics from overall confusion matrix
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    logger.info(f"\nOverall Results from Confusion Matrix:")
    logger.info(f"Confusion Matrix:\n{overall_confusion}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, f1

def train_model(features_extractor, train_loader, val_loader, num_epochs=300):
    logger = setup_logging()
    
    optimizer = optim.Adam(features_extractor.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    
    d_star = 1.0
    margin = 5.0
    
    # Setup sparse structure
    img_height = img_width = 112
    window_size = 30
    device = 'cuda'
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edges = edges.cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    best_val_f1 = 0.0
    
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        features_extractor.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (bands, label) in enumerate(tqdm(train_loader, desc="Training")):
            # Extract features
            features = features_extractor(bands)
            features = features.reshape(-1, features.shape[-1])
            label = label.reshape(-1)
            
            # Create random pairs
            pairs, pair_labels = create_feature_pairs(
                features=features,
                labels=label,
                num_pairs=5000
            )

            positive_pairs = pair_labels == 1
            negative_pairs = pair_labels == -1
            
            distances = ((pairs[:, 0] - pairs[:, 1]) ** 2).sum(dim=1)
            weights = modified_sigmoid(distances, d_star=d_star)
            
            sum1 = ((2 - weights[positive_pairs]) * distances[positive_pairs]).mean()
            sum2 = ((2 + weights[negative_pairs]) * torch.clamp(margin - distances[negative_pairs], min=0)).mean()
            loss = sum1 + sum2
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(features_extractor.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Calculate feature centers after each epoch
        positive_center, negative_center = features_extractor.calculate_feature_centers(train_loader)
        
        # Validation phase (after epoch 10 and every 5 epochs)
        if (epoch + 1) >= 75 and (epoch + 1) % 10 == 0:
            val_accuracy, val_f1 = validate_model(
                features_extractor, val_loader, positive_center, negative_center,
                d_star, order, edges, edge_i, edge_j, logger
            )
            
            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                checkpoint_path = f'checkpoints/crop{TARGET_CROP}_vs_all.pth'
                torch.save(features_extractor, checkpoint_path)
                logger.info(f"New best model saved with validation F1 score: {best_val_f1:.4f}")
                logger.info(f"Model saved to: {checkpoint_path}")
        
        scheduler.step()

if __name__ == "__main__":
    logger = setup_logging()
    train_loader, val_loader = setup_data_loaders()
    features_extractor = setup_model(logger)
    train_model(features_extractor, train_loader, val_loader) 