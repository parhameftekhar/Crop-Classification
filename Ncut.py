from model import FeatureExtractor
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model import modified_sigmoid, create_coo_sparse_matrix
import torch
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
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
TARGET_CROP = 23  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Setup logging
def setup_logging():
    # Create logs/ncut directory if it doesn't exist
    if not os.path.exists('logs/ncut'):
        os.makedirs('logs/ncut')
    
    # Create a unique log file name with timestamp and target crop
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/ncut/ncut_crop{TARGET_CROP}_{timestamp}.log'
    
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

def setup_validation_loader():
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

    return val_loader

def setup_test_loader():
    # Setup test loader
    test_loader = setup_training_loader(
        path_to_train_data='./training_data/test_patches.npy',
        unchanged_crops=UNCHANGED_CROPS,
        target_crops=[TARGET_CROP],
        train_batch_size=1,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )

    return test_loader

def setup_training_loader_for_centers():
    # Setup training loader for feature centers calculation
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

    return train_loader

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
    logger.info(f"Validation for target crop: {TARGET_CROP}")

    return features_extractor

def validate_model(features_extractor, data_loader, positive_center, negative_center, d_star, order, edges, edge_i, edge_j, logger, dataset_name="Validation"):
    features_extractor.eval()
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    # Initialize overall confusion matrix
    overall_confusion = np.zeros((2, 2))

    with torch.no_grad():
        for bands, label in tqdm(data_loader, desc=f"{dataset_name}"):
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
                    weights = torch.exp(distances)
                    
                    # Create sparse matrix and compute Laplacian
                    coo_mat = create_coo_sparse_matrix(edges, weights.cpu().numpy())
                    sparse_adjacency = coo_mat + coo_mat.T
                    
                    # Compute degree matrix
                    degree = sparse_adjacency.sum(axis=1).A1
                    D = diags(degree)
                    
                    # Compute combinatorial Laplacian
                    L_combinatorial = D - sparse_adjacency
                    
                    # Compute normalized Laplacian: D^(-1/2) * L * D^(-1/2)
                    D_inv_sqrt = diags(1.0 / (degree ** 0.5))
                    laplacian = D_inv_sqrt @ L_combinatorial @ D_inv_sqrt
                    
                    # Compute second smallest eigenvector (k=2, which='SA' gives smallest algebraic)
                    _, eigen_vectors = eigsh(laplacian, k=2, which='SA', tol=1e-7)
                    # Use the second eigenvector (index 1)
                    eigen_vector = eigen_vectors[:, 1]
                    
                    # Apply D^(1/2) * eigenvector transformation for normalized cut
                    D_sqrt = diags(degree ** 0.5)
                    eigen_vector = D_sqrt @ eigen_vector
                    
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
    
    logger.info(f"\n{dataset_name} Results from Confusion Matrix:")
    logger.info(f"Confusion Matrix:\n{overall_confusion}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, f1

def run_validation():
    logger = setup_logging()
    train_loader = setup_training_loader_for_centers()  # Use training set for feature centers
    test_loader = setup_test_loader()
    features_extractor = setup_model(logger)
    
    # Load trained model if checkpoint exists
    checkpoint_path = f'checkpoints/v2/crop{TARGET_CROP}_vs_all.pth'
    if os.path.exists(checkpoint_path):
        features_extractor = torch.load(checkpoint_path, map_location='cuda')
        logger.info(f"Loaded trained model from: {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using untrained model.")
    
    # Setup sparse structure
    img_height = img_width = 112
    window_size = 30
    device = 'cuda'
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edges = edges.cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    # Calculate feature centers using training data
    positive_center, negative_center = features_extractor.calculate_feature_centers(train_loader)
    
    # Run test evaluation only
    logger.info("\n" + "=" * 50)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 50)
    
    d_star = 1.0
    test_accuracy, test_f1 = validate_model(
        features_extractor, test_loader, positive_center, negative_center,
        d_star, order, edges, edge_i, edge_j, logger, "Test"
    )
    
    logger.info(f"\nFinal Test Results:")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    
    # Add results summary to log file
    logger.info("\n" + "=" * 50)
    logger.info("FINAL TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Target Crop: {TARGET_CROP}")
    logger.info(f"Method: Normalized Cut")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_validation() 