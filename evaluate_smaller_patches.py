from model import FeatureExtractor
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model import create_feature_pairs, modified_sigmoid, create_coo_sparse_matrix
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
PATCH_SIZE = 112  # New patch size

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/evaluation_{PATCH_SIZE}x{PATCH_SIZE}_crop{TARGET_CROP}_{timestamp}.log'
    
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

def evaluate_model(features_extractor, val_loader, positive_center, negative_center, d_star, order, edges, edge_i, edge_j, logger):
    features_extractor.eval()
    valid_accuracy_list = []
    valid_fake_accuracy_list = []
    
    # Initialize overall confusion matrix
    overall_confusion = np.zeros((2, 2))
    # Initialize perfect confusion matrix
    perfect_confusion = np.zeros((2, 2))

    with torch.no_grad():
        for bands, label in tqdm(val_loader, desc="Evaluation"):
            features = features_extractor(bands).squeeze(0)
            
            # Process each patch
            for i in range(2):
                for j in range(2):
                    # Extract the patch
                    start_h = i * PATCH_SIZE
                    start_w = j * PATCH_SIZE
                    patch_features = features[start_h:start_h+PATCH_SIZE, start_w:start_w+PATCH_SIZE, :]
                    patch_label = label.squeeze(0)[start_h:start_h+PATCH_SIZE, start_w:start_w+PATCH_SIZE]
                    
                    # Reshape and reorder
                    patch_features = patch_features.reshape(-1, patch_features.shape[-1])[order, :]
                    patch_label = patch_label.reshape(-1)[order]
                    
                    # Calculate distances and weights
                    features_i, features_j = patch_features[edge_i], patch_features[edge_j]
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
                    pred = correct_pred_sign(pred, patch_features, positive_center, negative_center)
                    y = patch_label.cpu().numpy()

                    # Convert predictions and labels to binary (0 and 1)
                    y_binary = (y == 1).astype(int)
                    pred_binary = (pred == 1).astype(int)
                    
                    # Check if patch has all same class
                    # unique_classes = np.unique(y_binary)
                    # if len(unique_classes) == 1:
                    #     logger.info(f"\nERROR - Patch ({i},{j}) has all same class: {unique_classes[0]}")
                    #     logger.info(f"Total samples in patch: {len(y_binary)}")
                    #     logger.info(f"Class distribution: {np.bincount(y_binary)}")
                    #     raise ValueError(f"Found homogeneous patch ({i},{j}) with all samples in class {unique_classes[0]}. This should not happen.")
                    
                    # Compute confusion matrix for this patch
                    patch_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
                    overall_confusion += patch_confusion

                    # Compute perfect confusion matrix for this patch
                    perfect_pred = y_binary  # Perfect prediction is same as ground truth
                    perfect_patch_confusion = confusion_matrix(y_binary, perfect_pred, labels=[0, 1])
                    
                    # Debug logging and raise exception for non-diagonal perfect confusion matrix
                    if perfect_patch_confusion[0, 1] != 0 or perfect_patch_confusion[1, 0] != 0:
                        logger.info(f"\nDEBUG - Non-diagonal perfect confusion matrix found in patch ({i},{j}):")
                        logger.info(f"y_binary unique values and counts: {np.unique(y_binary, return_counts=True)}")
                        logger.info(f"perfect_pred unique values and counts: {np.unique(perfect_pred, return_counts=True)}")
                        logger.info(f"Are y_binary and perfect_pred identical? {np.array_equal(y_binary, perfect_pred)}")
                        logger.info(f"Number of mismatches: {np.sum(y_binary != perfect_pred)}")
                        logger.info(f"Perfect patch confusion matrix:\n{perfect_patch_confusion}")
                        raise ValueError(f"Found non-diagonal perfect confusion matrix in patch ({i},{j}). This should not happen as perfect_pred is set to y_binary.")
                    
                    perfect_confusion += perfect_patch_confusion

                    valid_accuracy_list.append(np.sum(y == pred) / len(pred))
                    sign = 1 if np.sum(y == pred) > np.sum(y == -pred) else -1
                    valid_fake_accuracy_list.append(max(np.sum(y == pred), np.sum(y == -pred)) / len(pred))

                    # Log patch results including confusion matrix
                    logger.info(f"Patch ({i},{j}) results:")
                    logger.info(f"Accuracy: {valid_accuracy_list[-1]:.4f}")
                    logger.info(f"Fake Accuracy: {valid_fake_accuracy_list[-1]:.4f}")
                    logger.info(f"Confusion Matrix:\n{patch_confusion}")
                    logger.info(f"Perfect Confusion Matrix:\n{perfect_patch_confusion}")
    
    mean_accuracy = np.mean(valid_accuracy_list)
    mean_fake_accuracy = np.mean(valid_fake_accuracy_list)
    
    # Calculate metrics from overall confusion matrix
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate perfect metrics
    perfect_tn, perfect_fp, perfect_fn, perfect_tp = perfect_confusion.ravel()
    perfect_precision = perfect_tp / (perfect_tp + perfect_fp) if (perfect_tp + perfect_fp) > 0 else 0
    perfect_recall = perfect_tp / (perfect_tp + perfect_fn) if (perfect_tp + perfect_fn) > 0 else 0
    perfect_specificity = perfect_tn / (perfect_tn + perfect_fp) if (perfect_tn + perfect_fp) > 0 else 0
    perfect_f1 = 2 * (perfect_precision * perfect_recall) / (perfect_precision + perfect_recall) if (perfect_precision + perfect_recall) > 0 else 0
    
    # Log overall results including confusion matrix
    logger.info("\nOverall Results:")
    logger.info(f"Overall Confusion Matrix:\n{overall_confusion}")
    logger.info(f"Perfect Confusion Matrix:\n{perfect_confusion}")
    logger.info(f"Mean accuracy across all patches: {mean_accuracy:.4f}")
    logger.info(f"Mean fake accuracy across all patches: {mean_fake_accuracy:.4f}")
    logger.info(f"F1 score from overall confusion matrix: {f1:.4f}")
    logger.info(f"Perfect F1 score: {perfect_f1:.4f}")
    
    logger.info("\nDetailed Metrics from Confusion Matrix:")
    logger.info(f"True Positives: {tp} (Perfect: {perfect_tp})")
    logger.info(f"False Positives: {fp} (Perfect: {perfect_fp})")
    logger.info(f"True Negatives: {tn} (Perfect: {perfect_tn})")
    logger.info(f"False Negatives: {fn} (Perfect: {perfect_fn})")
    logger.info(f"Precision: {precision:.4f} (Perfect: {perfect_precision:.4f})")
    logger.info(f"Recall: {recall:.4f} (Perfect: {perfect_recall:.4f})")
    logger.info(f"Specificity: {specificity:.4f} (Perfect: {perfect_specificity:.4f})")
    
    return mean_accuracy, mean_fake_accuracy, f1, perfect_f1, overall_confusion, perfect_confusion

def main():
    logger = setup_logging()
    logger.info(f"Starting evaluation with {PATCH_SIZE}x{PATCH_SIZE} patches")
    
    # Load the model
    model_path = f'checkpoints/crop{TARGET_CROP}_vs_all.pth'
    features_extractor = torch.load(model_path)
    features_extractor.eval()
    logger.info(f"Model loaded from {model_path}")
    
    # Setup validation loader
    val_loader = setup_validation_loader()
    
    # Setup sparse structure
    d_star = 1.0
    window_size = 30
    device = 'cuda'
    sparse_image_obj = create_sparse_structure_from_images(PATCH_SIZE, PATCH_SIZE, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edges = edges.cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    # Calculate feature centers
    positive_center, negative_center = features_extractor.calculate_feature_centers(val_loader)
    
    # Evaluate model
    accuracy, fake_accuracy, f1, perfect_f1, confusion, perfect_confusion = evaluate_model(
        features_extractor, val_loader, positive_center, negative_center,
        d_star, order, edges, edge_i, edge_j, logger
    )
    
    logger.info(f"\nFinal Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Fake Accuracy: {fake_accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Perfect F1 Score: {perfect_f1:.4f}")
    logger.info(f"Final Confusion Matrix:\n{confusion}")
    logger.info(f"Perfect Confusion Matrix:\n{perfect_confusion}")

if __name__ == "__main__":
    main() 