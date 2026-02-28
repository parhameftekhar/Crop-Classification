import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model import create_coo_sparse_matrix
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from sklearn.metrics import f1_score, confusion_matrix
import torch
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

TARGET_CROP = 1

def setup_data_loader():
    # Setup validation loader (same as in validate_checkpoint.py)
    val_loader = setup_training_loader(
        path_to_train_data='./training_data/test_patches.npy',
        unchanged_crops=[1, 5, 23, 176],
        target_crops=[TARGET_CROP],
        train_batch_size=1,
        crop_band_index=18,
        device='cpu',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )
    return val_loader

def validate_oracle_performance(val_loader, order, edges, edge_i, edge_j):
    """
    Oracle validation: Uses Ground Truth labels to build the Signed Laplacian matrix.
    Skips feature extraction and distance calculations.
    """
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    # Initialize overall confusion matrix
    overall_confusion = np.zeros((2, 2))

    # We iterate through the loader, but we only need the labels
    for _, labels in tqdm(val_loader, desc="Oracle Validation"):
        labels = labels.cpu().numpy()
        
        # Process each 112x112 quadrant
        for i in range(2):
            for j in range(2):
                # Extract the 112x112 quadrant
                start_h = i * 112
                start_w = j * 112
                quadrant_label = labels[0, start_h:start_h+112, start_w:start_w+112]
                
                # Reshape and reorder according to the sparse structure
                y = quadrant_label.reshape(-1)[order]
                
                # Define Oracle weights from Ground Truth: 
                # W_ij = 1 if y_i == y_j, else -1
                # This can be computed as y_i * y_j
                y_i, y_j = y[edge_i], y[edge_j]
                oracle_weights = y_i * y_j
                
                # Create sparse matrix and compute Laplacian matching validate_checkpoint.py structure
                coo_mat = create_coo_sparse_matrix(edges, oracle_weights)
                sparse_adjacency = coo_mat + coo_mat.T
                
                # Use signed sum for degree as in validate_checkpoint.py
                degree = sparse_adjacency.sum(axis=1).A1
                D = diags(degree)
                L = D - sparse_adjacency
                
                # Compute eigenvector (smallest algebraic eigenvalue)
                try:
                    _, eigen_vector = eigsh(L, k=1, which='SA', tol=1e-7)
                    pred_v = eigen_vector.flatten()
                    pred = np.sign(pred_v)
                except Exception as e:
                    print(f"Eigsh failed for quadrant ({i},{j}): {e}")
                    continue
                
                # Sign correction: check which sign gives better agreement with ground truth
                # ground truth y is in {-1, 1}
                acc_positive = np.mean(pred == y)
                acc_negative = np.mean(-pred == y)
                
                if acc_negative > acc_positive:
                    pred = -pred
                    
                # Convert predictions and labels to binary (0 and 1) for metrics
                y_binary = (y == 1).astype(np.int32)
                pred_binary = (pred == 1).astype(np.int32)
                
                # Compute confusion matrix for this quadrant
                quadrant_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
                overall_confusion += quadrant_confusion

                accuracy = np.mean(y == pred)
                f1 = f1_score(y, pred, pos_label=1, zero_division=1)
                
                valid_accuracy_list.append(accuracy)
                valid_f1_score_list.append(f1)

                # print(f"Quadrant ({i},{j}) accuracy: {accuracy:.4f} f1_score: {f1:.4f}")
    
    # Calculate metrics from overall confusion matrix
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\n--- Oracle Best-Case Performance (Crop {TARGET_CROP}) ---")
    print(f"Confusion Matrix:\n{overall_confusion}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"------------------------------------------------------")
    
    return accuracy, f1

def main():
    # Parameters
    img_height = img_width = 112
    window_size = 30
    device = 'cpu'
    
    # Setup sparse structure
    print("Setting up sparse structure...")
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges'].cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    # Load data
    val_loader = setup_data_loader()
    
    # Perform Oracle validation
    validate_oracle_performance(val_loader, order, edges, edge_i, edge_j)

if __name__ == "__main__":
    main()
