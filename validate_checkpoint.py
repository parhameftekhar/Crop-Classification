import numpy as np
from model.graph_learning import FeatureExtractor
from data_manager import setup_training_loader
from model.graph_learning import modified_sigmoid, create_coo_sparse_matrix, smallest_eigenpair_via_shifted_power
from tqdm import tqdm
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.metrics import f1_score, confusion_matrix
import torch
from utils import correct_pred_sign
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

TARGET_CROP = 176

def setup_data_loader(resolution=56):
    # Setup test loader using the preprocessed subpatch dataset with filtering
    from data_manager import setup_training_loader
    val_loader = setup_training_loader(
        path_to_train_data=f'./training_data{resolution}/test_patches.npy',
        unchanged_crops=[1, 5, 23, 176],
        target_crops=[TARGET_CROP],
        train_batch_size=1,
        crop_band_index=18,
        device='cuda',
        min_ratio=0.1,
        max_ratio=0.9
    )
    return val_loader

def load_checkpoint(checkpoint_path):
    # Load the model from checkpoint
    features_extractor = torch.load(checkpoint_path, weights_only=False)
    features_extractor.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return features_extractor

def validate_model(features_extractor, val_loader, positive_center, negative_center, d_star, order, edges, edge_i, edge_j):
    features_extractor.eval()
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    # Initialize overall confusion matrix
    overall_confusion = np.zeros((2, 2))

    with torch.no_grad():
        for bands, label in tqdm(val_loader, desc="Validation"):
            # bands shape: (1, H, W, C), label shape: (1, H, W)
            features = features_extractor(bands).squeeze(0)
            subpatch_label = label.squeeze(0)
            
            # Reshape and reorder features and labels
            features_flat = features.reshape(-1, features.shape[-1])[order, :]
            label_flat = subpatch_label.reshape(-1)[order]
            
            # Calculate distances and weights
            features_i, features_j = features_flat[edge_i], features_flat[edge_j]
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
            y = label_flat.cpu().numpy()

            # Oracle Sign Correction: Choose the sign that maximizes accuracy
            acc1 = np.sum(y == pred) / len(pred)
            acc2 = np.sum(y == -pred) / len(pred)
            
            if acc2 > acc1:
                pred = -pred
                current_acc = acc2
            else:
                current_acc = acc1

            # Convert predictions and labels to binary (0 and 1)
            y_binary = (y == 1).astype(np.int32)
            pred_binary = (pred == 1).astype(np.int32)
            
            # Compute confusion matrix for this subpatch
            subpatch_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
            overall_confusion += subpatch_confusion

            valid_accuracy_list.append(current_acc)
            valid_f1_score_list.append(f1_score(y, pred, pos_label=1, zero_division=1.0))
    
    # Calculate metrics from overall confusion matrix
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\nOverall Results from Confusion Matrix:")
    print(f"Confusion Matrix:\n{overall_confusion}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy, f1

def main():
    # Parameters
    RESOLUTION = 32 # or 32
    checkpoint_path = f'checkpoints/v2/crop{TARGET_CROP}_vs_all.pth'
    d_star = 1.0
    
    # Setup sparse structure
    img_height = img_width = RESOLUTION
    window_size = 30
    device = 'cuda'
    from data_manager import create_sparse_structure_from_images
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edges = edges.cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    # Load data and model
    val_loader = setup_data_loader(resolution=RESOLUTION)
    features_extractor = load_checkpoint(checkpoint_path)
    
    # Calculate feature centers
    positive_center, negative_center = features_extractor.calculate_feature_centers(val_loader)
    
    # Perform validation
    validate_model(features_extractor, val_loader, positive_center, negative_center, d_star, order, edges, edge_i, edge_j)

if __name__ == "__main__":
    main() 