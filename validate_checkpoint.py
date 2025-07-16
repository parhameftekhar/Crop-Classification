import numpy as np
from model import FeatureExtractor
from data_manager import setup_training_loader
from model import modified_sigmoid, create_coo_sparse_matrix
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from sklearn.metrics import f1_score, confusion_matrix
import torch
from utils import correct_pred_sign

TARGET_CROP = 176

def setup_data_loader():
    # Setup validation loader
    val_loader = setup_training_loader(
        path_to_train_data='./training_data/val_patches.npy',
        unchanged_crops=[1, 5, 23, 176],
        target_crops=[TARGET_CROP],
        train_batch_size=1,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )
    return val_loader

def load_checkpoint(checkpoint_path):
    # Load the model from checkpoint
    features_extractor = torch.load(checkpoint_path)
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
                    pred = correct_pred_sign(pred, quadrant_features, positive_center, negative_center)
                    y = quadrant_label.cpu().numpy()

                    # Convert predictions and labels to binary (0 and 1)
                    y_binary = (y == 1).astype(int)
                    pred_binary = (pred == 1).astype(int)
                    
                    # Compute confusion matrix for this quadrant
                    quadrant_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
                    overall_confusion += quadrant_confusion

                    valid_accuracy_list.append(np.sum(y == pred) / len(pred))
                    valid_f1_score_list.append(f1_score(y, pred, pos_label=1))

                    print(f"Quadrant ({i},{j}) accuracy: {valid_accuracy_list[-1]} f1_score: {valid_f1_score_list[-1]}")
    
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
    
    checkpoint_path = f'checkpoints/v2/crop{TARGET_CROP}_vs_all.pth'
    d_star = 1.0
    
    # Setup sparse structure
    img_height = img_width = 112
    window_size = 30
    device = 'cuda'
    from data_manager import create_sparse_structure_from_images
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edges = edges.cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    # Load data and model
    val_loader = setup_data_loader()
    features_extractor = load_checkpoint(checkpoint_path)
    
    # Calculate feature centers
    positive_center, negative_center = features_extractor.calculate_feature_centers(val_loader)
    
    # Perform validation
    validate_model(features_extractor, val_loader, positive_center, negative_center, d_star, order, edges, edge_i, edge_j)

if __name__ == "__main__":
    main() 