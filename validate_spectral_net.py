import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from model.graph_spectral_net import GraphSpectralNet
from data_manager import setup_training_loader, create_sparse_structure_from_images
import yaml
import os

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_path='config_finetune.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_data_loader(config, resolution=56, target_crops=[176], unchanged_crops=[1, 5, 23, 176], batch_size=1):
    # Setup test/validation loader using preprocessed subpatches
    val_loader = setup_training_loader(
        path_to_train_data=f'./training_data{resolution}/val_patches.npy',
        unchanged_crops=unchanged_crops,
        target_crops=target_crops,
        train_batch_size=batch_size,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0,
        max_ratio=1
    )
    return val_loader

def validate_spectral_net():
    config = load_config()
    TARGET_CROP = 176
    device = 'cuda'
    
    # Parameters for validation
    RESOLUTION = 112 # or 56
    BATCH_SIZE = 1 # Test higher batch size
    patch_h = patch_w = RESOLUTION
    window_size = 30
    d_star = 1.0
    
    # Setup sparse structure for specified resolution
    sparse_image_obj = create_sparse_structure_from_images(patch_h, patch_w, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    checkpoint_path = f'checkpoints/v2/crop{TARGET_CROP}_vs_all.pth'
    
    # Initialize the End-to-End model
    solver_cfg = {
        'type': 'rayleigh',
        'iterations': 200,
        'n_nodes': patch_h * patch_w
    }
    
    model = GraphSpectralNet(
        feature_extractor_checkpoint=checkpoint_path,
        solver_cfg=solver_cfg,
        order=order,
        edge_i=edge_i,
        edge_j=edge_j,
        d_star=d_star,
        device=device
    )
    model.eval()
    
    val_loader = setup_data_loader(
        config, 
        resolution=RESOLUTION, 
        target_crops=[TARGET_CROP],
        unchanged_crops=[1, 5, 23, 176],
        batch_size=BATCH_SIZE
    )
    
    overall_confusion = np.zeros((2, 2))
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    with torch.no_grad():
        for bands, labels in tqdm(val_loader, desc="Validation"):
            bands = bands.to(device)
            # labels shape: (B, H, W)
            labels = labels.to(device)
            
            # Forward pass through the end-to-end model (ignoring 5th return: init_guess)
            eigen_val, eigen_vector, L, features_flat, _ = model(bands)
            
            # eigen_vector shape: (B, N)
            B_curr = eigen_vector.shape[0]
            
            for b in range(B_curr):
                # Compute metrics for each batch item
                pred_eigen = eigen_vector[b].cpu().numpy().flatten()
                pred = np.sign(pred_eigen)
                
                # Reorder labels to match the graph order (Morton order)
                y = labels[b].cpu().numpy().flatten()[order.cpu().numpy()]
                
                # Oracle Sign Correction: Choose the sign that maximizes accuracy
                acc1 = np.sum(y == pred) / len(pred)
                acc2 = np.sum(y == -pred) / len(pred)
                
                if acc2 > acc1:
                    pred = -pred
                    current_acc = acc2
                else:
                    current_acc = acc1
                    
                # Convert to binary
                y_binary = (y == 1).astype(np.int32)
                pred_binary = (pred == 1).astype(np.int32)
                
                # Confusion Matrix
                subpatch_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
                overall_confusion += subpatch_confusion
                
                valid_accuracy_list.append(current_acc)
                valid_f1_score_list.append(f1_score(y_binary, pred_binary, zero_division=1.0))

    # Final overall metrics
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\nOverall Results from GraphSpectralNet:")
    print(f"Confusion Matrix:\n{overall_confusion}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    validate_spectral_net()
