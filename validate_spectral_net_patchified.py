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

def setup_data_loader(config):
    # Setup test/validation loader using the ORIGINAL 224x224 patches
    val_loader = setup_training_loader(
        path_to_train_data='./training_data/test_patches.npy',
        unchanged_crops=[1, 5, 23, 176],
        target_crops=[176],
        train_batch_size=1,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )
    return val_loader

def validate_spectral_net_patchified():
    config = load_config()
    TARGET_CROP = 176
    device = 'cuda'
    
    # Configuration for subpatching
    SUBPATCH_SIZE = 56 # We iterate over 224x224 in 56x56 chunks
    window_size = 30
    d_star = 1.0
    
    # Setup sparse structure for SUBPATCH_SIZE
    sparse_image_obj = create_sparse_structure_from_images(SUBPATCH_SIZE, SUBPATCH_SIZE, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    checkpoint_path = f'checkpoints/v2/crop{TARGET_CROP}_vs_all.pth'
    
    # Initialize the End-to-End model
    solver_cfg = {
        'type': 'rayleigh',
        'iterations': 100,
        'n_nodes': SUBPATCH_SIZE * SUBPATCH_SIZE
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
    
    val_loader = setup_data_loader(config)
    
    overall_confusion = np.zeros((2, 2))
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    with torch.no_grad():
        for bands, label in tqdm(val_loader, desc="Validation"):
            bands = bands.to(device)
            # bands shape: (1, 224, 224, 19), label shape: (1, 224, 224)
            _, H_full, W_full, _ = bands.shape
            
            # Manually iterate through non-overlapping subpatches
            for i in range(0, H_full, SUBPATCH_SIZE):
                for j in range(0, W_full, SUBPATCH_SIZE):
                    # Skip if the subpatch doesn't fit exactly
                    if i + SUBPATCH_SIZE > H_full or j + SUBPATCH_SIZE > W_full:
                        continue
                        
                    subpatch_bands = bands[:, i:i+SUBPATCH_SIZE, j:j+SUBPATCH_SIZE, :]
                    subpatch_labels = label.squeeze(0)[i:i+SUBPATCH_SIZE, j:j+SUBPATCH_SIZE].to(device)
                    
                    # Forward pass through the end-to-end model
                    # The model expects patch_size for graph construction
                    eigen_val, eigen_vector, L, features_flat = model(subpatch_bands)
                    
                    # Compute metrics
                    pred_eigen = eigen_vector.cpu().numpy().flatten()
                    pred = np.sign(pred_eigen)
                    
                    # Reorder labels to match the graph order (Morton order)
                    y = subpatch_labels.cpu().numpy().flatten()[order.cpu().numpy()]
                    
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
                    quadrant_confusion = confusion_matrix(y_binary, pred_binary, labels=[0, 1])
                    overall_confusion += quadrant_confusion
                    
                    valid_accuracy_list.append(current_acc)
                    valid_f1_score_list.append(f1_score(y_binary, pred_binary, zero_division=1.0))

    # Final overall metrics
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\nOverall Results from GraphSpectralNet (Patchified 224x224):")
    print(f"Subpatch Size: {SUBPATCH_SIZE}x{SUBPATCH_SIZE}")
    print(f"Confusion Matrix:\n{overall_confusion}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    validate_spectral_net_patchified()
