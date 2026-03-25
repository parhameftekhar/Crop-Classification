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

def setup_data_loader(config, resolution=32, target_crops=[176], unchanged_crops=[1, 5, 23, 176], batch_size=1):
    # Setup test/validation loader using preprocessed subpatches
    path = f'./training_data{resolution}/val_patches.npy'
    
    loader = setup_training_loader(
        path_to_train_data=path,
        unchanged_crops=unchanged_crops,
        target_crops=target_crops,
        train_batch_size=batch_size,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )
    return loader

def validate_spectral_net():
    config = load_config()
    TARGET_CROP = 176
    device = 'cuda'
    
    # Parameters for validation
    RESOLUTION = 112 # Default to 112 as in your recent tests
    BATCH_SIZE = 1 
    patch_h = patch_w = RESOLUTION
    window_size = 30
    d_star = 1.0
    
    # Setup sparse structure for specified resolution
    sparse_image_obj = create_sparse_structure_from_images(patch_h, patch_w, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    checkpoint_path = 'checkpoints/finetune/crop176_finetuned_best.pth'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Finetuned checkpoint not found at: {checkpoint_path}")

    # Load the model directly
    print(f"Loading fine-tuned model from {checkpoint_path}...")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Update graph structure and order buffers to match current session's resolution
    model.register_buffer('order', order)
    model.register_buffer('edge_i', edge_i)
    model.register_buffer('edge_j', edge_j)
    model.to(device)
    model.eval()
    
    # Loader
    val_loader = setup_data_loader(
        config, 
        resolution=RESOLUTION, 
        target_crops=[TARGET_CROP],
        batch_size=BATCH_SIZE
    )
    
    overall_confusion = np.zeros((2, 2))
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    with torch.no_grad():
        for bands, labels in tqdm(val_loader, desc="Validation"):
            bands = bands.to(device)
            labels = labels.to(device)
            
            # Forward pass
            _, eigen_vector, _, _, _ = model(bands)
            
            B_curr = eigen_vector.shape[0]
            
            for b in range(B_curr):
                # Compute metrics for each batch item
                pred_eigen = eigen_vector[b].cpu().numpy().flatten()
                pred_sign = np.sign(pred_eigen)
                
                # Reorder labels to match the graph order (Morton order)
                y = labels[b].cpu().numpy().flatten()[order.cpu().numpy()]
                
                # Oracle Sign Correction: Choose the sign that results in higher accuracy
                acc1 = np.sum(y == pred_sign) / len(pred_sign)
                acc2 = np.sum(y == -pred_sign) / len(pred_sign)
                
                if acc1 >= acc2:
                    current_acc = acc1
                    pred = pred_sign
                else:
                    current_acc = acc2
                    pred = -pred_sign

                # Convert to binary
                y_binary = (y == 1).astype(np.int32)
                pred_binary = (pred == 1).astype(np.int32)
                
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
    
    print(f"\nOverall Results from Fine-tuned GraphSpectralNet (Oracle Sign Correction):")
    print(f"Confusion Matrix:\n{overall_confusion}")
    print(f"Precision: {precision:.4f}\nRecall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\nAccuracy: {accuracy:.4f}")

if __name__ == "__main__":
    validate_spectral_net()
