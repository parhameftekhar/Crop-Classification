from model.graph_learning import FeatureExtractor, MLP
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model.graph_learning import create_feature_pairs, modified_sigmoid, create_coo_sparse_matrix
from losses.signed_laplacian_loss import SignedLaplacianLoss
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
import yaml

# Configuration will be loaded in main()
config = None

# Helper to load config
def load_config(config_path='config_finetune.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Global configuration (will be initialized in main)
SEED = None
TARGET_CROP = None
UNCHANGED_CROPS = None

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp and target crop
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/fine_tuning/fine_tuning_crop{config['general']['target_crop']}_{timestamp}.log"
    
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


def setup_data_loaders(config):
    # Setup training loader
    train_loader = setup_training_loader(
        path_to_train_data=config['data_loader']['train_path'],
        unchanged_crops=config['general']['unchanged_crops'],
        target_crops=[config['general']['target_crop']],
        train_batch_size=config['data_loader']['batch_size'],
        crop_band_index=config['data_loader']['crop_band_index'],
        device=config['data_loader']['device'],
        ignore_crops=None,
        min_ratio=config['data_loader']['min_ratio'],
        max_ratio=config['data_loader']['max_ratio']
    )

    # Setup validation loader
    val_loader = setup_training_loader(
        path_to_train_data=config['data_loader']['val_path'],
        unchanged_crops=config['general']['unchanged_crops'],
        target_crops=[config['general']['target_crop']],
        train_batch_size=config['data_loader']['batch_size'],
        crop_band_index=config['data_loader']['crop_band_index'],
        device=config['data_loader']['device'],
        ignore_crops=None,
        min_ratio=config['data_loader']['min_ratio'],
        max_ratio=config['data_loader']['max_ratio']
    )

    return train_loader, val_loader



def load_feature_extractor(logger, config):
    checkpoint_dir = config['paths']['checkpoint_dir']
    target_crop = config['general']['target_crop']
    checkpoint_path = os.path.join(checkpoint_dir, f'crop{target_crop}_vs_all.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.exists(checkpoint_path):
        features_extractor = torch.load(checkpoint_path, weights_only=False)
        features_extractor.to(device)
        logger.info(f"Loaded checkpoint for crop {target_crop} from {checkpoint_path}")
        
        return features_extractor
    else:
        logger.error(f"Checkpoint not found for crop {target_crop} at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def pytorch_shifted_power_iteration(L, max_iter=100, device='cuda'):
    """
    Find the smallest eigenpair of symmetric matrix L using shifted power iteration.
    This implementation is fully differentiable.
    """
    n = L.shape[0]
    
    # Estimate U (upper bound on lambda_max)
    # Since L is a Laplacian, we can use the max row sum of abs values
    # For a sparse COO tensor L
    abs_L_values = torch.abs(L.values())
    row_indices = L.indices()[0]
    row_sums = torch.zeros(n, device=device)
    row_sums.index_add_(0, row_indices, abs_L_values)
    U = torch.max(row_sums)
    
    # Initialize random vector
    x = torch.randn(n, 1, device=device)
    x = x / torch.norm(x)
    
    for _ in range(max_iter):
        # y = (U*I - L)x = U*x - L@x
        # Sparse matrix multiplication in PyTorch
        Lx = torch.sparse.mm(L, x)
        y = U * x - Lx
        x = y / torch.norm(y)
    
    # Rayleigh quotient for the smallest eigenvalue of L
    Lx = torch.sparse.mm(L, x)
    lam = torch.mm(x.t(), Lx)
    
    return lam, x


def validate_model(features_extractor, val_loader, positive_center, negative_center, config, order, edges, edge_i, edge_j, logger=None):
    features_extractor.eval()
    valid_accuracy_list = []
    valid_f1_score_list = []
    
    # Initialize overall confusion matrix
    overall_confusion = np.zeros((2, 2))
    device = next(features_extractor.parameters()).device

    with torch.no_grad():
        for bands, label in tqdm(val_loader, desc="Validation"):
            bands = bands.to(device)
            features = features_extractor(bands).squeeze(0)
            
            # Process each 112x112 quadrant
            for i in range(2):
                for j in range(2):
                    # Extract the 112x112 quadrant
                    start_h = i * config['validation']['img_height']
                    start_w = j * config['validation']['img_width']
                    quadrant_features = features[start_h:start_h+config['validation']['img_height'], start_w:start_w+config['validation']['img_width'], :]
                    quadrant_label = label.squeeze(0)[start_h:start_h+config['validation']['img_height'], start_w:start_w+config['validation']['img_width']]
                    
                    # Reshape and reorder
                    quadrant_features = quadrant_features.reshape(-1, quadrant_features.shape[-1])[order, :]
                    quadrant_label = quadrant_label.reshape(-1)[order]
                    
                    # Calculate distances and weights
                    features_i, features_j = quadrant_features[edge_i], quadrant_features[edge_j]
                    distances = ((features_i - features_j) ** 2).sum(dim=1)
                    weights = modified_sigmoid(distances, config['training']['d_star'], scale=1)
                    
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
    
    # Calculate metrics from overall confusion matrix
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    msg = f"\nOverall Results from Confusion Matrix:\n"
    msg += f"Confusion Matrix:\n{overall_confusion}\n"
    msg += f"Precision: {precision:.4f}\n"
    msg += f"Recall: {recall:.4f}\n"
    msg += f"F1 Score: {f1:.4f}\n"
    msg += f"Accuracy: {accuracy:.4f}"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return accuracy, f1


def train_model(features_extractor, train_loader, val_loader, config, train_order, train_edge_i, train_edge_j, val_order, val_edges, val_edge_i, val_edge_j, device, criterion, optimizer, logger=None):
    best_val_f1 = 0.0
    num_epochs = config['training']['num_epochs']
    patch_size = config['training']['img_height']
    accumulation_steps = config['training']['accumulation_steps']
    d_star = config['training']['d_star']
    target_crop = config['general']['target_crop']
    save_dir = config['paths']['save_dir']
    
    for epoch in range(num_epochs):
        features_extractor.train()
        running_loss = 0.0
        latest_grad_norm = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (bands, labels) in enumerate(pbar):
            bands, labels = bands.to(device), labels.to(device)
            
            # Forward pass through feature extractor
            features = features_extractor(bands).squeeze(0)
            
            # Process exactly one random subpatch to save memory
            start_h = random.randint(0, features.shape[0] - patch_size)
            start_w = random.randint(0, features.shape[1] - patch_size)
            
            quadrant_features = features[start_h:start_h+patch_size, start_w:start_w+patch_size, :]
            quadrant_labels_orig = labels.squeeze(0)[start_h:start_h+patch_size, start_w:start_w+patch_size]
            
            quadrant_features = quadrant_features.reshape(-1, quadrant_features.shape[-1])[train_order, :]
            
            # Calculate weights
            features_i = quadrant_features[train_edge_i]
            features_j = quadrant_features[train_edge_j]
            distances = ((features_i - features_j) ** 2).sum(dim=1)
            weights = modified_sigmoid(distances, d_star, scale=1.0)
            
            num_nodes = quadrant_features.shape[0]
            degree = torch.zeros(num_nodes, device=device)
            degree.index_add_(0, train_edge_i, weights)
            degree.index_add_(0, train_edge_j, weights)
            
            diag_indices = torch.stack([torch.arange(num_nodes, device=device)] * 2, dim=0)
            off_diag_indices = torch.cat([
                torch.stack([train_edge_i, train_edge_j], dim=0),
                torch.stack([train_edge_j, train_edge_i], dim=0)
            ], dim=1)
            
            L_indices = torch.cat([diag_indices, off_diag_indices], dim=1)
            L_values = torch.cat([degree, -weights, -weights], dim=0)
            
            L = torch.sparse_coo_tensor(L_indices, L_values, (num_nodes, num_nodes)).coalesce()
            
            _, eigen_vector = pytorch_shifted_power_iteration(L, max_iter=100, device=device)
            preds = eigen_vector.t() 
            
            gtbound = quadrant_labels_orig.unsqueeze(0) 
            loss = criterion(preds, gtbound)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Step optimizer only after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Inspect first layer gradients before step
                first_layer_grad_norm = 0.0
                if hasattr(features_extractor, 'cnn') and hasattr(features_extractor.cnn, 'block_in'):
                    first_layer = features_extractor.cnn.block_in[0]
                    if first_layer.weight.grad is not None:
                        first_layer_grad_norm = first_layer.weight.grad.norm().item()
                        latest_grad_norm = first_layer_grad_norm
                
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += (loss.item() * accumulation_steps)
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps, 
                'avg_loss': running_loss / (pbar.n + 1),
                'grad': f"{latest_grad_norm:.2e}"
            })

        avg_epoch_loss = running_loss / len(train_loader)
        end_msg = f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}, Latest Grad Norm: {latest_grad_norm:.2e}"
        if logger:
            logger.info(end_msg)
        else:
            print(end_msg)

        # Validation phase
        positive_center, negative_center = features_extractor.calculate_feature_centers(val_loader)
        val_accuracy, val_f1 = validate_model(
            features_extractor, val_loader, positive_center, negative_center,
            config, val_order, val_edges, val_edge_i, val_edge_j, logger=logger
        )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'crop{target_crop}_finetuned_best.pth')
            torch.save(features_extractor, save_path)
            if logger:
                logger.info(f"New best model saved with F1: {best_val_f1:.4f} at {save_path}")
            else:
                print(f"New best model saved with F1: {best_val_f1:.4f} at {save_path}")

    # Save the final model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'crop{target_crop}_finetuned.pth')
    torch.save(features_extractor, save_path)
    if logger:
        logger.info(f"Model saved to {save_path}")
    else:
        print(f"Model saved to {save_path}")

def main():
    global config, SEED, TARGET_CROP, UNCHANGED_CROPS
    config = load_config()
    
    # Set random seeds for reproducibility from config
    SEED = config['general']['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    TARGET_CROP = config['general']['target_crop']
    UNCHANGED_CROPS = config['general']['unchanged_crops']

    logger = setup_logging()
    device = config['data_loader']['device']
    
    # Parameters from config
    d_star = config['training']['d_star']
    img_height = config['training']['img_height']
    img_width = config['training']['img_width']
    window_size = config['training']['window_size']
    
    # Setup sparse structure for training
    sparse_image_obj_train = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    train_order = sparse_image_obj_train['order']
    train_edges = sparse_image_obj_train['edges']
    train_edge_i, train_edge_j = train_edges[:, 0], train_edges[:, 1]
    
    # Setup sparse structure for validation (using validation specific dims)
    val_height = config['validation']['img_height']
    val_width = config['validation']['img_width']
    sparse_image_obj_val = create_sparse_structure_from_images(val_height, val_width, window_size, device)
    val_order = sparse_image_obj_val['order']
    val_edges = sparse_image_obj_val['edges'].cpu().numpy() # eigsh needs numpy
    val_edge_i = val_edges[:, 0]
    val_edge_j = val_edges[:, 1]
    
    # Setup data and model
    train_loader, val_loader = setup_data_loaders(config)
    features_extractor = load_feature_extractor(logger, config)
    
    logger.info(f"Starting fine-tuning for crop {TARGET_CROP} using SignedLaplacianLoss and differentiable shifted power iteration for eigenvector extraction.")
    
    # Initialize loss and optimizer
    criterion = SignedLaplacianLoss(img_height=img_height, img_width=img_width, window_size=window_size)
    criterion.to(device)
    
    optimizer = optim.Adam(features_extractor.parameters(), lr=config['training']['learning_rate'])
    
    # Start training
    train_model(features_extractor, train_loader, val_loader, config,
                train_order, train_edge_i, train_edge_j, 
                val_order, val_edges, val_edge_i, val_edge_j, 
                device, criterion, optimizer, logger=logger)

if __name__ == "__main__":
    main()
