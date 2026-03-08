from model.graph_learning import FeatureExtractor, MLP
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model.graph_learning import create_feature_pairs, modified_sigmoid, create_coo_sparse_matrix
from model.eigen_solver import build_eigen_solver
from model.graph_spectral_net import GraphSpectralNet
from losses.signed_laplacian_loss import SignedLaplacianLoss
from losses.normalized_correlation_loss import NormalizedCorrelationLoss
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


# Solver implementation is now in model/eigen_solver.py


def validate_model(model, val_loader, criterion, positive_center, negative_center, config, logger=None):
    model.eval()
    running_loss = 0.0
    overall_confusion = np.zeros((2, 2))
    device = next(model.parameters()).device
    
    # Use the training patch size as the atomic unit for splitting the validation image
    # GraphSpectralNet is already initialized with this structure (buffers)
    patch_h = config['training']['img_height']
    patch_w = config['training']['img_width']
    
    # Count total subpatches processed for average loss
    total_subpatches = 0

    with torch.no_grad():
        for bands, labels in tqdm(val_loader, desc="Validation"):
            bands, labels = bands.to(device), labels.to(device)
            # labels shape: (B, H, W)
            
            # Forward pass through the end-to-end model
            eigen_val, eigen_vector, L, features_flat = model(bands)
            
            # eigen_vector shape: (B, N)
            # features_flat shape: (B, N, C)
            B_curr = eigen_vector.shape[0]
            
            # Calculate Loss for the batch
            # criterion expects preds (B, N) and ground_truth (B, H, W)
            loss = criterion(eigen_vector, labels)
            running_loss += loss.item()
            total_subpatches += B_curr
            
            for b in range(B_curr):
                # Compute metrics for each batch item
                pred_eigen = eigen_vector[b].cpu().numpy().flatten()
                pred_sign = np.sign(pred_eigen)
                
                # Reorder labels to match the graph order (Morton order)
                order_np = model.order.cpu().numpy()
                y = labels[b].cpu().numpy().flatten()[order_np]
                
                # Oracle Sign Correction (or heuristic if centers are updated)
                # Note: features_flat is (B, N, C), so we take features_flat[b]
                sign_correct = correct_pred_sign(pred_sign, features_flat[b], positive_center, negative_center)
                
                # Correct sign and get binary prediction (0 or 1)
                pred_final = (sign_correct * pred_sign == 1).astype(np.int32)
                y_binary = (y == 1).astype(np.int32)
                
                overall_confusion += confusion_matrix(y_binary, pred_final, labels=[0, 1])

    # Calculate metrics from overall confusion matrix
    tn, fp, fn, tp = overall_confusion.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    avg_val_loss = running_loss / total_subpatches if total_subpatches > 0 else 0
    
    msg = f"\nOverall Validation Results (Loss: {avg_val_loss:.6f}):\n"
    msg += f"Confusion Matrix:\n{overall_confusion}\n"
    msg += f"Precision: {precision:.4f}\n"
    msg += f"Recall: {recall:.4f}\n"
    msg += f"F1 Score: {f1:.4f}\n"
    msg += f"Accuracy: {accuracy:.4f}"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return accuracy, f1, avg_val_loss


def train_model(model, train_loader, val_loader, config, device, criterion, optimizer, logger=None):
    best_val_f1 = 0.0
    num_epochs = config['training']['num_epochs']
    patch_size = config['training']['img_height']
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    target_crop = config['general']['target_crop']
    save_dir = config['paths']['save_dir']
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        latest_grad_norm = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (bands, labels) in enumerate(pbar):
            bands, labels = bands.to(device), labels.to(device)
            # labels shape: (B, H, W)
            
            # Forward pass through the end-to-end model
            eigen_val, eigen_vector, L, features_flat = model(bands)
            
            # eigen_vector shape: (B, N), labels shape: (B, H, W)
            loss = criterion(eigen_vector, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Step optimizer only after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Inspect first layer gradients before step
                first_layer_grad_norm = 0.0
                if hasattr(model.feature_extractor, 'cnn') and hasattr(model.feature_extractor.cnn, 'block_in'):
                    first_layer = model.feature_extractor.cnn.block_in[0]
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
        
        # Validation phase
        positive_center, negative_center = model.feature_extractor.calculate_feature_centers(val_loader)
        val_accuracy, val_f1, avg_val_loss = validate_model(
            model, val_loader, criterion, positive_center, negative_center,
            config, logger=logger
        )

        end_msg = f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_epoch_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}, Latest Grad Norm: {latest_grad_norm:.2e}"
        if logger:
            logger.info(end_msg)
        else:
            print(end_msg)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'crop{target_crop}_finetuned_best.pth')
            # Save the whole model
            torch.save(model, save_path)
            if logger:
                logger.info(f"New best model saved with F1: {best_val_f1:.4f} at {save_path}")
            else:
                print(f"New best model saved with F1: {best_val_f1:.4f} at {save_path}")

    # Save the final model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'crop{target_crop}_finetuned.pth')
    torch.save(model, save_path)
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
    
    # Setup shared sparse structure for both training and validation
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    # Setup data and model
    train_loader, val_loader = setup_data_loaders(config)
    
    checkpoint_dir = config['paths']['checkpoint_dir']
    checkpoint_path = os.path.join(checkpoint_dir, f'crop{TARGET_CROP}_vs_all.pth')
    
    loss_type = config['training'].get('loss_type', 'signed_laplacian')
    solver_type_train = config['training']['solver']['type']
    logger.info(f"Starting end-to-end fine-tuning for crop {TARGET_CROP} using {loss_type} and {solver_type_train} solver.")
    
    # Inject n_nodes into solver config automatically
    config['training']['solver']['n_nodes'] = img_height * img_width
    
    # Initialize the End-to-End model with the shared structure
    model = GraphSpectralNet(
        feature_extractor_checkpoint=checkpoint_path,
        solver_cfg=config['training']['solver'],
        order=order,
        edge_i=edge_i,
        edge_j=edge_j,
        d_star=d_star,
        device=device
    )
    model.to(device)
    
    # Initialize loss and optimizer
    if loss_type == 'signed_laplacian':
        criterion = SignedLaplacianLoss(img_height=img_height, img_width=img_width, window_size=window_size)
    elif loss_type == 'normalized_correlation':
        criterion = NormalizedCorrelationLoss(img_height=img_height, img_width=img_width, window_size=window_size)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    criterion.to(device)
    
    # Collect all trainable parameters from the entire model pipeline
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Start training
    train_model(model, train_loader, val_loader, config,
                device, criterion, optimizer, 
                logger=logger)

if __name__ == "__main__":
    main()
