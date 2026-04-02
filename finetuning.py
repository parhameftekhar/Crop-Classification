from model.graph_learning import FeatureExtractor, MLP
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model.graph_learning import create_feature_pairs, modified_sigmoid, create_coo_sparse_matrix
from model.eigen_solver import build_eigen_solver
from model.graph_spectral_net import GraphSpectralNet
from losses.signed_laplacian_loss import SignedLaplacianLoss
from losses.normalized_correlation_loss import NormalizedCorrelationLoss
from losses.mse_loss import MSELoss
from losses.rayleigh_quotient_loss import RayleighQuotientLoss
import torch.optim as optim
import torch
import torch.nn.functional as F
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
    dl = config['data_loader']
    train_cfg = dl.get('train', {})
    val_cfg = dl.get('val', {})
    # Backward compat: if no train/val block, use top-level min_ratio/max_ratio
    train_min = train_cfg.get('min_ratio', dl.get('min_ratio'))
    train_max = train_cfg.get('max_ratio', dl.get('max_ratio'))
    val_min = val_cfg.get('min_ratio', dl.get('min_ratio'))
    val_max = val_cfg.get('max_ratio', dl.get('max_ratio'))

    # Setup training loader
    train_loader = setup_training_loader(
        path_to_train_data=dl['train_path'],
        unchanged_crops=config['general']['unchanged_crops'],
        target_crops=[config['general']['target_crop']],
        train_batch_size=dl['batch_size'],
        crop_band_index=dl['crop_band_index'],
        device=dl['device'],
        ignore_crops=None,
        min_ratio=train_min,
        max_ratio=train_max,
        shuffle=True
    )

    # Setup validation loader
    val_loader = setup_training_loader(
        path_to_train_data=dl['val_path'],
        unchanged_crops=config['general']['unchanged_crops'],
        target_crops=[config['general']['target_crop']],
        train_batch_size=dl['batch_size'],
        crop_band_index=dl['crop_band_index'],
        device=dl['device'],
        ignore_crops=None,
        min_ratio=val_min,
        max_ratio=val_max,
        shuffle=False
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


def validate_model(model, train_loader, val_loader, criterion, config, logger=None):
    model.eval()
    running_loss = 0.0
    overall_confusion = np.zeros((2, 2))
    device = next(model.parameters()).device
    
    # Use the training patch size as the atomic unit for splitting the validation image
    # GraphSpectralNet is already initialized with this structure (buffers)
    patch_h = config['training']['img_height']
    patch_w = config['training']['img_width']
    
    # Heuristic Sign Correction: Calculate feature centers from the TRAINING set
    if logger: logger.info("Calculating feature centers from training set for sign correction...")
    positive_center, negative_center = model.feature_extractor.calculate_feature_centers(train_loader)
    
    # Count total subpatches processed for average loss
    total_subpatches = 0

    with torch.no_grad():
        for bands, labels in tqdm(val_loader, desc="Validation"):
            bands, labels = bands.to(device), labels.to(device)
            # labels shape: (B, H, W)
            
            # Forward pass through the end-to-end model
            eigen_val, eigen_vector, res_loss, L, features_flat, init_guess = model(bands)
            
            # eigen_vector shape: (B, N)
            # features_flat shape: (B, N, C)
            B_curr = eigen_vector.shape[0]
            
            # Calculate Loss for the batch
            if isinstance(criterion, RayleighQuotientLoss):
                loss = criterion(eigen_vector, L)
            else:
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
                
                # Heuristic Sign Correction: Use feature centers to orient the prediction
                sign_correct = correct_pred_sign(pred_sign, features_flat[b], positive_center, negative_center)
                pred_final = ((sign_correct * pred_sign) == 1).astype(np.int32)
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


def log_gamma_stats(model, logger=None):
    """Monitor Gamma values from the unrolled Lanczos solver."""
    if hasattr(model, 'solver') and hasattr(model.solver, 'gamma'):
        g_vals = model.solver.gamma.detach().cpu().numpy()
        g_msg = f"Gamma stats - Mean: {g_vals.mean():.4f}, Min: {g_vals.min():.4f}, Max: {g_vals.max():.4f}\n"
        g_msg += f"Gamma vector: {np.array2string(g_vals, precision=3, separator=', ')}"
        if logger:
            logger.info(g_msg)
        else:
            print(g_msg)


def train_model(model, train_loader, val_loader, config, device, criterion, optimizer, logger=None):
    best_val_f1 = 0.0
    num_epochs = config['training']['num_epochs']
    patch_size = config['training']['img_height']
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    target_crop = config['general']['target_crop']
    aux_weight = config['training'].get('aux_loss_weight', 1)
    save_dir = config['paths']['save_dir']
    validation_steps = config['training'].get('validation_steps', 0)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_spectral_loss = 0.0
        running_aux_loss = 0.0
        latest_grad_norm = 0.0
        
        # --- Phased Parameter Control Logic ---
        if epoch == 0:
            # Phase 1: Only train gamma
            if logger: logger.info("Phase 1: Pre-training solver (Training: GAMMA | Frozen: INIT_HEAD, SUB_SOLVER)")
            # FREEZE init_head (reverting to frozen state for Phase 1)
            if hasattr(model.feature_extractor, 'init_head'):
                for p in model.feature_extractor.init_head.parameters():
                    p.requires_grad = False
            # Freeze sub_solver step sizes
            if hasattr(model.solver, 'sub_solver') and hasattr(model.solver.sub_solver, 'raw_step_sizes'):
                model.solver.sub_solver.raw_step_sizes.requires_grad = False
            # Ensure gamma is trainable
            if hasattr(model.solver, 'gamma'):
                model.solver.gamma.requires_grad = True
        
        elif epoch == 1:
            # Phase 2: Train everything EXCEPT gamma
            if logger: logger.info("Phase 2: Main Training (Training: INIT_HEAD, SUB_SOLVER, Frozen: GAMMA)")
            # Unfreeze init_head
            if hasattr(model.feature_extractor, 'init_head'):
                for p in model.feature_extractor.init_head.parameters():
                    p.requires_grad = True
            # Unfreeze sub_solver step sizes
            if hasattr(model.solver, 'sub_solver') and hasattr(model.solver.sub_solver, 'raw_step_sizes'):
                model.solver.sub_solver.raw_step_sizes.requires_grad = True
            # Freeze gamma
            if hasattr(model.solver, 'gamma'):
                model.solver.gamma.requires_grad = False

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (bands, labels) in enumerate(pbar):
            bands, labels = bands.to(device), labels.to(device)
            # labels shape: (B, H, W)
            
            # Forward pass through the end-to-end model
            # eigen_vector: (B, N), init_guess: (B, H, W, 1)
            eigen_val, eigen_vector, res_loss, L, features_flat, init_guess = model(bands)
            
            # Pick the right arguments for the criterion (Unsupervised vs Supervised)
            if isinstance(criterion, RayleighQuotientLoss):
                loss = criterion(eigen_vector, L)
            else:
                loss = criterion(eigen_vector, labels)
            
            # --- Phased Training (Objective 11 - Simplified to Main Loss Only) ---
            # Both phases now use the main objective (loss)
            # Parameter control is handled at the start of the epoch
            total_loss = res_loss
            

            # Scale loss for gradient accumulation
            loss_scaled = total_loss / accumulation_steps
            loss_scaled.backward()
            
            # Step optimizer only after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Detailed gradient logging (Objective 7️⃣)
                grads = {}
                # Feature Extractor Check
                if hasattr(model.feature_extractor, 'cnn') and hasattr(model.feature_extractor.cnn, 'block_in'):
                    first_layer = model.feature_extractor.cnn.block_in[0]
                    if first_layer.weight.grad is not None:
                        grads['cnn_norm'] = first_layer.weight.grad.norm().item()
                
                # Init Head Check
                if hasattr(model.feature_extractor, 'init_head'):
                    if model.feature_extractor.init_head.weight.grad is not None:
                        grads['init_norm'] = model.feature_extractor.init_head.weight.grad.norm().item()
                
                # M Matrix Check
                if hasattr(model.feature_extractor, 'M'):
                    if model.feature_extractor.M.grad is not None:
                        grads['M_norm'] = model.feature_extractor.M.grad.norm().item()

                # Solver Check for gamma gradients
                if hasattr(model.solver, 'gamma') and model.solver.gamma.grad is not None:
                    grads['gamma_norm'] = model.solver.gamma.grad.norm().item()

                # Prefer showing gradients for parts that are actually training
                if 'init_norm' in grads:
                    latest_grad_norm = grads['init_norm']
                elif 'gamma_norm' in grads:
                    latest_grad_norm = grads['gamma_norm']
                else:
                    latest_grad_norm = grads.get('cnn_norm', 0.0)
                
                # Optional: log all grad norms to a separate file or console if debug is on
                # For now, we update 'latest_grad_norm' which is used in the postfix

                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += (total_loss.item() * accumulation_steps)
            running_spectral_loss += (loss.item() * accumulation_steps)
            running_aux_loss += (res_loss.item() * accumulation_steps)

            # Get mean gamma value for real-time monitoring
            gamma_mean = 0.0
            if hasattr(model.solver, 'gamma'):
                gamma_mean = model.solver.gamma.mean().item()

            pbar.set_postfix({
                'loss': total_loss.item() * accumulation_steps, 
                'spec': loss.item() * accumulation_steps,
                'res': res_loss.item() * accumulation_steps,
                'avg': running_loss / (pbar.n + 1),
                'grad': f"{latest_grad_norm:.2e}",
                'g_m': f"{gamma_mean:.2f}"
            })

            # Check for Step-Based Validation (Objective 8️⃣)
            if validation_steps > 0 and (batch_idx + 1) % validation_steps == 0:
                val_accuracy, val_f1, avg_val_loss = validate_model(
                    model, train_loader, val_loader, criterion,
                    config, logger=logger
                )
                log_gamma_stats(model, logger=logger)
                
                # Save best model based on step-based validation
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, f'crop{target_crop}_finetuned_best.pth')
                    torch.save(model, save_path)
                    msg = f"Step [{batch_idx+1}], New best F1: {best_val_f1:.4f} saved to {save_path}"
                    if logger: logger.info(msg)
                    else: print(msg)
                
                # Reset model to training mode after validation
                model.train()

        avg_epoch_loss = running_loss / len(train_loader)
        
        # Validation phase (End of Epoch)
        val_accuracy, val_f1, avg_val_loss = validate_model(
            model, train_loader, val_loader, criterion,
            config, logger=logger
        )
        log_gamma_stats(model, logger=logger)

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
    
    # Freeze the Shallow CNN backbone and M matrix as requested
    logger.info("Freezing the Shallow CNN backbone and M matrix...")
    for param in model.feature_extractor.cnn.parameters():
        param.requires_grad = False
    model.feature_extractor.M.requires_grad = False
    
    # Initialize loss and optimizer based on config
    if loss_type == 'signed_laplacian':
        criterion = SignedLaplacianLoss(img_height=img_height, img_width=img_width, window_size=window_size)
    elif loss_type == 'normalized_correlation':
        criterion = NormalizedCorrelationLoss(img_height=img_height, img_width=img_width, window_size=window_size)
    elif loss_type == 'mse':
        criterion = MSELoss(img_height=img_height, img_width=img_width, window_size=window_size)
    elif loss_type == 'rayleigh_quotient':
        criterion = RayleighQuotientLoss()
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
