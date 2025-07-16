from model import FeatureExtractor, MLP
import numpy as np
from data_manager import setup_training_loader, create_sparse_structure_from_images
from model import create_feature_pairs, modified_sigmoid, create_coo_sparse_matrix
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
TARGET_CROP = 1  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp and target crop
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/fine_tuning_crop{TARGET_CROP}_{timestamp}.log'
    
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

def setup_data_loaders():
    # Setup training loader
    train_loader = setup_training_loader(
        path_to_train_data='./training_data/train_patches.npy',
        unchanged_crops=UNCHANGED_CROPS,
        target_crops=[TARGET_CROP],
        train_batch_size=16,
        crop_band_index=18,
        device='cuda',
        ignore_crops=None,
        min_ratio=0.1,
        max_ratio=0.9
    )

    # Setup validation loader
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

    return train_loader, val_loader

def load_feature_extractor(logger, checkpoint_dir='./checkpoints'):
    checkpoint_path = os.path.join(checkpoint_dir, f'crop{TARGET_CROP}_vs_all.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.exists(checkpoint_path):
        features_extractor = torch.load(checkpoint_path, weights_only=False)
        features_extractor.to(device)
        logger.info(f"Loaded checkpoint for crop {TARGET_CROP} from {checkpoint_path}")
        
        # Freeze the model parameters
        for param in features_extractor.parameters():
            param.requires_grad = False
        logger.info("Feature extractor parameters are frozen. No updates will be made during training.")
        
        # Count trainable parameters (should be 0 since model is frozen)
        trainable_params = sum(p.numel() for p in features_extractor.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters in feature extractor: {trainable_params}")
        logger.info(f"Finetuning for target crop: {TARGET_CROP} with frozen feature extractor")
        
        return features_extractor
    else:
        logger.error(f"Checkpoint not found for crop {TARGET_CROP} at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def setup_mlp(logger, num_features, num_layers=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    mlp = MLP(num_features=num_features, num_layers=num_layers, device=device)
    logger.info(f"MLP setup with {num_features} input features and {num_layers} hidden layers")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters in MLP: {trainable_params}")
    
    return mlp

def train_model(features_extractor, mlp, train_loader, val_loader, num_epochs=300):
    logger = setup_logging()
    
    optimizer = optim.Adam(features_extractor.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    
    d_star = 1.0
    margin = 5.0
    
    # Setup sparse structure
    img_height = img_width = 112
    window_size = 30
    device = 'cuda'
    sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, device)
    order = sparse_image_obj['order']
    edges = sparse_image_obj['edges']
    edges = edges.cpu().numpy()
    edge_i, edge_j = edges[:, 0], edges[:, 1]
    
    best_val_f1 = 0.0
    
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    features_extractor.eval()
    mlp.train()

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (bands, label) in enumerate(tqdm(train_loader, desc="Training")):
            # Extract features

            # Select random 112x112 patch from each 224x224 image in the batch
            batch_size = bands.size(0)
            patch_size = 112
            start_h = torch.randint(0, bands.size(1) - patch_size + 1, (batch_size,)).to(bands.device)
            start_w = torch.randint(0, bands.size(2) - patch_size + 1, (batch_size,)).to(bands.device)
            
            patches = torch.zeros(batch_size, patch_size, patch_size, bands.size(3), device=bands.device)
            patch_labels = torch.zeros(batch_size, patch_size, patch_size, device=bands.device)
            
            for i in range(batch_size):
                patches[i] = bands[i, start_h[i]:start_h[i]+patch_size, start_w[i]:start_w[i]+patch_size, :]
                patch_labels[i] = label[i, start_h[i]:start_h[i]+patch_size, start_w[i]:start_w[i]+patch_size]
            
            bands = patches
            label = patch_labels
            
            with torch.no_grad():
                features = features_extractor(bands)

            features_i, features_j = quadrant_features[edge_i], quadrant_features[edge_j]
            distances = ((features_i - features_j) ** 2).sum(dim=1)
            weights = modified_sigmoid(distances, d_star, scale=1)

            features = features.reshape(-1, features.shape[-1])
            label = label.reshape(-1)
            
            
