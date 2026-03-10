# train_pan_binary.py

import torch
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from data_manager import create_modified_crop_labels, filter_balanced_patches, setup_training_loader
from tqdm import tqdm
import logging
import sys


# Data setup
TARGET_CROP = 176  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Directories
os.makedirs('logs/benchmark/binary_case', exist_ok=True)
os.makedirs('checkpoints/benchmark/binary_case', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/benchmark/binary_case/training_pan_binary_crop{TARGET_CROP}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model setup
logger.info('Initializing PAN (Pyramid Attention Network) model for binary classification')
model = smp.PAN(
    encoder_name="resnet50",  # You can change this to other encoders like "efficientnet-b0", "densenet121", etc.
    encoder_weights="imagenet",  # Pre-trained weights
    in_channels=18,  # Your 18 input channels
    classes=2,  # Binary classification (background + target crop)
    activation=None,  # No activation for CrossEntropyLoss
    encoder_depth=5,  # Number of encoder blocks
    decoder_channels=256,  # Number of channels in decoder blocks
    decoder_use_batchnorm=True,  # Use batch normalization in decoder
    decoder_attention_type="scse",  # Use scSE attention mechanism
    decoder_use_attention=True,  # Enable attention mechanism
)

logger.info(f'Target crop: {TARGET_CROP}, Unchanged crops: {UNCHANGED_CROPS}')

# Setup data loaders
logger.info('Setting up data loaders')
train_loader = setup_training_loader(
    path_to_train_data='./training_data/train_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=16,
    crop_band_index=18,
    device='cuda',
    ignore_crops=None,
    min_ratio=0,
    max_ratio=1
)

val_loader = setup_training_loader(
    path_to_train_data='./training_data/val_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=16,
    crop_band_index=18,
    device='cuda',
    ignore_crops=None,
    min_ratio=0,
    max_ratio=1
)

test_loader = setup_training_loader(
    path_to_train_data='./training_data/test_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=16,
    crop_band_index=18,
    device='cuda',
    ignore_crops=None,
    min_ratio=0,
    max_ratio=1
)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
logger.info(f'Training on device: {device}')

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Function to transform labels from +1/-1 to 0/1
def transform_labels(labels):
    return ((labels + 1) / 2).long()  # Converts -1 to 0 and +1 to 1

# Function to calculate precision, recall, and F1-score for binary classification
def calculate_metrics(predictions, targets):
    # Convert predictions to binary (0 or 1)
    pred_binary = (predictions > 0.5).astype(float)
    
    # Calculate TP, FP, TN, FN
    TP = ((pred_binary == 1) & (targets == 1)).sum()
    FP = ((pred_binary == 1) & (targets == 0)).sum()
    TN = ((pred_binary == 0) & (targets == 0)).sum()
    FN = ((pred_binary == 0) & (targets == 1)).sum()
    
    # Calculate metrics
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    
    return float(precision), float(recall), float(f1), float(accuracy)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, labels) in enumerate(progress_bar):
        # Fix tensor dimensions: data is (B, H, W, C) -> (B, C, H, W)
        if len(data.shape) == 4 and data.shape[-1] == 18:  # (B, H, W, C)
            data = data.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        data, labels = data.to(device), labels.to(device)
        
        # Transform labels from +1/-1 to 0/1
        labels = transform_labels(labels)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities[:, 1]  # Probability of positive class
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and targets for metrics
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    precision, recall, f1, accuracy = calculate_metrics(
        np.array(all_predictions), np.array(all_targets)
    )
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, precision, recall, f1, accuracy

# Validation function
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch_idx, (data, labels) in enumerate(progress_bar):
            # Fix tensor dimensions: data is (B, H, W, C) -> (B, C, H, W)
            if len(data.shape) == 4 and data.shape[-1] == 18:  # (B, H, W, C)
                data = data.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            data, labels = data.to(device), labels.to(device)
            
            # Transform labels from +1/-1 to 0/1
            labels = transform_labels(labels)
            
            outputs = model(data)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities[:, 1]  # Probability of positive class
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    precision, recall, f1, accuracy = calculate_metrics(
        np.array(all_predictions), np.array(all_targets)
    )
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, precision, recall, f1, accuracy

# Test function
def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for batch_idx, (data, labels) in enumerate(progress_bar):
            # Fix tensor dimensions: data is (B, H, W, C) -> (B, C, H, W)
            if len(data.shape) == 4 and data.shape[-1] == 18:  # (B, H, W, C)
                data = data.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            data, labels = data.to(device), labels.to(device)
            
            # Transform labels from +1/-1 to 0/1
            labels = transform_labels(labels)
            
            outputs = model(data)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities[:, 1]  # Probability of positive class
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    precision, recall, f1, accuracy = calculate_metrics(
        np.array(all_predictions), np.array(all_targets)
    )
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss, precision, recall, f1, accuracy

# Main training loop
def main():
    try:
        # Training parameters
        num_epochs = 100
        best_val_f1 = 0.0
        
        logger.info(f'Starting training for {num_epochs} epochs')
        logger.info(f'Model: PAN with ResNet50 encoder')
        logger.info(f'Target crop: {TARGET_CROP}')
        logger.info(f'Input channels: 18')
        logger.info(f'Input size: 224x224')
        logger.info(f'PyTorch version: {torch.__version__}')
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training phase
            train_loss, train_precision, train_recall, train_f1, train_accuracy = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validation phase
            val_loss, val_precision, val_recall, val_f1, val_accuracy = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.info(f'Train - Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, '
                       f'Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}')
            logger.info(f'Val   - Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, '
                       f'Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}')
            logger.info(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model based on validation F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'checkpoints/benchmark/binary_case/best_pan_binary_crop{TARGET_CROP}.pth')
                logger.info(f'New best model saved with validation F1: {val_f1:.4f}')
                logger.info(f'Validation metrics - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_accuracy:.4f}')
        
        # Load best model for validation and testing
        logger.info('Loading best model for validation and testing...')
        model.load_state_dict(torch.load(f'checkpoints/benchmark/binary_case/best_pan_binary_crop{TARGET_CROP}.pth'))
        
        # Validation phase with best model
        logger.info('Evaluating best model on validation set...')
        best_val_loss, best_val_precision, best_val_recall, best_val_f1, best_val_accuracy = validate_epoch(
            model, val_loader, criterion, device
        )
        logger.info('Best Model Validation Results:')
        logger.info(f'Val Loss: {best_val_loss:.4f}')
        logger.info(f'Val Precision: {best_val_precision:.4f}')
        logger.info(f'Val Recall: {best_val_recall:.4f}')
        logger.info(f'Val F1-Score: {best_val_f1:.4f}')
        logger.info(f'Val Accuracy: {best_val_accuracy:.4f}')
        
        # Test phase
        logger.info('Testing best model...')
        test_loss, test_precision, test_recall, test_f1, test_accuracy = test_model(
            model, test_loader, criterion, device
        )
        
        logger.info(f'Final Test Results:')
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f'Test Precision: {test_precision:.4f}')
        logger.info(f'Test Recall: {test_recall:.4f}')
        logger.info(f'Test F1-Score: {test_f1:.4f}')
        logger.info(f'Test Accuracy: {test_accuracy:.4f}')
        
        logger.info('Training completed successfully!')
        
    except Exception as e:
        logger.error(f'Training failed with error: {str(e)}')
        raise

if __name__ == '__main__':
    main()
