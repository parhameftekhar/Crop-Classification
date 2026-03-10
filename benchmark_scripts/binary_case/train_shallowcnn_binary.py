# train_shallowcnn_binary.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from data_manager import create_modified_crop_labels, filter_balanced_patches, setup_training_loader
from model import ShallowCNN
from tqdm import tqdm
import logging
import sys


# Data setup
TARGET_CROP = 176  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_shallowcnn_binary_crop{TARGET_CROP}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ShallowCNNBinaryClassifier(nn.Module):
    """
    ShallowCNN with a binary classification layer on top.
    
    Args:
        num_block (int): Number of blocks in the ShallowCNN
        kernel_size (int): Kernel size for convolutions
        stride (int): Stride for convolutions
        padding (int): Padding for convolutions
        num_channel_in (int): Number of input channels
        num_channel_internal (int): Number of internal channels
        num_channel_out (int): Number of output channels from ShallowCNN
        device (torch.device): Device to place the model on
    """
    def __init__(self, num_block, kernel_size, stride, padding, num_channel_in, 
                 num_channel_internal, num_channel_out, device):
        super(ShallowCNNBinaryClassifier, self).__init__()
        
        self.device = device
        
        # Create the ShallowCNN backbone
        self.backbone = ShallowCNN(
            num_block=num_block,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_channel_in=num_channel_in,
            num_channel_internal=num_channel_internal,
            num_channel_out=num_channel_out,
            device=device
        )
        
        # Binary classification head: reduce from num_channel_out to 2 classes
        self.classifier = nn.Conv2d(num_channel_out, 2, kernel_size=1, stride=1, padding=0).to(device)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, 2, H, W) for binary classification
        """
        # Get features from ShallowCNN (output shape: B, H, W, C)
        features = self.backbone(x)
        
        # Permute to (B, C, H, W) for Conv2d
        features = features.permute(0, 3, 1, 2)
        
        # Apply classification head
        logits = self.classifier(features)
        
        return logits


# Model setup with the specified config
logger.info('Initializing ShallowCNN Binary Classifier')
model_config = {
    'num_block': 4,
    'kernel_size': 9,
    'stride': 1,
    'padding': 4,
    'num_channel_in': 18,
    'num_channel_internal': 18,
    'num_channel_out': 18,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

model = ShallowCNNBinaryClassifier(
    num_block=model_config['num_block'],
    kernel_size=model_config['kernel_size'],
    stride=model_config['stride'],
    padding=model_config['padding'],
    num_channel_in=model_config['num_channel_in'],
    num_channel_internal=model_config['num_channel_internal'],
    num_channel_out=model_config['num_channel_out'],
    device=model_config['device']
)

logger.info(f'Target crop: {TARGET_CROP}, Unchanged crops: {UNCHANGED_CROPS}')
logger.info(f'Model configuration: {model_config}')

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
def calculate_metrics(outputs, labels, debug=False):
    _, predicted = torch.max(outputs, 1)
    true_positive = ((predicted == 1) & (labels == 1)).sum().item()
    false_positive = ((predicted == 1) & (labels == 0)).sum().item()
    false_negative = ((predicted == 0) & (labels == 1)).sum().item()
    true_negative = ((predicted == 0) & (labels == 0)).sum().item()
    
    if debug:
        total_pred_positive = (predicted == 1).sum().item()
        total_pred_negative = (predicted == 0).sum().item()
        total_label_positive = (labels == 1).sum().item()
        total_label_negative = (labels == 0).sum().item()
        logger.info(f'Debug - Predicted: {total_pred_positive} positive, {total_pred_negative} negative')
        logger.info(f'Debug - Labels: {total_label_positive} positive, {total_label_negative} negative')
        logger.info(f'Debug - TP:{true_positive}, FP:{false_positive}, FN:{false_negative}, TN:{true_negative}')
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return accuracy, precision, recall, f1

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, debug=False):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)  # Already in (B, H, W, C) format
        labels = transform_labels(labels).to(device)
        
        # Forward pass
        outputs = model(images)  # Output shape: (B, 2, H, W)
        
        # Reshape for loss computation
        outputs = outputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 2)
        outputs = outputs.view(-1, 2)  # (B*H*W, 2)
        labels = labels.view(-1)  # (B*H*W)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Debug only the first batch of first epoch
        debug_batch = debug and batch_idx == 0
        accuracy, precision, recall, f1 = calculate_metrics(outputs, labels, debug=debug_batch)
        
        total_loss += loss.item()
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'f1': f'{f1:.4f}'
        })
    
    avg_loss = total_loss / batches
    avg_acc = total_accuracy / batches
    avg_prec = total_precision / batches
    avg_rec = total_recall / batches
    avg_f1 = total_f1 / batches
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Training - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}')
    return avg_loss, avg_acc, avg_prec, avg_rec, avg_f1

# Validation function
def validate(model, val_loader, criterion, device, debug=False):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    batches = 0
    
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = transform_labels(labels).to(device)
            
            # Forward pass
            outputs = model(images)  # Output shape: (B, 2, H, W)
            
            # Reshape for loss computation
            outputs = outputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 2)
            outputs = outputs.view(-1, 2)  # (B*H*W, 2)
            labels = labels.view(-1)  # (B*H*W)
            
            loss = criterion(outputs, labels)
            
            # Debug only the first batch
            debug_batch = debug and batch_idx == 0
            accuracy, precision, recall, f1 = calculate_metrics(outputs, labels, debug=debug_batch)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'f1': f'{f1:.4f}'
            })
    
    avg_loss = total_loss / batches
    avg_acc = total_accuracy / batches
    avg_prec = total_precision / batches
    avg_rec = total_recall / batches
    avg_f1 = total_f1 / batches
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}')
    return avg_loss, avg_acc, avg_prec, avg_rec, avg_f1

# Training loop
num_epochs = 100
best_val_f1 = -1.0  # Set to -1 to ensure at least one checkpoint is saved
logger.info(f'Starting training for {num_epochs} epochs')

epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
debug_first_epoch = True
for epoch in epoch_pbar:
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device, debug=debug_first_epoch)
    val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device, debug=debug_first_epoch)
    debug_first_epoch = False  # Only debug first epoch
    
    scheduler.step()
    
    epoch_pbar.set_postfix({
        'train_loss': f'{train_loss:.4f}',
        'train_acc': f'{train_acc:.4f}',
        'train_f1': f'{train_f1:.4f}',
        'val_loss': f'{val_loss:.4f}',
        'val_acc': f'{val_acc:.4f}',
        'val_f1': f'{val_f1:.4f}'
    })
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), f'best_shallowcnn_model_binary_crop{TARGET_CROP}.pth')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - New best model saved with validation F1-score: {val_f1:.4f}')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation metrics - Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}')

# Load best model for validation and testing
import os
checkpoint_path = f'best_shallowcnn_model_binary_crop{TARGET_CROP}.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info('Loaded best model for validation and testing')
else:
    logger.warning(f'Checkpoint not found: {checkpoint_path}. Using final model weights for validation and testing.')

# Evaluate best model on validation set
best_val_loss, best_val_acc, best_val_prec, best_val_rec, best_val_f1 = validate(model, val_loader, criterion, device)
logger.info('Best Model Validation Results:')
logger.info(f'Val Loss: {best_val_loss:.4f}')
logger.info(f'Val Accuracy: {best_val_acc:.4f}')
logger.info(f'Val Precision: {best_val_prec:.4f}')
logger.info(f'Val Recall: {best_val_rec:.4f}')
logger.info(f'Val F1-score: {best_val_f1:.4f}')

# Test the model
test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)
logger.info('Test Results:')
logger.info(f'Test Loss: {test_loss:.4f}')
logger.info(f'Test Accuracy: {test_acc:.4f}')
logger.info(f'Test Precision: {test_prec:.4f}')
logger.info(f'Test Recall: {test_rec:.4f}')
logger.info(f'Test F1-score: {test_f1:.4f}')

# Count and log model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Total parameters: {total_params:,}')
logger.info(f'Trainable parameters: {trainable_params:,}')

