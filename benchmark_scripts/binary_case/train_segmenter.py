# train_segmenter.py

import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
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

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see shape information
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_segmenter_crop{TARGET_CROP}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model setup - Using Segmenter model
logger.info('Initializing Segmenter model for binary classification')

model_name = "nvidia/segmenter-vit-b"

try:
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification (background + target crop)
        ignore_mismatched_sizes=True
    )
    
    # Get the image processor for preprocessing
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    logger.info(f'Successfully loaded model: {model_name}')
    
except Exception as e:
    logger.error(f'Failed to load model {model_name}: {e}')
    logger.info('Falling back to SegFormer-B0')
    
    # Fallback to a simpler model
    fallback_model = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = AutoModelForSemanticSegmentation.from_pretrained(
        fallback_model,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    image_processor = AutoImageProcessor.from_pretrained(fallback_model)
    logger.info(f'Loaded fallback model: {fallback_model}')

# Modify the model to accept 18 input channels
logger.info('Modifying model input layer for 18 channels')

# For Segmenter models, we need to modify the patch embedding layer
if hasattr(model, 'segmenter') and hasattr(model.segmenter, 'embeddings'):
    # For Segmenter models
    original_conv = model.segmenter.embeddings.patch_embeddings.projection
    new_conv = torch.nn.Conv2d(
        in_channels=18,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    model.segmenter.embeddings.patch_embeddings.projection = new_conv
    logger.info('Modified Segmenter input layer for 18 channels')
    
elif hasattr(model, 'segformer') and hasattr(model.segformer, 'embeddings'):
    # For SegFormer models (fallback)
    original_conv = model.segformer.embeddings.patch_embeddings.projection
    new_conv = torch.nn.Conv2d(
        in_channels=18,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    model.segformer.embeddings.patch_embeddings.projection = new_conv
    logger.info('Modified SegFormer input layer for 18 channels')
    
else:
    logger.warning('Model architecture not recognized for input channel modification')
    # Try to find and modify the first convolutional layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            logger.info(f'Found convolutional layer: {name}')
            new_conv = torch.nn.Conv2d(
                in_channels=18,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
            )
            # Replace the layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_conv)
            else:
                setattr(model, child_name, new_conv)
            logger.info(f'Modified layer {name} for 18 channels')
            break

logger.info(f'Target crop: {TARGET_CROP}, Unchanged crops: {UNCHANGED_CROPS}')

# Setup data loaders
logger.info('Setting up data loaders')
train_loader = setup_training_loader(
    path_to_train_data='./training_data/train_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=8,  # Reduced batch size for Segmenter as it's larger
    crop_band_index=18,
    device='cuda',
    ignore_crops=None,
    min_ratio=0.1,
    max_ratio=0.9
)

val_loader = setup_training_loader(
    path_to_train_data='./training_data/val_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=8,
    crop_band_index=18,
    device='cuda',
    ignore_crops=None,
    min_ratio=0.1,
    max_ratio=0.9
)

test_loader = setup_training_loader(
    path_to_train_data='./training_data/test_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=8,
    crop_band_index=18,
    device='cuda',
    ignore_crops=None,
    min_ratio=0.1,
    max_ratio=0.9
)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
logger.info(f'Training on device: {device}')

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Function to transform labels from +1/-1 to 0/1
def transform_labels(labels):
    return ((labels + 1) / 2).long()  # Converts -1 to 0 and +1 to 1

# Function to calculate precision, recall, and F1-score for binary classification
def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    true_positive = ((predicted == 1) & (labels == 1)).sum().item()
    false_positive = ((predicted == 1) & (labels == 0)).sum().item()
    false_negative = ((predicted == 0) & (labels == 1)).sum().item()
    true_negative = ((predicted == 0) & (labels == 0)).sum().item()
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return accuracy, precision, recall, f1

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.permute(0, 3, 1, 2).to(device)  # Change to (B, C, H, W)
        labels = transform_labels(labels).to(device)
        
        # Segmenter model forward pass
        outputs = model(images)
        
        # Extract logits from the output
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Debug: Print shapes
        logger.debug(f'Images shape: {images.shape}')
        logger.debug(f'Labels shape: {labels.shape}')
        logger.debug(f'Logits shape: {logits.shape}')
        
        # Handle different output resolutions
        B, C, H_out, W_out = logits.shape
        B_l, H_l, W_l = labels.shape
        
        # If output resolution is different from input, resize labels to match
        if H_out != H_l or W_out != W_l:
            logger.info(f'Resizing labels from {H_l}x{W_l} to {H_out}x{W_out}')
            labels = torch.nn.functional.interpolate(
                labels.unsqueeze(1).float(),  # Add channel dimension
                size=(H_out, W_out),
                mode='nearest'
            ).squeeze(1).long()  # Remove channel dimension and convert back to long
        
        # Reshape for loss calculation - CrossEntropyLoss expects (N, C) and (N,)
        logits = logits.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        logits = logits.reshape(-1, C)  # (B*H*W, C)
        labels = labels.view(-1)  # (B*H*W,)
        
        # Debug: Print flattened shapes
        logger.debug(f'Flattened logits shape: {logits.shape}')
        logger.debug(f'Flattened labels shape: {labels.shape}')
        
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Reshape back for metrics calculation
        logits = logits.view(B, H_out*W_out, C).permute(0, 2, 1).view(B, C, H_out, W_out)
        labels = labels.view(B, H_out, W_out)
        
        accuracy, precision, recall, f1 = calculate_metrics(logits, labels)
        
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
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    batches = 0
    
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.permute(0, 3, 1, 2).to(device)  # Change to (B, C, H, W)
            labels = transform_labels(labels).to(device)
            
            # Segmenter model forward pass
            outputs = model(images)
            
            # Extract logits from the output
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Handle different output resolutions
            B, C, H_out, W_out = logits.shape
            B_l, H_l, W_l = labels.shape
            
            # If output resolution is different from input, resize labels to match
            if H_out != H_l or W_out != W_l:
                labels = torch.nn.functional.interpolate(
                    labels.unsqueeze(1).float(),  # Add channel dimension
                    size=(H_out, W_out),
                    mode='nearest'
                ).squeeze(1).long()  # Remove channel dimension and convert back to long
            
            # Reshape for loss calculation - CrossEntropyLoss expects (N, C) and (N,)
            logits = logits.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            logits = logits.reshape(-1, C)  # (B*H*W, C)
            labels = labels.view(-1)  # (B*H*W,)
            
            loss = criterion(logits, labels)
            
            # Reshape back for metrics calculation
            logits = logits.view(B, H_out*W_out, C).permute(0, 2, 1).view(B, C, H_out, W_out)
            labels = labels.view(B, H_out, W_out)
            
            accuracy, precision, recall, f1 = calculate_metrics(logits, labels)
            
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
best_val_f1 = 0.0
logger.info(f'Starting training for {num_epochs} epochs')

epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
for epoch in epoch_pbar:
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
    
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
        torch.save(model.state_dict(), f'best_segmenter_model_crop{TARGET_CROP}.pth')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - New best model saved with validation F1-score: {val_f1:.4f}')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation metrics - Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}')

# Load best model for testing
model.load_state_dict(torch.load(f'best_segmenter_model_crop{TARGET_CROP}.pth'))
logger.info('Loaded best model for testing')

# Test the model
test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)
logger.info('Test Results:')
logger.info(f'Test Loss: {test_loss:.4f}')
logger.info(f'Test Accuracy: {test_acc:.4f}')
logger.info(f'Test Precision: {test_prec:.4f}')
logger.info(f'Test Recall: {test_rec:.4f}')
logger.info(f'Test F1-score: {test_f1:.4f}')
