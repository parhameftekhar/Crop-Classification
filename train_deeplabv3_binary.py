# train_deeplabv3_binary.py

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from data_manager import create_modified_crop_labels, filter_balanced_patches, setup_training_loader
from tqdm import tqdm
import logging
import sys


# Data setup
TARGET_CROP = -1  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_deeplabv3_binary_crop{TARGET_CROP}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model setup
logger.info('Initializing DeepLabV3 model for binary classification')
weights = deeplabv3.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(num_classes=2)

# Modify the first convolution layer for 18 input channels
original_conv = model.backbone.conv1
new_conv = torch.nn.Conv2d(
    in_channels=18,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None,
)
model.backbone.conv1 = new_conv

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
    min_ratio=0.1,
    max_ratio=0.9
)

val_loader = setup_training_loader(
    path_to_train_data='./training_data/val_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=16,
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
    train_batch_size=16,
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
        
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy, precision, recall, f1 = calculate_metrics(outputs, labels)
        
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
            
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            
            accuracy, precision, recall, f1 = calculate_metrics(outputs, labels)
            
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
        torch.save(model.state_dict(), 'best_model_binary.pth')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - New best model saved with validation F1-score: {val_f1:.4f}')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation metrics - Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}')

# Load best model for testing
model.load_state_dict(torch.load('best_model_binary.pth'))
logger.info('Loaded best model for testing')

# Test the model
test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)
logger.info('Test Results:')
logger.info(f'Test Loss: {test_loss:.4f}')
logger.info(f'Test Accuracy: {test_acc:.4f}')
logger.info(f'Test Precision: {test_prec:.4f}')
logger.info(f'Test Recall: {test_rec:.4f}')
logger.info(f'Test F1-score: {test_f1:.4f}') 