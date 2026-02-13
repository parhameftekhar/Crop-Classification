# train_ShallowCNN.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from data_manager import setup_multiclass_loader
from model import ShallowCNN
from tqdm import tqdm
import logging

# Data setup
TARGET_CROPS = [1, 5, 23, 176]  # Corn, Soybean, Spring Wheat, Grassland/Pasture
UNCHANGED_CROPS = TARGET_CROPS  # Keep these crops unchanged
NUM_CLASSES = len(TARGET_CROPS) + 1  # +1 for background

# Create directories for logs and checkpoints
os.makedirs('logs/benchmark/multi_case', exist_ok=True)
os.makedirs('checkpoints/benchmark/multi_case', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/benchmark/multi_case/training_shallowcnn_multiclass_{NUM_CLASSES}classes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ShallowCNNMultiClassClassifier(nn.Module):
    """
    ShallowCNN with a multi-class classification layer on top.
    
    Args:
        num_block (int): Number of blocks in the ShallowCNN
        kernel_size (int): Kernel size for convolutions
        stride (int): Stride for convolutions
        padding (int): Padding for convolutions
        num_channel_in (int): Number of input channels
        num_channel_internal (int): Number of internal channels
        num_channel_out (int): Number of output channels from ShallowCNN
        num_classes (int): Number of target classes
        device (torch.device): Device to place the model on
    """
    def __init__(self, num_block, kernel_size, stride, padding, num_channel_in, 
                 num_channel_internal, num_channel_out, num_classes, device):
        super(ShallowCNNMultiClassClassifier, self).__init__()
        
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
        
        # Classification head: reduce from num_channel_out to num_classes
        self.classifier = nn.Conv2d(num_channel_out, num_classes, kernel_size=1, stride=1, padding=0).to(device)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, H, W)
        """
        # Get features from ShallowCNN (output shape: B, H, W, C)
        features = self.backbone(x)
        
        # Permute to (B, C, H, W) for Conv2d
        features = features.permute(0, 3, 1, 2)
        
        # Apply classification head
        logits = self.classifier(features)
        
        return logits

# Model setup
logger.info(f'Initializing ShallowCNN Multi-Class Classifier for {NUM_CLASSES} classes')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_config = {
    'num_block': 4,
    'kernel_size': 9,
    'stride': 1,
    'padding': 4,
    'num_channel_in': 18,
    'num_channel_internal': 18,
    'num_channel_out': 18,
    'num_classes': NUM_CLASSES,
    'device': device
}

model = ShallowCNNMultiClassClassifier(
    num_block=model_config['num_block'],
    kernel_size=model_config['kernel_size'],
    stride=model_config['stride'],
    padding=model_config['padding'],
    num_channel_in=model_config['num_channel_in'],
    num_channel_internal=model_config['num_channel_internal'],
    num_channel_out=model_config['num_channel_out'],
    num_classes=model_config['num_classes'],
    device=model_config['device']
)

logger.info(f'Target crops: {TARGET_CROPS}')
logger.info(f'Number of classes: {NUM_CLASSES} (0=background, 1-{NUM_CLASSES-1}=target crops)')

# Setup data loaders
logger.info('Setting up data loaders')
train_loader = setup_multiclass_loader(
    path_to_data='./training_data/train_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=TARGET_CROPS,
    batch_size=16,
    crop_band_index=18,
    device='cuda'
)

val_loader = setup_multiclass_loader(
    path_to_data='./training_data/val_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=TARGET_CROPS,
    batch_size=16,
    crop_band_index=18,
    device='cuda'
)

test_loader = setup_multiclass_loader(
    path_to_data='./training_data/test_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=TARGET_CROPS,
    batch_size=16,
    crop_band_index=18,
    device='cuda'
)

# Training setup
model = model.to(device)
logger.info(f'Training on device: {device}')

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Function to accumulate confusion matrix components per batch
def accumulate_confusion_matrix(outputs, labels, num_classes, confusion_matrix):
    """
    Accumulate confusion matrix components for multi-class segmentation.
    For each class, we track: true_positive, false_positive, false_negative
    """
    _, predicted = torch.max(outputs, 1)
    
    for class_id in range(num_classes):
        pred_mask = (predicted == class_id)
        true_mask = (labels == class_id)
        
        tp = (pred_mask & true_mask).sum().item()
        fp = (pred_mask & ~true_mask).sum().item()
        fn = (~pred_mask & true_mask).sum().item()
        
        confusion_matrix['tp'][class_id] += tp
        confusion_matrix['fp'][class_id] += fp
        confusion_matrix['fn'][class_id] += fn

def compute_metrics_from_confusion_matrix(confusion_matrix, num_classes):
    """
    Compute per-class and mean metrics from accumulated confusion matrix.
    
    Returns:
        tuple: (mean_iou, ious, mean_precision, precisions, mean_recall, recalls, mean_f1, f1_scores)
    """
    tp = confusion_matrix['tp']
    fp = confusion_matrix['fp']
    fn = confusion_matrix['fn']
    
    ious = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_id in range(num_classes):
        # IoU
        union = tp[class_id] + fp[class_id] + fn[class_id]
        if union > 0:
            ious.append(tp[class_id] / union)
        else:
            ious.append(0.0)
        
        # Precision
        if (tp[class_id] + fp[class_id]) > 0:
            precision = tp[class_id] / (tp[class_id] + fp[class_id])
            precisions.append(precision)
        else:
            precision = 0.0
            precisions.append(0.0)
        
        # Recall
        if (tp[class_id] + fn[class_id]) > 0:
            recall = tp[class_id] / (tp[class_id] + fn[class_id])
            recalls.append(recall)
        else:
            recall = 0.0
            recalls.append(0.0)
            
        # F1-score
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)
    
    mean_iou = np.mean(ious)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    
    return mean_iou, ious, mean_precision, precisions, mean_recall, recalls, mean_f1, f1_scores

def calculate_accuracy(outputs, labels):
    """Simple accuracy calculation."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    return correct / total

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    total_accuracy = 0
    batches = 0
    
    confusion_matrix = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes)
    }
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        # ShallowCNN expects (B, H, W, C) but we move to device first
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = calculate_accuracy(outputs, labels)
        accumulate_confusion_matrix(outputs, labels, num_classes, confusion_matrix)
        
        total_loss += loss.item()
        total_accuracy += accuracy
        batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })
    
    # Compute final metrics from confusion matrix
    mean_iou, _, mean_prec, _, mean_rec, _, mean_f1, _ = compute_metrics_from_confusion_matrix(confusion_matrix, num_classes)
    
    avg_loss = total_loss / batches
    avg_acc = total_accuracy / batches
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Training - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, mIoU: {mean_iou:.4f}')
    return avg_loss, avg_acc, mean_iou

# Validation function
def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    batches = 0
    
    confusion_matrix = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes)
    }
    
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            accuracy = calculate_accuracy(outputs, labels)
            accumulate_confusion_matrix(outputs, labels, num_classes, confusion_matrix)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    # Compute final metrics from confusion matrix
    mean_iou, ious, mean_prec, precisions, mean_rec, recalls, mean_f1, f1_scores = compute_metrics_from_confusion_matrix(confusion_matrix, num_classes)
    
    avg_loss = total_loss / batches
    avg_acc = total_accuracy / batches
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, mIoU: {mean_iou:.4f}')
    logger.info(f'Per-class IoU: {", ".join([f"Class {i}: {iou:.4f}" for i, iou in enumerate(ious)])}')
    logger.info(f'Per-class Precision: {", ".join([f"Class {i}: {prec:.4f}" for i, prec in enumerate(precisions)])}')
    logger.info(f'Per-class Recall: {", ".join([f"Class {i}: {rec:.4f}" for i, rec in enumerate(recalls)])}')
    logger.info(f'Per-class F1-Score: {", ".join([f"Class {i}: {f1:.4f}" for i, f1 in enumerate(f1_scores)])}')
    
    return avg_loss, avg_acc, mean_iou

# Training loop
num_epochs = 100
best_val_miou = 0.0
logger.info(f'Starting training for {num_epochs} epochs')

epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
for epoch in epoch_pbar:
    train_loss, train_acc, train_miou = train_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES)
    val_loss, val_acc, val_miou = validate(model, val_loader, criterion, device, NUM_CLASSES)
    
    scheduler.step()
    
    epoch_pbar.set_postfix({
        'train_loss': f'{train_loss:.4f}',
        'train_acc': f'{train_acc:.4f}',
        'train_mIoU': f'{train_miou:.4f}',
        'val_loss': f'{val_loss:.4f}',
        'val_acc': f'{val_acc:.4f}',
        'val_mIoU': f'{val_miou:.4f}'
    })
    
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        torch.save(model.state_dict(), f'checkpoints/benchmark/multi_case/best_shallowcnn_model_multiclass_{NUM_CLASSES}classes.pth')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - New best model saved with validation mIoU: {val_miou:.4f}')

# Load best model for testing
model.load_state_dict(torch.load(f'checkpoints/benchmark/multi_case/best_shallowcnn_model_multiclass_{NUM_CLASSES}classes.pth'))
logger.info('Loaded best model for testing')

# Test the model
test_loss, test_acc, test_miou = validate(model, test_loader, criterion, device, NUM_CLASSES)
logger.info('Test Results:')
logger.info(f'Test Loss: {test_loss:.4f}')
logger.info(f'Test Accuracy: {test_acc:.4f}')
logger.info(f'Test mIoU: {test_miou:.4f}')
