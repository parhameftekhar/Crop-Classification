# train_unet_multiclass.py

import torch
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from data_manager import setup_multiclass_loader
from tqdm import tqdm
import logging
import sys
from torch.utils.tensorboard import SummaryWriter


# Data setup
TARGET_CROPS = [1, 5, 23, 176]  # Corn, Soybean, Spring Wheat, Grassland/Pasture
UNCHANGED_CROPS = TARGET_CROPS  # Keep these crops unchanged
NUM_CLASSES = len(TARGET_CROPS) + 1  # +1 for background

# Create directories for logs, checkpoints, and TensorBoard runs
os.makedirs('logs/benchmark/multi_case', exist_ok=True)
os.makedirs('checkpoints/benchmark/multi_case', exist_ok=True)
os.makedirs('tensorboard/benchmark/multi_case', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/benchmark/multi_case/training_unet_multiclass_{NUM_CLASSES}classes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# TensorBoard writer
writer = SummaryWriter(log_dir=f'tensorboard/benchmark/multi_case/unet')

# Model setup
logger.info(f'Initializing UNet model for {NUM_CLASSES}-class classification')
model = smp.Unet(
    encoder_name="resnet50",  # You can change this to other encoders
    encoder_weights="imagenet",  # Pre-trained weights
    in_channels=18,  # Your 18 input channels
    classes=NUM_CLASSES,  # Multi-class classification
    activation=None,  # No activation for CrossEntropyLoss
    encoder_depth=5,  # Number of encoder blocks
    decoder_channels=(256, 128, 64, 32, 16),  # Number of channels in decoder blocks
    decoder_use_batchnorm=True,  # Use batch normalization in decoder
    decoder_attention_type=None,  # No attention mechanism (you can use "scse" for attention)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    Args:
        outputs: Model predictions (logits)
        labels: Ground truth labels
        num_classes: Number of classes
        confusion_matrix: Dictionary with TP, FP, FN arrays to accumulate into
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
    
    # Initialize confusion matrix
    confusion_matrix = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes)
    }
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.permute(0, 3, 1, 2).to(device)  # Change to (B, C, H, W)
        labels = labels.to(device)
        
        # UNet returns logits directly
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
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
    return avg_loss, avg_acc, mean_iou, confusion_matrix

# Validation function
def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    batches = 0
    
    # Initialize confusion matrix
    confusion_matrix = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes)
    }
    
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.permute(0, 3, 1, 2).to(device)  # Change to (B, C, H, W)
            labels = labels.to(device)
            
            # UNet returns logits directly
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate metrics
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
    
    return avg_loss, avg_acc, mean_iou, ious, precisions, recalls, f1_scores

# Training loop
num_epochs = 100
best_val_miou = 0.0
logger.info(f'Starting training for {num_epochs} epochs')

epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
for epoch in epoch_pbar:
    train_loss, train_acc, train_miou, train_cm = train_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES)
    val_loss, val_acc, val_miou, val_ious, val_precs, val_recs, val_f1s = validate(model, val_loader, criterion, device, NUM_CLASSES)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    # ── TensorBoard logging ──────────────────────────────────────────────
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('mIoU/train', train_miou, epoch)
    writer.add_scalar('mIoU/val', val_miou, epoch)
    writer.add_scalar('LearningRate', current_lr, epoch)
    for cls_id, (iou, prec, rec, f1) in enumerate(zip(val_ious, val_precs, val_recs, val_f1s)):
        writer.add_scalar(f'PerClass_IoU/class_{cls_id}', iou, epoch)
        writer.add_scalar(f'PerClass_Precision/class_{cls_id}', prec, epoch)
        writer.add_scalar(f'PerClass_Recall/class_{cls_id}', rec, epoch)
        writer.add_scalar(f'PerClass_F1/class_{cls_id}', f1, epoch)
    # ────────────────────────────────────────────────────────────────────
    
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
        torch.save(model.state_dict(), f'checkpoints/benchmark/multi_case/best_unet_model_multiclass_{NUM_CLASSES}classes.pth')
        logger.info(f'Epoch {epoch+1}/{num_epochs} - New best model saved with validation mIoU: {val_miou:.4f}')

# Load best model for testing
model.load_state_dict(torch.load(f'checkpoints/benchmark/multi_case/best_unet_model_multiclass_{NUM_CLASSES}classes.pth'))
logger.info('Loaded best model for testing')

# Test the model
test_loss, test_acc, test_miou, test_ious, test_precs, test_recs, test_f1s = validate(model, test_loader, criterion, device, NUM_CLASSES)
logger.info('Test Results:')
logger.info(f'Test Loss: {test_loss:.4f}')
logger.info(f'Test Accuracy: {test_acc:.4f}')
logger.info(f'Test mIoU: {test_miou:.4f}')

# Log final test metrics to TensorBoard
writer.add_scalar('Test/Loss', test_loss, 0)
writer.add_scalar('Test/Accuracy', test_acc, 0)
writer.add_scalar('Test/mIoU', test_miou, 0)
for cls_id, (iou, prec, rec, f1) in enumerate(zip(test_ious, test_precs, test_recs, test_f1s)):
    writer.add_scalar(f'Test_PerClass_IoU/class_{cls_id}', iou, 0)
    writer.add_scalar(f'Test_PerClass_F1/class_{cls_id}', f1, 0)

writer.close()
logger.info('TensorBoard writer closed. Run: tensorboard --logdir=tensorboard/benchmark/multi_case')
