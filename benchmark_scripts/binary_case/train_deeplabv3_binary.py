# train_deeplabv3_binary.py

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from torch.utils.data import DataLoader
import os
from data_manager import setup_training_loader
from tqdm import tqdm
import logging
import sys
from torch.utils.tensorboard import SummaryWriter

# Data setup
TARGET_CROP = 176  # The crop ID we're training to detect
UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops

# Create directories for logs, checkpoints, and TensorBoard runs
os.makedirs('logs/benchmark/binary_case', exist_ok=True)
os.makedirs('checkpoints/benchmark/binary_case', exist_ok=True)
os.makedirs('tensorboard/benchmark/binary_case', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/benchmark/binary_case/training_deeplabv3_binary_crop{TARGET_CROP}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# TensorBoard writer
writer = SummaryWriter(log_dir=f'tensorboard/benchmark/binary_case/deeplabv3_crop{TARGET_CROP}')

# Model setup
logger.info(f'Initializing DeepLabV3 model for binary classification (Target: {TARGET_CROP})')
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

# Setup data loaders
logger.info('Setting up data loaders')
train_loader = setup_training_loader(
    path_to_train_data='./training_data/train_patches.npy',
    unchanged_crops=UNCHANGED_CROPS,
    target_crops=[TARGET_CROP],
    train_batch_size=16,
    crop_band_index=18,
    device='cuda',
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def transform_labels(labels):
    return ((labels + 1) / 2).long()

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()
    total = tp + fp + fn + tn
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / total if total > 0 else 0
    return acc, prec, rec, f1

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_prec, total_rec, total_f1, batches = 0, 0, 0, 0, 0, 0
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs = imgs.permute(0, 3, 1, 2).to(device)
        labels = transform_labels(labels).to(device)
        outputs = model(imgs)['out']
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc, prec, rec, f1 = calculate_metrics(outputs, labels)
        total_loss += loss.item()
        total_acc += acc
        total_prec += prec
        total_rec += rec
        total_f1 += f1
        batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}', 'f1': f'{f1:.4f}'})
    
    avg_loss = total_loss/batches
    avg_acc = total_acc/batches
    avg_prec = total_prec/batches
    avg_rec = total_rec/batches
    avg_f1 = total_f1/batches
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Training - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}')
    return avg_loss, avg_acc, avg_prec, avg_rec, avg_f1

def validate(model, loader, criterion, device):
    model.eval()
    t_loss, t_acc, t_prec, t_rec, t_f1, batches = 0, 0, 0, 0, 0, 0
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.permute(0, 3, 1, 2).to(device)
            labels = transform_labels(labels).to(device)
            outputs = model(imgs)['out']
            loss = criterion(outputs, labels)
            acc, prec, rec, f1 = calculate_metrics(outputs, labels)
            t_loss += loss.item()
            t_acc += acc
            t_prec += prec
            t_rec += rec
            t_f1 += f1
            batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}', 'f1': f'{f1:.4f}'})
    
    avg_loss = t_loss/batches
    avg_acc = t_acc/batches
    avg_prec = t_prec/batches
    avg_rec = t_rec/batches
    avg_f1 = t_f1/batches
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}')
    return avg_loss, avg_acc, avg_prec, avg_rec, avg_f1

# Training loop
num_epochs = 100
best_val_f1 = 0.0
for epoch in range(num_epochs):
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    v_loss, v_acc, v_prec, v_rec, v_f1 = validate(model, val_loader, criterion, device)
    scheduler.step()
    curr_lr = optimizer.param_groups[0]['lr']
    
    # TensorBoard
    writer.add_scalar('Loss/train', tr_loss, epoch)
    writer.add_scalar('Loss/val', v_loss, epoch)
    writer.add_scalar('Accuracy/train', tr_acc, epoch)
    writer.add_scalar('Accuracy/val', v_acc, epoch)
    writer.add_scalar('Precision/train', tr_prec, epoch)
    writer.add_scalar('Precision/val', v_prec, epoch)
    writer.add_scalar('Recall/train', tr_rec, epoch)
    writer.add_scalar('Recall/val', v_rec, epoch)
    writer.add_scalar('F1/train', tr_f1, epoch)
    writer.add_scalar('F1/val', v_f1, epoch)
    writer.add_scalar('LearningRate', curr_lr, epoch)
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Validation F1: {v_f1:.4f}, Accuracy: {v_acc:.4f}, Precision: {v_prec:.4f}, Recall: {v_rec:.4f}')
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        torch.save(model.state_dict(), f'checkpoints/benchmark/binary_case/best_deeplabv3_model_binary_crop{TARGET_CROP}.pth')
        logger.info(f'New best model saved with validation F1: {v_f1:.4f}')

# Validation and Test with the best model
model.load_state_dict(torch.load(f'checkpoints/benchmark/binary_case/best_deeplabv3_model_binary_crop{TARGET_CROP}.pth'))

# Evaluate best model on validation set
best_val_loss, best_val_acc, best_val_prec, best_val_rec, best_val_f1 = validate(model, val_loader, criterion, device)
logger.info(
    f'Best Model Validation Results - Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.4f}, '
    f'Prec: {best_val_prec:.4f}, Rec: {best_val_rec:.4f}, F1: {best_val_f1:.4f}'
)

# Evaluate best model on test set
test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)
logger.info(f'Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}')

writer.close()
logger.info('Training complete.')