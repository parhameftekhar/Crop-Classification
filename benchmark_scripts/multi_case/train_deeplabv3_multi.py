# train_deeplabv3_multiclass.py

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from torch.utils.data import DataLoader
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
        logging.FileHandler(f'logs/benchmark/multi_case/training_deeplabv3_multiclass_{NUM_CLASSES}classes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# TensorBoard writer
writer = SummaryWriter(log_dir=f'tensorboard/benchmark/multi_case/deeplabv3')

# Model setup
logger.info(f'Initializing DeepLabV3_ResNet50 model for {NUM_CLASSES}-class classification')
model = deeplabv3_resnet50(num_classes=NUM_CLASSES)

# Modify the first convolution layer to accept 18 input channels
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

# Metrics functions (standardized across benchmark scripts)
def accumulate_confusion_matrix(outputs, labels, num_classes, confusion_matrix):
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
    tp = confusion_matrix['tp']
    fp = confusion_matrix['fp']
    fn = confusion_matrix['fn']
    ious, precisions, recalls, f1_scores = [], [], [], []
    for class_id in range(num_classes):
        union = tp[class_id] + fp[class_id] + fn[class_id]
        ious.append(tp[class_id] / union if union > 0 else 0.0)
        precisions.append(tp[class_id] / (tp[class_id] + fp[class_id]) if (tp[class_id] + fp[class_id]) > 0 else 0.0)
        recalls.append(tp[class_id] / (tp[class_id] + fn[class_id]) if (tp[class_id] + fn[class_id]) > 0 else 0.0)
        p, r = precisions[-1], recalls[-1]
        f1_scores.append(2 * (p * r) / (p + r) if (p + r) > 0 else 0.0)
    return np.mean(ious), ious, np.mean(precisions), precisions, np.mean(recalls), recalls, np.mean(f1_scores), f1_scores

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return (predicted == labels).sum().item() / labels.numel()

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss, total_accuracy, batches = 0, 0, 0
    confusion_matrix = {'tp': np.zeros(num_classes), 'fp': np.zeros(num_classes), 'fn': np.zeros(num_classes)}
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(outputs, labels)
        accumulate_confusion_matrix(outputs, labels, num_classes, confusion_matrix)
        total_loss += loss.item()
        total_accuracy += accuracy
        batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
    mean_iou, _, _, _, _, _, _, _ = compute_metrics_from_confusion_matrix(confusion_matrix, num_classes)
    return total_loss / batches, total_accuracy / batches, mean_iou

# Validation function
def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_accuracy, batches = 0, 0, 0
    confusion_matrix = {'tp': np.zeros(num_classes), 'fp': np.zeros(num_classes), 'fn': np.zeros(num_classes)}
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            accuracy = calculate_accuracy(outputs, labels)
            accumulate_confusion_matrix(outputs, labels, num_classes, confusion_matrix)
            total_loss += loss.item()
            total_accuracy += accuracy
            batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
    m_iou, ious, m_prec, precs, m_rec, recs, m_f1, f1s = compute_metrics_from_confusion_matrix(confusion_matrix, num_classes)
    return total_loss/batches, total_accuracy/batches, m_iou, ious, precs, recs, f1s

# Training loop
num_epochs = 100
best_val_miou = 0.0
logger.info(f'Starting training for {num_epochs} epochs')
for epoch in range(num_epochs):
    train_loss, train_acc, train_miou = train_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES)
    val_loss, val_acc, val_miou, v_ious, v_precs, v_recs, v_f1s = validate(model, val_loader, criterion, device, NUM_CLASSES)
    curr_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    # Logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('mIoU/train', train_miou, epoch)
    writer.add_scalar('mIoU/val', val_miou, epoch)
    writer.add_scalar('LearningRate', curr_lr, epoch)
    for i, (iou, prec, rec, f1) in enumerate(zip(v_ious, v_precs, v_recs, v_f1s)):
        writer.add_scalar(f'PerClass_IoU/class_{i}', iou, epoch)
        writer.add_scalar(f'PerClass_Precision/class_{i}', prec, epoch)
        writer.add_scalar(f'PerClass_Recall/class_{i}', rec, epoch)
        writer.add_scalar(f'PerClass_F1/class_{i}', f1, epoch)
    
    logger.info(f'Epoch {epoch+1}/{num_epochs} - Val mIoU: {val_miou:.4f}, Accuracy: {val_acc:.4f}')
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        torch.save(model.state_dict(), f'checkpoints/benchmark/multi_case/best_deeplabv3_model_multiclass_{NUM_CLASSES}classes.pth')
        logger.info(f'New best model saved with validation mIoU: {val_miou:.4f}')

# Test phase
model.load_state_dict(torch.load(f'checkpoints/benchmark/multi_case/best_deeplabv3_model_multiclass_{NUM_CLASSES}classes.pth'))
t_loss, t_acc, t_miou, t_ious, t_precs, t_recs, t_f1s = validate(model, test_loader, criterion, device, NUM_CLASSES)
logger.info(f'Test Results: Loss: {t_loss:.4f}, Accuracy: {t_acc:.4f}, mIoU: {t_miou:.4f}')
for i in range(NUM_CLASSES):
    logger.info(f'Class {i} - IoU: {t_ious[i]:.4f}, F1: {t_f1s[i]:.4f}')

writer.close()
logger.info('Training complete.')