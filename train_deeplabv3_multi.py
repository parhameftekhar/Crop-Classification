from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from data_manager import create_modified_crop_labels
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log')
logger = logging.getLogger(__name__)

# Model setup
weights = deeplabv3.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(num_classes=5)

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

# Dataset class
class CropDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data.astype(float)
        self.transform = transform
        
        # Fixed mapping for known labels
        self.label_map = {
            -1: 0,  # background
            1: 1,   # corn
            5: 2,   # soybean
            23: 3,  # spring wheat
            176: 4  # grassland/pasture
        }
        self.num_classes = 5  # 5 classes including background
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the image and label
        image = self.data[idx, :, :, :-1]  # All bands except last one (label)
        label = self.data[idx, :, :, -1]   # Last band is the label
        
        # Scale first 18 bands by 0.0001 and clip to [0,1]
        image[:, :, :18] = np.clip(image[:, :, :18] * 0.0001, 0, 1)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        
        # Map labels to 0 to 4 range
        label = np.vectorize(self.label_map.get)(label)
        label = torch.from_numpy(label).long()
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Load data
data_dir = './training_data'
train_data = np.load(os.path.join(data_dir, 'train_patches.npy'))
valid_data = np.load(os.path.join(data_dir, 'val_patches.npy'))
test_data = np.load(os.path.join(data_dir, 'test_patches.npy'))

unchanged_crops = [1, 5, 23, 176]
train_data = create_modified_crop_labels(train_data, unchanged_crops=unchanged_crops)
valid_data = create_modified_crop_labels(valid_data, unchanged_crops=unchanged_crops)
test_data = create_modified_crop_labels(test_data, unchanged_crops=unchanged_crops)

# Create datasets
train_dataset = CropDataset(train_data)
val_dataset = CropDataset(valid_data)
test_dataset = CropDataset(test_data)

# Log dataset information
logger.info(f'Number of classes: {train_dataset.num_classes}')
logger.info(f'Label mapping: {train_dataset.label_map}')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Log dataset sizes and sample shapes
logger.info(f'Training samples: {len(train_dataset)}')
logger.info(f'Validation samples: {len(val_dataset)}')
logger.info(f'Test samples: {len(test_dataset)}')

# Get and log shape of a single sample
sample_image, sample_label = next(iter(train_loader))
logger.info(f'Image shape: {sample_image.shape}')
logger.info(f'Label shape: {sample_label.shape}')
logger.info(f'Unique labels in sample: {torch.unique(sample_label)}')

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_pixels = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.permute(0, 3, 1, 2).to(device)  # Change to (B, C, H, W)
        labels = labels.to(device)
        
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total_pixels += labels.numel()
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(predicted == labels).float().mean().item():.4f}'
        })
    
    return total_loss / len(train_loader), correct / total_pixels

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_pixels = 0
    
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.permute(0, 3, 1, 2).to(device)  # Change to (B, C, H, W)
            labels = labels.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(predicted == labels).float().mean().item():.4f}'
            })
    
    return total_loss / len(val_loader), correct / total_pixels

# Training loop
num_epochs = 300
best_val_acc = 0.0

epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
for epoch in epoch_pbar:
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    scheduler.step()
    
    epoch_pbar.set_postfix({
        'train_loss': f'{train_loss:.4f}',
        'train_acc': f'{train_acc:.4f}',
        'val_loss': f'{val_loss:.4f}',
        'val_acc': f'{val_acc:.4f}'
    })
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        logger.info(f'New best model saved with validation accuracy: {val_acc:.4f}')

# Load best model for testing
model.load_state_dict(torch.load('best_model.pth'))

# Test the model
test_loss, test_acc = validate(model, test_loader, criterion, device)
logger.info(f'Test Results:')
logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}') 