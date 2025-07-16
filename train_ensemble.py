import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse
import logging
from model import EnsembleCNN
import numpy as np

# Set up logging - will be configured in main() after args are parsed
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train EnsembleCNN for multi-crop classification')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing pre-trained model checkpoints')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='./ensemble', help='Directory to save model checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num-block', type=int, default=5, help='Number of blocks in top CNN')
    parser.add_argument('--kernel-size', type=int, default=7, help='Kernel size for top CNN')
    parser.add_argument('--stride', type=int, default=1, help='Stride for top CNN')
    parser.add_argument('--padding', type=int, default=3, help='Padding for top CNN')
    parser.add_argument('--num-channel-internal', type=int, default=64, help='Number of internal channels in top CNN')
    return parser.parse_args()

class CropDataset(torch.utils.data.Dataset):
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

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs, output_dir):
    model.train()
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total_pixels = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()

            if (i + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {(predicted == labels).float().mean().item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total_pixels
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

        # Validate after each epoch
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(output_dir, f'ensemble_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_pixels = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total_pixels
    model.train()
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    # Configure logging after args are available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training_ensemble_binary_classifiers.log')),
            logging.StreamHandler()
        ]
    )
    logger.info('Logging configured.')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define data transforms - adjust based on your needs
    transform = transforms.Compose([
        # Add other transformations as needed
    ])

    # Load data from .npy files
    logger.info('Loading data...')
    train_data = np.load(os.path.join(args.data_dir, 'train_patches.npy'))
    valid_data = np.load(os.path.join(args.data_dir, 'val_patches.npy'))
    test_data = np.load(os.path.join(args.data_dir, 'test_patches.npy'))

    unchanged_crops = [1, 5, 23, 176]
    from data_manager import create_modified_crop_labels
    train_data = create_modified_crop_labels(train_data, unchanged_crops=unchanged_crops)
    valid_data = create_modified_crop_labels(valid_data, unchanged_crops=unchanged_crops)
    test_data = create_modified_crop_labels(test_data, unchanged_crops=unchanged_crops)

    # Create datasets
    train_dataset = CropDataset(train_data, transform=transform)
    valid_dataset = CropDataset(valid_data, transform=transform)
    test_dataset = CropDataset(test_data, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f'Training samples: {len(train_dataset)}')
    logger.info(f'Validation samples: {len(valid_dataset)}')
    logger.info(f'Test samples: {len(test_dataset)}')

    # Initialize model
    model = EnsembleCNN(
        checkpoint_dir=args.checkpoint_dir,
        num_block=args.num_block,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        num_channel_internal=args.num_channel_internal,
        device=device
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info('Starting training...')
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs, args.output_dir)
    logger.info('Training completed.')

    # Test the model
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main() 