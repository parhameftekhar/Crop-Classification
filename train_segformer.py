import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import logging
from datetime import datetime
from data_manager import create_modified_crop_labels

# Setup logging
def setup_logging():
    # Use fixed name for log file in current directory
    log_file = 'segformer.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return log_file

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

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_pixels = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            # Move data to device
            images = images.permute(0, 3, 1, 2).to(device)  # Change from (B, H, W, C) to (B, C, H, W)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            logits = outputs.logits
            # For accuracy calculation, upsample logits to match original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            _, predicted = torch.max(upsampled_logits, 1)
            correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(predicted == labels).float().mean().item():.4f}'
            })
            
            # Log batch metrics every 10 batches
            # if batch_idx % 10 == 0:
            #     logging.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {(predicted == labels).float().mean().item():.4f}')
                
        except Exception as e:
            logging.error(f'Error in training batch {batch_idx}: {str(e)}')
            raise
    
    return total_loss / len(train_loader), correct / total_pixels

def analyze_label_distribution(dataset):
    """Analyze the distribution of labels in the dataset."""
    label_counts = {i: 0 for i in range(5)}  # 5 classes
    total_pixels = 0
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        unique, counts = torch.unique(label, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            label_counts[u] += c
            total_pixels += c
    
    print("\nLabel Distribution:")
    for label, count in label_counts.items():
        percentage = (count / total_pixels) * 100
        print(f"Class {label}: {count} pixels ({percentage:.2f}%)")

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_pixels = 0
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(5, 5, device=device)
    
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                # Move data to device
                images = images.permute(0, 3, 1, 2).to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss
                
                # Calculate accuracy
                logits = outputs.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                _, predicted = torch.max(upsampled_logits, 1)
                
                # Update confusion matrix
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
                correct += (predicted == labels).sum().item()
                total_pixels += labels.numel()
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(predicted == labels).float().mean().item():.4f}'
                })
                
            except Exception as e:
                logging.error(f'Error in validation batch {batch_idx}: {str(e)}')
                raise
    
    # Calculate per-class accuracy
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    logging.info("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        logging.info(f"Class {i}: {acc:.4f}")
    
    # Log confusion matrix
    logging.info("\nConfusion Matrix:")
    logging.info(confusion_matrix.cpu().numpy())
    
    return total_loss / len(val_loader), correct / total_pixels

def main():
    try:
        # Setup logging
        log_file = setup_logging()
        logging.info("Starting training process")
        logging.info(f"Log file: {log_file}")
        
        # Log system info
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Load data
        logging.info("Loading datasets...")
        train_data = np.load('./training_data/train_patches.npy')
        valid_data = np.load('./training_data/val_patches.npy')
        test_data = np.load('./training_data/test_patches.npy')
        
        logging.info(f"Training samples: {len(train_data)}")
        logging.info(f"Validation samples: {len(valid_data)}")
        logging.info(f"Test samples: {len(test_data)}")

        unchanged_crops = [1, 5, 23, 176]
        train_data = create_modified_crop_labels(train_data, unchanged_crops=unchanged_crops)
        valid_data = create_modified_crop_labels(valid_data, unchanged_crops=unchanged_crops)
        test_data = create_modified_crop_labels(test_data, unchanged_crops=unchanged_crops)

        # Create datasets
        train_dataset = CropDataset(train_data)
        val_dataset = CropDataset(valid_data)
        test_dataset = CropDataset(test_data)
        
        # Analyze label distribution
        logging.info("\nAnalyzing training set label distribution:")
        analyze_label_distribution(train_dataset)
        logging.info("\nAnalyzing validation set label distribution:")
        analyze_label_distribution(val_dataset)

        batch_size = 8
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Initialize model
        logging.info("Initializing Segformer model from scratch...")
        # Comment out pretrained model loading
        # model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/mit-b0",
        #     num_labels=5,
        #     ignore_mismatched_sizes=True
        # )

        # Initialize model from scratch
        # Create configuration
        config = SegformerConfig(
            num_labels=5,
            image_size=224,  # Assuming your input size is 224x224
            num_channels=18,  # 18 input channels for satellite data
            depths=[2, 2, 2, 2],  # Number of transformer blocks in each stage
            sr_ratios=[8, 4, 2, 1],  # Spatial reduction ratios
            hidden_sizes=[32, 64, 160, 256],  # Hidden sizes for each stage
            num_attention_heads=[1, 2, 5, 8],  # Number of attention heads
            drop_path_rate=0.1,  # Drop path rate for stochastic depth
            semantic_loss_ignore_index=255,  # Ignore index for loss calculation
            loss_type="weighted_ce",  # Use weighted cross entropy
            label_smoothing=0.1  # Add label smoothing
        )
        
        # Initialize model with custom config
        model = SegformerForSemanticSegmentation(config)
        
        # Calculate class weights based on inverse frequency
        def calculate_class_weights(dataset):
            label_counts = {i: 0 for i in range(5)}
            total_pixels = 0
            
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                unique, counts = torch.unique(label, return_counts=True)
                for u, c in zip(unique.tolist(), counts.tolist()):
                    label_counts[u] += c
                    total_pixels += c
            
            # Calculate weights as inverse of frequency
            weights = torch.tensor([
                total_pixels / (5 * label_counts[i]) if label_counts[i] > 0 else 1.0
                for i in range(5)
            ], device=device)
            
            return weights
        
        # Calculate and set class weights
        class_weights = calculate_class_weights(train_dataset)
        model.classifier.weight = nn.Parameter(class_weights.unsqueeze(1) * model.classifier.weight)
        logging.info(f"Class weights: {class_weights.tolist()}")
        
        # Initialize weights using Kaiming initialization
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        logging.info("Model initialized with Kaiming initialization")

        # Modify the first conv layer to accept 18 input channels instead of 3
        logging.info("Modifying model architecture for 18 input channels...")
        old_proj = model.segformer.encoder.patch_embeddings[0].proj
        new_proj = nn.Conv2d(
            in_channels=18,  # Changed from 3 to 18
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )

        # Initialize the new projection layer with Kaiming initialization
        nn.init.kaiming_normal_(new_proj.weight, mode='fan_out', nonlinearity='relu')
        if new_proj.bias is not None:
            nn.init.zeros_(new_proj.bias)

        # Replace the old projection layer with the new one
        model.segformer.encoder.patch_embeddings[0].proj = new_proj

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logging.info(f"Model moved to device: {device}")

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        logging.info("Training setup completed")

        # Training loop
        num_epochs = 100
        best_val_acc = 0.0

        # Create checkpoints directory if it doesn't exist
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
            logging.info("Created checkpoints directory")

        # Add tqdm for epochs
        logging.info("Starting training loop...")
        epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
        for epoch in epoch_pbar:
            try:
                # Training
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
                
                # Validation
                val_loss, val_acc = validate(model, val_loader, device)
                
                # Update learning rate
                scheduler.step()
                
                # Log epoch metrics
                logging.info(f"Epoch {epoch + 1}/{num_epochs}:")
                logging.info(f"  Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                logging.info(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_acc:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint_path = 'checkpoints/best_segformer_model.pth'
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
                    logging.info(f"Checkpoint saved to: {checkpoint_path}")
                    
            except Exception as e:
                logging.error(f"Error in epoch {epoch + 1}: {str(e)}")
                raise

        # Load best model for testing
        logging.info("Loading best model for final testing...")
        model.load_state_dict(torch.load('checkpoints/best_segformer_model.pth'))

        # Test the model
        logging.info("Running final test evaluation...")
        test_loss, test_acc = validate(model, test_loader, device)
        logging.info(f"Final Test Results:")
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main() 