import torch
import torch.nn as nn
import os
import numpy as np
from model import ShallowCNN
from torch.utils.data import Dataset, DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler to save logs to a file
log_file = 'training_shallow_cnn_on_eigenvectors.txt'
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.info(f"Logging initialized. Logs will be saved to {log_file}")

class EigenvectorDataset(Dataset):
    def __init__(self, data_dir, subset='train'):
        self.data_dir = data_dir  # Use base directory directly, no subset subfolder
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Directory {self.data_dir} does not exist")
        
        # Define file names based on subset
        self.eigen_file = f"{subset}_eigen_vectors.npy"
        self.label_file = f"{subset}_labels.npy"
        
        self.eigen_path = os.path.join(self.data_dir, self.eigen_file)
        self.label_path = os.path.join(self.data_dir, self.label_file)
        
        # Check if files exist
        if not os.path.exists(self.eigen_path):
            raise ValueError(f"Eigenvector file {self.eigen_path} does not exist")
        if not os.path.exists(self.label_path):
            raise ValueError(f"Label file {self.label_path} does not exist")
        
        # Load data
        self.eigen_data = np.load(self.eigen_path)
        self.label_data = np.load(self.label_path)
        
        # Check if data shapes are compatible
        if len(self.eigen_data) != len(self.label_data):
            raise ValueError(f"Mismatch in number of samples between eigenvector data ({len(self.eigen_data)}) and labels ({len(self.label_data)}) in {subset} set")
        
        logger.info(f"Loaded {len(self.eigen_data)} samples from {self.eigen_file} and {self.label_file} in {self.data_dir}")

    def __len__(self):
        return len(self.eigen_data)

    def __getitem__(self, idx):
        try:
            # Get eigenvector and label data for the given index
            eigenvector = torch.from_numpy(self.eigen_data[idx]).float()
            label = torch.from_numpy(self.label_data[idx]).float()

            # Reshape to add channel dimension if needed
            if len(eigenvector.shape) == 2:
                eigenvector = eigenvector.unsqueeze(-1)  # Add channel dimension

            return eigenvector, label  # Change to (C, H, W) format
        except Exception as e:
            logger.error(f"Error loading data at index {idx} from {self.eigen_file} or {self.label_file}: {str(e)}")
            raise

class ShallowCNNWithFC(nn.Module):
    def __init__(self, num_block, kernel_size, stride, padding, num_channel_in, num_channel_internal, num_channel_out_cnn, num_channel_out_fc, device):
        super(ShallowCNNWithFC, self).__init__()
        self.device = device
        self.cnn = ShallowCNN(
            num_block=num_block,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_channel_in=num_channel_in,
            num_channel_internal=num_channel_internal,
            num_channel_out=num_channel_out_cnn,
            device=device
        )
        # Fully connected layer to reduce to specified output channels for classification
        self.fc = nn.Conv2d(num_channel_out_cnn, num_channel_out_fc, kernel_size=1, stride=1, padding=0).to(device)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)  # Change to (B, C, H, W) for Conv2d
        x = self.fc(x)
        return x  # Return in (B, C, H, W) format for CrossEntropyLoss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if len(labels.shape) == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)  # Remove channel dimension if present
            labels = labels.long()  # Convert to long for CrossEntropyLoss

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {running_loss/10:.4f}')
                running_loss = 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if len(labels.shape) == 4 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                labels = labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy for multi-class
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total * 100
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_shallow_cnn_eigenvectors.pth')
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if len(labels.shape) == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            labels = labels.long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy for multi-class
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    test_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100
    logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

def main():
    # Configuration
    base_data_dir = 'binary_classifiers_outputs_deprecated'
    device = torch.cuda.is_available() and 'cuda' or 'cpu'
    logger.info(f"Using device: {device}")

    # Model parameters
    model_params = {
        'num_block': 4,
        'kernel_size': 9,
        'stride': 1,
        'padding': 4,
        'num_channel_in': 4,  # Adjust based on your eigenvector data
        'num_channel_internal': 32,
        'num_channel_out_cnn': 32,  # Output channels for ShallowCNN
        'num_channel_out_fc': 5,    # Output channels for fully connected layer (for 5-class classification)
        'device': device
    }

    # Training parameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001

    # Create datasets and dataloaders
    train_dataset = EigenvectorDataset(base_data_dir, subset='train')
    val_dataset = EigenvectorDataset(base_data_dir, subset='val')
    test_dataset = EigenvectorDataset(base_data_dir, subset='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, criterion, optimizer
    model = ShallowCNNWithFC(**model_params).to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model with validation
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Load best model for testing
    model.load_state_dict(torch.load('best_shallow_cnn_eigenvectors.pth'))
    logger.info("Loaded best model for testing")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluate_model(model, test_loader, criterion, device)

    # Save the final model
    torch.save(model.state_dict(), 'shallow_cnn_eigenvectors_final.pth')
    logger.info("Final model saved as shallow_cnn_eigenvectors_final.pth")

if __name__ == '__main__':
    main() 