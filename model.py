import torch
import torch.nn as nn
import random
import itertools
import collections
import numpy as np
from scipy.sparse import coo_matrix
import os
import logging
from data_manager import create_sparse_structure_from_images
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from utils import correct_pred_sign

class ShallowCNN(nn.Module):
    def __init__(self, num_block, kernel_size, stride, padding, num_channel_in, num_channel_internal, num_channel_out, device):
        super(ShallowCNN, self).__init__()
        
        ## Initial block
        self.device = device
        self.block_in = nn.Sequential(
            nn.Conv2d(num_channel_in, num_channel_internal, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        ).to(self.device)

        ##
        list_of_blocks = []
        for i in range(num_block):
            list_of_blocks.append(
                nn.Sequential(
                    nn.Conv2d(num_channel_internal, num_channel_internal, kernel_size, stride=stride, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(num_channel_internal, num_channel_internal, kernel_size, stride=stride, padding=padding),
                    nn.ReLU()
                )
            )
            
        self.blocks_internal = nn.ModuleList(list_of_blocks).to(self.device)

        ## Output block
        self.block_out = nn.Sequential(
            nn.Conv2d(num_channel_internal, num_channel_out, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        ).to(self.device)
        

    def forward(self, inputImg):
        ##
        inputImg_permuted = torch.permute(inputImg, (0, 3, 1, 2))

        ##
        output = self.block_in(inputImg_permuted)
        for i, block in enumerate(self.blocks_internal):
            output = output + block(output)
        output = self.block_out(output)

        ##
        output_unpermuted = torch.permute(output, (0, 2, 3, 1))
        return output_unpermuted


class FeatureExtractor(nn.Module):
    """
    A feature extractor that combines a ShallowCNN with a learnable square matrix.
    
    This class has two main components:
    1. A ShallowCNN instance for processing input images
    2. A learnable square matrix that can be used for additional processing
    
    Args:
        num_block (int): Number of internal blocks in the ShallowCNN
        kernel_size (int): Kernel size for convolutions
        stride (int): Stride for convolutions
        padding (int): Padding for convolutions
        num_channel_in (int): Number of input channels
        num_channel_internal (int): Number of internal channels
        num_channel_out (int): Number of output channels
        device (torch.device): Device to place the model on
        matrix_size (int): Size of the square matrix (matrix will be matrix_size x matrix_size)
    """
    def __init__(self, num_block, kernel_size, stride, padding, num_channel_in, 
                 num_channel_internal, num_channel_out, device, matrix_size):
        super(FeatureExtractor, self).__init__()
        
        # Create the ShallowCNN instance
        self.cnn = ShallowCNN(
            num_block=num_block,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_channel_in=num_channel_in,
            num_channel_internal=num_channel_internal,
            num_channel_out=num_channel_out,
            device=device
        )
        
        # Create a learnable square matrix initialized as an identity matrix
        self.M = nn.Parameter(torch.eye(matrix_size, device=device))
        
        # Store the device
        self.device = device
        
        # Initialize feature centers as None
        self.positive_feature_center = None
        self.negative_feature_center = None
    
    def forward(self, inputBands):
        """
        Forward pass of the model.
        
        Args:
            inputBands (torch.Tensor): Input tensor of shape (B, H, W, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C)
        """
        # Get output from the CNN
        features = self.cnn(inputBands)  # shape: (B, H, W, C)
        
        # Apply matrix multiplication along the channel dimension
        # Reshape features to (B*H*W, C) for matrix multiplication
        B, H, W, C = features.shape
        features_reshaped = features.reshape(-1, C)  # shape: (B*H*W, C)
        
        # Apply matrix multiplication
        transformed_features = torch.matmul(features_reshaped, self.M)  # shape: (B*H*W, C)
        
        # Reshape back to original shape
        features = transformed_features.reshape(B, H, W, C)
        
        return features

    def calculate_feature_centers(self, train_loader):
        """
        Calculate the mean of features extracted from the training data, separately for
        positive (label=1) and negative (label=-1) samples.
        
        Args:
            train_loader (DataLoader): DataLoader containing training data
            
        Returns:
            tuple: (positive_feature_center, negative_feature_center)
                - positive_feature_center: Mean feature vector of positive samples
                - negative_feature_center: Mean feature vector of negative samples
        """
        self.eval()  # Set to evaluation mode
        positive_sum = 0
        negative_sum = 0
        positive_count = 0
        negative_count = 0
        
        with torch.no_grad():
            for bands, labels in train_loader:
                # Extract features
                features = self(bands)
                # Reshape features and labels to match
                features = features.reshape(-1, features.shape[-1])
                labels = labels.reshape(-1)
                
                # Separate positive and negative features
                positive_mask = labels == 1
                negative_mask = labels == -1
                
                positive_features = features[positive_mask]
                negative_features = features[negative_mask]
                
                # Accumulate sums and counts
                if len(positive_features) > 0:
                    positive_sum += positive_features.sum(dim=0)
                    positive_count += len(positive_features)
                
                if len(negative_features) > 0:
                    negative_sum += negative_features.sum(dim=0)
                    negative_count += len(negative_features)
        
        if positive_count == 0 or negative_count == 0:
            raise ValueError("No positive or negative samples found in the training data")
            
        # Calculate means
        positive_center = positive_sum / positive_count
        negative_center = negative_sum / negative_count
        
        # Set the feature centers
        self.positive_feature_center = positive_center
        self.negative_feature_center = negative_center
        
        return positive_center, negative_center

    def get_feature_centers(self):
        """
        Get the current feature centers for both positive and negative classes.
        
        Returns:
            tuple: (positive_feature_center, negative_feature_center)
                - positive_feature_center: The positive class feature center vector
                - negative_feature_center: The negative class feature center vector
        """
        if self.positive_feature_center is None or self.negative_feature_center is None:
            raise ValueError("Feature centers have not been calculated yet. Call calculate_feature_centers first.")
        return self.positive_feature_center, self.negative_feature_center


def create_feature_pairs(features, labels, num_pairs=1000):
    """
    Create random positive and negative pairs from features.
    
    Args:
        features (torch.Tensor): Features tensor of shape (N, feature_dim)
        labels (torch.Tensor): Labels tensor of shape (N,)
        num_pairs (int): Number of pairs to generate
        
    Returns:
        tuple: (pairs, pair_labels)
            - pairs: Tensor of shape (num_pairs, 2, feature_dim) containing feature pairs
            - pair_labels: Tensor of shape (num_pairs,) containing 1 for positive pairs, -1 for negative pairs
    """
    # Get indices of positive and negative samples
    pos_indices = torch.where(labels == 1)[0].tolist()
    neg_indices = torch.where(labels == -1)[0].tolist()
    
    # Initialize lists to store pairs and labels
    pairs_list = []
    pair_labels_list = []
    
    # Generate pairs
    for _ in range(num_pairs):
        # Randomly decide if this will be a positive or negative pair
        is_positive = random.random() < 0.5
        
        if is_positive:
            # Create positive pair
            if random.random() < 0.5:
                # Both from positive samples
                idx1, idx2 = random.sample(pos_indices, 2)
            else:
                # Both from negative samples
                idx1, idx2 = random.sample(neg_indices, 2)
            pair_label = 1
        else:
            # Create negative pair
            idx1 = random.choice(pos_indices)
            idx2 = random.choice(neg_indices)
            pair_label = -1
        
        # Store the pair and label
        pairs_list.append([features[idx1], features[idx2]])
        pair_labels_list.append(pair_label)
    
    # Convert lists to tensors
    pairs = torch.stack([torch.stack(pair) for pair in pairs_list])
    pair_labels = torch.tensor(pair_labels_list, device=features.device)
    
    return pairs, pair_labels


def modified_sigmoid(d, d_star, scale=1.0):
    """
    Compute the edge weight based on the function.

    Args:
        d (torch.Tensor): Input distance values.
        d_star (float): Hyperparameter d*.
        scale (float): Scaling factor for the sigmoid function.

    Returns:
        torch.Tensor: Computed edge weights.
    """
    return (-2 / (1 + torch.exp(-scale * (d - d_star)))) + 1


def create_coo_sparse_matrix(edges, weights, num_nodes=None):
    """
    Creates a sparse adjacency matrix in COO format from an edge list.
    
    Parameters:
        edges (numpy.ndarray): A (num_edges x 2) array where each row is an edge [node1, node2].
        
    Returns:
        scipy.sparse.coo_matrix: Sparse adjacency matrix.
    """
    # Extract rows and columns from edges
    row = edges[:, 0]
    col = edges[:, 1]
    
    # Determine the matrix size
    if num_nodes is None:
        num_nodes = max(row.max(), col.max()) + 1  # Infer from edge indices
    
    # Create COO sparse matrix
    sparse_matrix = coo_matrix((weights, (row, col)), shape=(num_nodes, num_nodes))
    
    return sparse_matrix


class CropClassifierTree:
    """
    A tree-based classifier that uses binary classifiers to assign crop labels to image pixels.
    
    Args:
        checkpoints_dir (str): Path to the directory containing binary classifier checkpoints
        crop_order (list): List of crop IDs specifying the order in which crops should be processed
        img_height (int, optional): Height of the image patches. Defaults to 112.
        img_width (int, optional): Width of the image patches. Defaults to 112.
        window_size (int, optional): Size of the window for creating sparse structure. Defaults to 30.
    """
    def __init__(self, checkpoints_dir: str, crop_order: list, img_height: int = 112, img_width: int = 112, window_size: int = 30):
        self.checkpoints_dir = checkpoints_dir
        self.crop_order = crop_order
        self.classifiers = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize model parameters (same as in training)
        self.model_params = {
            'num_block': 4,
            'kernel_size': 9,
            'stride': 1,
            'padding': 4,
            'num_channel_in': 18,
            'num_channel_internal': 18,
            'num_channel_out': 18,
            'matrix_size': 18,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Setup sparse structure
        sparse_image_obj = create_sparse_structure_from_images(img_height, img_width, window_size, self.model_params['device'])
        self.order = sparse_image_obj['order']
        self.reverse_order = torch.argsort(self.order).cpu().numpy()  # Add reverse order
        self.edges = sparse_image_obj['edges'].cpu().numpy()
        self.edge_i, self.edge_j = self.edges[:, 0], self.edges[:, 1]
        
        self._load_classifiers()
    
    def _load_classifiers(self):
        """
        Load all binary classifiers from the checkpoints directory.
        Each classifier is loaded based on the crop_order.
        """
        self.logger.info(f"Loading classifiers from {self.checkpoints_dir}")
        for crop_id in self.crop_order:
            checkpoint_path = os.path.join(self.checkpoints_dir, f'crop{crop_id}_vs_all.pth')
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"Checkpoint not found for crop {crop_id}: {checkpoint_path}")
                continue
                
            # Load the entire model
            classifier = torch.load(checkpoint_path, weights_only=False)
            classifier.eval()
            
            self.classifiers[crop_id] = classifier
            self.logger.info(f"Loaded classifier for crop {crop_id}")
    
    def run_binary_classifiers(self, image):
        """
        Predict crop labels for each pixel in the input image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (H, W, C)
            
        Returns:
            dict: Dictionary containing predictions for each crop classifier
                  Keys are crop IDs, values are prediction tensors of shape (H, W)
            dict: Dictionary containing eigen vectors for each crop classifier
                  Keys are crop IDs, values are eigen vector tensors of shape (H, W)
        """
        # Dictionary to store predictions and eigen vectors for each classifier
        predictions = {}
        eigen_vectors = {}
        
        with torch.no_grad():
            # Process each binary classifier in the specified order
            for crop_id in self.crop_order:
                if crop_id not in self.classifiers:
                    continue
                    
                classifier = self.classifiers[crop_id]
                
                # Get features from the classifier
                features = classifier(image.unsqueeze(0)).squeeze(0)  # Remove batch dimension
                
                # Get feature centers for this classifier
                positive_center, negative_center = classifier.get_feature_centers()
                
                # Reshape and reorder features
                features = features.reshape(-1, features.shape[-1])[self.order, :]
                
                # Calculate distances and weights
                features_i, features_j = features[self.edge_i], features[self.edge_j]
                distances = ((features_i - features_j) ** 2).sum(dim=1)
                weights = modified_sigmoid(distances, d_star=1.0, scale=1)
                
                # Create sparse matrix and compute Laplacian
                coo_mat = create_coo_sparse_matrix(self.edges, weights.cpu().numpy())
                sparse_adjacency = coo_mat + coo_mat.T
                
                degree = sparse_adjacency.sum(axis=1).A1
                D = diags(degree)
                L = D - sparse_adjacency
                
                # Compute eigenvector and prediction
                _, eigen_vector = eigsh(L, k=1, which='SA', tol=1e-7)
                pred = np.sign(eigen_vector).flatten()
                sign = correct_pred_sign(pred, features, positive_center, negative_center)
                pred = pred * sign
                eigen_vector = eigen_vector * sign  # Apply sign to eigen_vector
                
                # Reorder prediction and eigen vector back to original order
                pred = pred[self.reverse_order]
                eigen_vector = eigen_vector[self.reverse_order]
                
                # Reshape prediction and eigen vector back to image shape
                predictions[crop_id] = pred.reshape(image.shape[0], image.shape[1])
                eigen_vectors[crop_id] = eigen_vector.reshape(image.shape[0], image.shape[1])  # Reshape eigen vector
        
        return predictions, eigen_vectors

    def combine_binary_predictions(self, predictions):
        """
        Combine binary predictions in a tree-like manner according to crop_order.
        For each crop in order:
        - Pixels predicted as +1 are assigned that crop's ID
        - Pixels predicted as -1 are passed to the next crop
        - If a pixel is -1 for all crops, it's assigned -1 (background)
        
        Args:
            predictions (dict): Dictionary of binary predictions from run_binary_classifiers() method
                              Keys are crop IDs, values are prediction tensors of shape (H, W)
        
        Returns:
            numpy.ndarray: Combined prediction map with crop IDs
        """
        # Initialize the final prediction map with -1 (background)
        final_pred = np.full_like(next(iter(predictions.values())), -1)
        
        # Process each crop in order
        for crop_id in self.crop_order:
            if crop_id not in predictions:
                continue
                
            # Get binary predictions for this crop
            binary_pred = predictions[crop_id]
            
            # For pixels that are still background (-1) and predicted as +1 for this crop,
            # assign them this crop's ID
            mask = (final_pred == -1) & (binary_pred == 1)
            final_pred[mask] = crop_id
        
        return final_pred


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron to be applied on the output of FeatureExtractor,
    corresponding to self-loops in the Laplacian matrix.
    
    Args:
        num_features (int): Number of input features from FeatureExtractor
        num_layers (int): Number of hidden layers, each with dimension equal to num_features
        device (torch.device): Device to place the model on
    """
    def __init__(self, num_features, num_layers=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MLP, self).__init__()
        self.device = device
        
        # Build the layers
        layers = []
        prev_dim = num_features
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, num_features),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = num_features
        
        # Output layer - single value for classification
        layers.append(nn.Linear(prev_dim, 1))
        # Add Softplus to ensure positive outputs
        layers.append(nn.Softplus())
        
        self.network = nn.Sequential(*layers).to(self.device)
    
    def forward(self, features):
        """
        Forward pass of the MLP.
        
        Args:
            features (torch.Tensor): Input features of shape (B, H, W, C) or (N, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, 1) or (N, 1) with positive values
        """
        # Store original shape if 4D input
        if len(features.shape) == 4:
            B, H, W, C = features.shape
            features = features.reshape(-1, C)
        else:
            B, H, W = None, None, None
        
        # Pass through network
        output = self.network(features)
        
        # Reshape back if input was 4D
        if B is not None:
            output = output.reshape(B, H, W, 1)
        
        return output


class EnsembleCNN(nn.Module):
    """
    An ensemble model that combines multiple pre-trained models and adds a shallow CNN on top.
    The pre-trained models are loaded from a checkpoint directory and their weights are frozen.
    The outputs of these models are concatenated along the channel dimension and processed
    by a trainable shallow CNN followed by a fully connected layer for multi-crop classification.
    
    Args:
        checkpoint_dir (str): Directory containing the pre-trained model checkpoints
        num_block (int): Number of blocks in the top shallow CNN
        kernel_size (int): Kernel size for the top CNN convolutions
        stride (int): Stride for the top CNN convolutions
        padding (int): Padding for the top CNN convolutions
        num_channel_internal (int): Number of internal channels in the top CNN
        num_channel_out (int): Number of output channels from the top CNN before final fully connected layer
        device (torch.device): Device to place the model on
    """
    def __init__(self, checkpoint_dir, num_block=2, kernel_size=3, stride=1, padding=1, 
                 num_channel_internal=64, num_channel_out=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(EnsembleCNN, self).__init__()
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.models = nn.ModuleList()
        self.load_models()
        
        # Freeze the weights of the pre-trained models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Calculate input channels for the top CNN based on number of models and their output channels
        # Assuming each model outputs the same number of channels (taken from first model)
        if len(self.models) > 0:
            num_channel_in = len(self.models) * self.models[0].cnn.block_out[0].out_channels
        else:
            raise ValueError("No models loaded from checkpoint directory")
        
        # Define the top shallow CNN
        self.top_cnn = ShallowCNN(
            num_block=num_block,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_channel_in=num_channel_in,
            num_channel_internal=num_channel_internal,
            num_channel_out=num_channel_internal,  # Changed to use internal channels as output for CNN
            device=device
        )
        
        # Add a final fully connected layer to reduce to 5 channels for multi-class classification
        self.fc = nn.Conv2d(num_channel_internal, 5, kernel_size=1, stride=1, padding=0).to(device)
    
    def load_models(self):
        """Load all pre-trained models from the checkpoint directory."""
        if not os.path.exists(self.checkpoint_dir):
            raise ValueError(f"Checkpoint directory {self.checkpoint_dir} does not exist")
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
            try:
                model = torch.load(checkpoint_path, weights_only=False)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"Loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading model from {checkpoint_path}: {e}")
        
        if len(self.models) == 0:
            raise ValueError(f"No valid model checkpoints found in {self.checkpoint_dir}")
    
    def forward(self, x):
        """
        Forward pass of the ensemble model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, 1) after processing through ensemble, top CNN, and final fully connected layer
        """
        # Get outputs from all pre-trained models
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Concatenate outputs along channel dimension
        concatenated = torch.cat(outputs, dim=-1)
        
        # Pass through the top CNN
        cnn_output = self.top_cnn(concatenated)
        
        # Permute dimensions for the final fully connected layer
        cnn_output = cnn_output.permute(0, 3, 1, 2)
        
        # Pass through the final fully connected layer to get 1 channel output
        final_output = self.fc(cnn_output)
        
        return final_output
