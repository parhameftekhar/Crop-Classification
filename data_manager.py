import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import collections

def split_into_patches(array, h, w):
    """
    Split a large array of shape H*W*C into small non-overlapping patches of shape n*h*w*C.
    
    Args:
        array (numpy.ndarray): Input array of shape H*W*C
        h (int): Height of each patch
        w (int): Width of each patch
        
    Returns:
        numpy.ndarray: Array of patches with shape n*h*w*C, where n is the number of patches
        
    Example:
        >>> # For a 1000x1000x6 array and patch size of 64x64
        >>> patches = split_into_patches(large_array, 64, 64)
        >>> print(f"Number of patches: {patches.shape[0]}")
        >>> print(f"Patch shape: {patches.shape[1:3]}")
    """
    # Get the dimensions of the input array
    H, W, C = array.shape
    
    # Calculate how many patches can fit in each dimension
    n_h = H // h
    n_w = W // w
    
    # Calculate the actual dimensions that will be used (discarding margins)
    H_used = n_h * h
    W_used = n_w * w
    
    # Crop the array to remove margins
    array_cropped = array[:H_used, :W_used, :]
    
    # Reshape the array into patches
    patches = array_cropped.reshape(n_h, h, n_w, w, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(-1, h, w, C)
    
    return patches


def load_and_split_data(file_path, h, w):
    """
    Load a raster file and split it into patches.
    
    Args:
        file_path (str or Path): Path to the raster file
        h (int): Height of each patch
        w (int): Width of each patch
        
    Returns:
        numpy.ndarray: Array of patches with shape n*h*w*C
    """
    with rasterio.open(file_path) as src:
        # Read all bands
        data = src.read()
        # Transpose to H*W*C format
        data = np.transpose(data, (1, 2, 0))
    
    # Split into patches
    patches = split_into_patches(data, h, w)
    
    return patches


def process_all_files(input_dir, output_dir, h, w, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Process all raster files in a directory, split them into patches, stack them together,
    and then split into train, validation, and test sets using scikit-learn.
    
    Args:
        input_dir (str or Path): Directory containing raster files
        output_dir (str or Path): Directory to save patches
        h (int): Height of each patch
        w (int): Width of each patch
        train_ratio (float): Proportion of data to use for training (default: 0.7)
        val_ratio (float): Proportion of data to use for validation (default: 0.15)
        test_ratio (float): Proportion of data to use for testing (default: 0.15)
        seed (int): Random seed for reproducibility (default: 42)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all raster files
    raster_files = list(input_dir.glob("**/*.tif"))
    
    # List to store all patches
    all_patches = []
    
    # Process each file and collect patches
    for file_path in tqdm(raster_files, desc="Processing files"):
        try:
            # Load and split the data
            patches = load_and_split_data(file_path, h, w)
            
            # Add to the list of all patches
            all_patches.append(patches)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Stack all patches together
    if not all_patches:
        print("No patches were generated. Check your input files.")
        return
    
    stacked_patches = np.vstack(all_patches)
    print(f"Total number of patches: {stacked_patches.shape[0]}")
    
    # First split: separate test set
    test_size = test_ratio
    train_val_size = 1 - test_size
    
    # Calculate the ratio for the second split (validation from train+val)
    val_ratio_adjusted = val_ratio / train_val_size
    
    # First split: train+val and test
    train_val_patches, test_patches = train_test_split(
        stacked_patches, 
        test_size=test_size, 
        random_state=seed
    )
    
    # Second split: train and validation
    train_patches, val_patches = train_test_split(
        train_val_patches, 
        test_size=val_ratio_adjusted, 
        random_state=seed
    )
    
    # Save the splits
    np.save(output_dir / "train_patches.npy", train_patches)
    np.save(output_dir / "val_patches.npy", val_patches)
    np.save(output_dir / "test_patches.npy", test_patches)
    
    print(f"Train set: {train_patches.shape[0]} patches")
    print(f"Validation set: {val_patches.shape[0]} patches")
    print(f"Test set: {test_patches.shape[0]} patches")
    
    return train_patches, val_patches, test_patches


class CropBinaryDataset(Dataset):
    """
    A PyTorch Dataset for binary crop classification.
    
    This dataset creates pixel-wise binary labels (+1 and -1).
    Pixels belonging to target crops are labeled as +1,
    while all other pixels are labeled as -1.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        target_crops (list): List of crop IDs to be labeled as +1
        crop_band_index (int): Index of the band containing crop data (default: 18)
        input_bands (list): List of band indices to use as input features (default: all bands except crop band)
        transform (callable, optional): Optional transform to be applied on a sample
        device (str or torch.device): Device to move tensors to (default: 'cpu')
    """
    def __init__(self, patches, target_crops, crop_band_index=18, 
                 input_bands=None, transform=None, device='cpu'):
        self.patches = torch.from_numpy(patches).float()
        self.target_crops = target_crops
        self.crop_band_index = crop_band_index
        self.transform = transform
        self.device = device
        
        # Define all important crops
        self.important_crops = [1, 5, 23, 176]  # Corn, Soybean, Spring Wheat, Grassland/Pasture
        
        # If input_bands is not specified, use all bands except the crop band
        if input_bands is None:
            self.input_bands = list(range(self.patches.shape[-1]))
            self.input_bands.remove(crop_band_index)
        else:
            self.input_bands = input_bands
        
        # Calculate pixel-wise binary labels for each patch
        self.labels = self._calculate_pixel_labels()
        
        # Move data to device
        self.patches = self.patches.to(device)
        self.labels = self.labels.to(device)
        
        print(f"Dataset loaded with {len(self)} patches")
        print(f"Total pixels: {np.prod(self.labels.shape)}")
        print(f"Positive pixels (+1): {torch.sum(self.labels == 1).item()}")
        print(f"Negative pixels (-1): {torch.sum(self.labels == -1).item()}")
    
    def _calculate_pixel_labels(self):
        """
        Calculate pixel-wise binary labels for each patch.
        
        Returns:
            torch.Tensor: Tensor of pixel-wise labels with shape (n, h, w) where each pixel is +1 or -1
        """
        # Extract the crop data layer
        crop_data = self.patches[:, :, :, self.crop_band_index]
        
        # Initialize binary labels array with the same shape as crop_data
        binary_labels = torch.zeros_like(crop_data, dtype=torch.float32)
        
        # For each patch
        for i in range(len(self.patches)):
            # Create a mask for target crops
            target_mask = torch.zeros_like(crop_data[i], dtype=torch.bool)
            for crop_id in self.target_crops:
                target_mask = target_mask | (crop_data[i] == crop_id)
            
            # Assign labels: +1 for target crops, -1 for everything else
            binary_labels[i] = torch.where(target_mask, 1.0, -1.0)
        
        return binary_labels
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Get the patch and its pixel-wise labels
        patch = self.patches[idx]
        pixel_labels = self.labels[idx]
        
        # Extract only the specified input bands
        features = patch[:, :, self.input_bands]
        
        # Scale features by 0.0001 and clip to [0,1] range
        features = features * 0.0001  # Scale by 0.0001
        features = torch.clamp(features, min=0.0, max=1.0)  # Clip to [0,1] range
        
        # Apply transform if specified
        if self.transform:
            features = self.transform(features)
        
        return features, pixel_labels


def find_balanced_crop_combination(patches, important_crops, crop_band_index=18, 
                                  min_ratio=0.4, max_ratio=0.6, ignore_crops=None):
    
    """
    Find the best combination of target crops that results in the most balanced patches.
    
    This function tries different combinations of the important crops and evaluates
    which combination gives the most patches with a balanced distribution of positive and negative pixels.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        important_crops (list): List of all important crop IDs to consider
        crop_band_index (int): Index of the band containing crop data (default: 18)
        input_bands (list): List of band indices to use as input features
        min_ratio (float): Minimum acceptable ratio of positive to total pixels (default: 0.4)
        max_ratio (float): Maximum acceptable ratio of positive to total pixels (default: 0.6)
        ignore_crops (list, optional): List of crop IDs to ignore when calculating the ratio.
                                      These crops will be excluded from both positive and total counts.
        
    Returns:
        tuple: (best_target_crops, good_patches_percentage, all_results)
            - best_target_crops: List of crop IDs that gives the most balanced patches
            - good_patches_percentage: Percentage of patches with balanced distribution for the best combination
            - all_results: Dictionary with all combinations and their good patch percentages
    """
    import itertools
    
    # Extract the crop data layer
    crop_data = patches[:, :, :, crop_band_index]
    
    # Initialize results dictionary
    all_results = {}
    
    # Try all possible combinations of 1 to len(important_crops) crops
    for r in range(1, len(important_crops) + 1):
        for combo in itertools.combinations(important_crops, r):
            # Calculate binary labels for this combination
            binary_labels = np.zeros_like(crop_data, dtype=np.float32)
            
            # For each patch
            for i in range(len(patches)):
                # Create a mask for target crops
                target_mask = np.zeros_like(crop_data[i], dtype=bool)
                for crop_id in combo:
                    target_mask = target_mask | (crop_data[i] == crop_id)
                
                # Assign labels: +1 for target crops, -1 for everything else
                binary_labels[i] = np.where(target_mask, 1.0, -1.0)
            
            # Create a mask for crops to ignore
            ignore_mask = np.zeros_like(crop_data, dtype=bool)
            if ignore_crops is not None:
                for crop_id in ignore_crops:
                    ignore_mask = ignore_mask | (crop_data == crop_id)
            
            # Calculate the ratio of positive to total pixels for each patch, excluding ignored crops
            total_pixels_per_patch = np.sum(~ignore_mask, axis=(1, 2))
            positive_pixels_per_patch = np.sum((binary_labels == 1) & (~ignore_mask), axis=(1, 2))
            
            # Avoid division by zero
            ratios_per_patch = np.zeros_like(positive_pixels_per_patch, dtype=float)
            valid_patches = total_pixels_per_patch > 0
            ratios_per_patch[valid_patches] = positive_pixels_per_patch[valid_patches] / total_pixels_per_patch[valid_patches]
            
            # Count patches with ratio within the acceptable range
            good_patches = np.sum((ratios_per_patch >= min_ratio) & (ratios_per_patch <= max_ratio))
            good_patches_percentage = (good_patches / len(patches)) * 100
            
            # Store the result
            all_results[combo] = good_patches_percentage
            
            print(f"Combination {combo}: {good_patches} good patches out of {len(patches)} ({good_patches_percentage:.2f}%)")
    
    # Find the combination with the highest percentage of good patches
    best_combo = min(all_results.items(), key=lambda x: np.abs(x[1] - 50))
    
    print(f"\nBest combination: {best_combo[0]}")
    print(f"Percentage of good patches: {best_combo[1]:.2f}%")
    
    return list(best_combo[0]), best_combo[1], all_results




def create_modified_crop_labels(patches, unchanged_crops, other_value=-1, crop_band_index=18):
    """
    Create modified patches where specified crops keep their original values
    and all other crops are changed to a specified value.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        unchanged_crops (list): List of crop IDs to keep their original values
        other_value (int or float): Value to assign to all crops not in unchanged_crops
        crop_band_index (int): Index of the band containing crop data (default: 18)
        
    Returns:
        numpy.ndarray: Modified patches with shape (n, h, w, c)
    """
    # Create a copy of the patches to modify
    modified_patches = patches.copy()
    
    # Extract the crop data layer
    crop_data = patches[:, :, :, crop_band_index]
    
    # Create a mask for crops to keep unchanged
    unchanged_mask = np.zeros_like(crop_data, dtype=bool)
    for crop_id in unchanged_crops:
        unchanged_mask = unchanged_mask | (crop_data == crop_id)
    
    # Change all other crops to the specified value
    modified_patches[:, :, :, crop_band_index][~unchanged_mask] = other_value
    
    return modified_patches


def filter_balanced_patches(patches, target_crops, crop_band_index=18, 
                           min_ratio=0.4, max_ratio=0.6, ignore_crops=None):
    """
    Filter patches to keep only those with a balanced distribution of positive and negative pixels.
    
    This function uses the same logic as find_balanced_crop_combination to identify patches
    with a balanced distribution of positive and negative pixels, and returns only those patches.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        target_crops (list): List of crop IDs to be labeled as +1
        crop_band_index (int): Index of the band containing crop data (default: 18)
        min_ratio (float): Minimum acceptable ratio of positive to total pixels (default: 0.4)
        max_ratio (float): Maximum acceptable ratio of positive to total pixels (default: 0.6)
        ignore_crops (list, optional): List of crop IDs to ignore when calculating the ratio.
                                      These crops will be excluded from both positive and total counts.
        
    Returns:
        tuple: (filtered_patches, good_indices)
            - filtered_patches: Array of patches with shape (m, h, w, c) containing only the good patches
            - good_indices: Indices of the good patches in the original array
    """
    # Extract the crop data layer
    crop_data = patches[:, :, :, crop_band_index]
    
    # Calculate binary labels
    binary_labels = np.zeros_like(crop_data, dtype=np.float32)
    
    # For each patch
    for i in range(len(patches)):
        # Create a mask for target crops
        target_mask = np.zeros_like(crop_data[i], dtype=bool)
        for crop_id in target_crops:
            target_mask = target_mask | (crop_data[i] == crop_id)
        
        # Assign labels: +1 for target crops, -1 for everything else
        binary_labels[i] = np.where(target_mask, 1.0, -1.0)
    
    # Create a mask for crops to ignore
    ignore_mask = np.zeros_like(crop_data, dtype=bool)
    if ignore_crops is not None:
        for crop_id in ignore_crops:
            ignore_mask = ignore_mask | (crop_data == crop_id)
    
    # Calculate the ratio of positive to total pixels for each patch, excluding ignored crops
    total_pixels_per_patch = np.sum(~ignore_mask, axis=(1, 2))
    positive_pixels_per_patch = np.sum((binary_labels == 1) & (~ignore_mask), axis=(1, 2))
    
    # Avoid division by zero
    ratios_per_patch = np.zeros_like(positive_pixels_per_patch, dtype=float)
    valid_patches = total_pixels_per_patch > 0
    ratios_per_patch[valid_patches] = positive_pixels_per_patch[valid_patches] / total_pixels_per_patch[valid_patches]
    
    # Find patches with ratio within the acceptable range
    good_indices = np.where((ratios_per_patch >= min_ratio) & (ratios_per_patch <= max_ratio))[0]
    
    # Filter the patches
    filtered_patches = patches[good_indices]
    
    print(f"Filtered {len(patches)} patches to {len(filtered_patches)} good patches ({len(filtered_patches)/len(patches)*100:.2f}%)")
    
    return filtered_patches, good_indices



def calculate_gini_impurity(patches, ignore_crops=None, crop_band_index=18):
    """
    Calculate the Gini impurity for the dataset.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        ignore_crops (list, optional): List of crop IDs to ignore in the calculation
        crop_band_index (int): Index of the band containing crop data (default: 18)
        
    Returns:
        float: Gini impurity value
    """
    # Extract the crop data layer
    crop_data = patches[:, :, :, crop_band_index]
    
    # Flatten the crop data to get all pixels
    all_pixels = crop_data.reshape(-1)
    
    # If ignore_crops is specified, create a mask for pixels to ignore
    if ignore_crops is not None:
        ignore_mask = np.zeros_like(all_pixels, dtype=bool)
        for crop_id in ignore_crops:
            ignore_mask = ignore_mask | (all_pixels == crop_id)
        # Keep only the pixels we want to consider
        all_pixels = all_pixels[~ignore_mask]
    
    # Get unique crop IDs and their counts
    unique_crops, counts = np.unique(all_pixels, return_counts=True)
    
    # Calculate the total number of pixels
    total_pixels = len(all_pixels)
    
    # Calculate the probability of each crop
    probabilities = counts / total_pixels
    print(probabilities)
    # Calculate Gini impurity: 1 - sum(p_i^2)
    gini = 1 - np.sum(probabilities ** 2)
    
    return gini



def find_best_binary_split(patches, target_crops, ignore_crops=None, crop_band_index=18):
    """
    Find the best binary split of target crops based on Gini impurity reduction.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        target_crops (list): List of crop IDs to consider for splitting
        ignore_crops (list, optional): List of crop IDs to ignore in the calculation
        crop_band_index (int): Index of the band containing crop data (default: 18)
        
    Returns:
        tuple: (best_left_split, best_right_split, best_gini_reduction)
            - best_left_split: List of crops in the left split
            - best_right_split: List of crops in the right split
            - best_gini_reduction: The amount of Gini impurity reduction achieved
    """
    
    # Calculate the original Gini impurity
    original_gini = calculate_gini_impurity(patches, ignore_crops, crop_band_index)
    
    # Initialize variables to track the best split
    best_gini_reduction = 0
    best_left_split = None
    best_right_split = None
    
    # Generate all possible binary splits
    for r in range(1, len(target_crops)):
        for left_split in itertools.combinations(target_crops, r):
            left_split = list(left_split)
            right_split = [crop for crop in target_crops if crop not in left_split]
            
            # Calculate Gini for left split
            left_gini = calculate_gini_impurity(
                patches,
                ignore_crops=ignore_crops + right_split if ignore_crops else right_split,
                crop_band_index=crop_band_index
            )
            
            # Calculate Gini for right split
            right_gini = calculate_gini_impurity(
                patches,
                ignore_crops=ignore_crops + left_split if ignore_crops else left_split,
                crop_band_index=crop_band_index
            )
            
            # Calculate number of pixels for each split
            crop_data = patches[:, :, :, crop_band_index]
            
            # Create masks for left and right splits
            left_mask = np.zeros_like(crop_data, dtype=bool)
            for crop in left_split:
                left_mask = left_mask | (crop_data == crop)
            
            right_mask = np.zeros_like(crop_data, dtype=bool)
            for crop in right_split:
                right_mask = right_mask | (crop_data == crop)
            
            # Count pixels in each split
            left_pixels = np.sum(left_mask)
            right_pixels = np.sum(right_mask)
            total_pixels = left_pixels + right_pixels
            
            # Calculate weighted average Gini
            if total_pixels > 0:  # Avoid division by zero
                weighted_gini = (left_pixels/total_pixels) * left_gini + (right_pixels/total_pixels) * right_gini
                
                # Calculate Gini reduction
                gini_reduction = original_gini - weighted_gini
                
                # Update best split if this one is better
                if gini_reduction > best_gini_reduction:
                    best_gini_reduction = gini_reduction
                    best_left_split = left_split
                    best_right_split = right_split
    
    return best_left_split, best_right_split, best_gini_reduction


class CropSplitNode:
    def __init__(self, crops, gini_reduction=None, left=None, right=None):
        """
        A node in the crop split tree.
        
        Args:
            crops (list): List of crop IDs in this node
            gini_reduction (float, optional): Gini reduction achieved by this split
            left (CropSplitNode, optional): Left child node
            right (CropSplitNode, optional): Right child node
        """
        self.crops = crops
        self.gini_reduction = gini_reduction
        self.left = left
        self.right = right
        self.is_leaf = left is None and right is None

def build_crop_split_tree(patches, target_crops, ignore_crops=None, crop_band_index=18):
    """
    Build a binary tree of crop splits, continuing until each leaf contains a single crop.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape (n, h, w, c)
        target_crops (list): List of crop IDs to consider for splitting
        ignore_crops (list, optional): List of crop IDs to ignore in the calculation
        crop_band_index (int): Index of the band containing crop data (default: 18)
        
    Returns:
        CropSplitNode: Root node of the built tree
    """
    def build_node(current_crops, current_ignore_crops):
        # If we have only one crop, create a leaf node
        if len(current_crops) == 1:
            return CropSplitNode(crops=current_crops)
        
        # Find the best binary split for current crops
        left_split, right_split, gini_reduction = find_best_binary_split(
            patches=patches,
            target_crops=current_crops,
            ignore_crops=current_ignore_crops,
            crop_band_index=crop_band_index
        )
        print(left_split)
        print(right_split)
        print("#########################")
        # Create a new node with the split
        node = CropSplitNode(
            crops=current_crops,
            gini_reduction=gini_reduction
        )
        
        # Recursively build left and right subtrees
        # For left subtree, ignore right split crops
        left_ignore = current_ignore_crops + right_split if current_ignore_crops else right_split
        node.left = build_node(left_split, left_ignore)
        
        # For right subtree, ignore left split crops
        right_ignore = current_ignore_crops + left_split if current_ignore_crops else left_split
        node.right = build_node(right_split, right_ignore)
        
        return node
    
    # Start building from the root
    root = build_node(target_crops, ignore_crops)
    return root


def print_tree(node, level=0):
    """
    Print the tree structure in a readable format.
    
    Args:
        node (CropSplitNode): Current node to print
        level (int): Current level in the tree
    """
    prefix = "  " * level
    if node.is_leaf:
        print(f"{prefix}Leaf: {node.crops}")
    else:
        print(f"{prefix}Node: {node.crops}")
        print(f"{prefix}Gini reduction: {node.gini_reduction:.4f}")
        print(f"{prefix}Left:")
        print_tree(node.left, level + 1)
        print(f"{prefix}Right:")
        print_tree(node.right, level + 1)


#############################
def MortonFromPosition(position):
    """Convert integer (x,y,z) positions to Morton codes

    Args:
      positions: Nx3 np array (will be cast to int32)

    Returns:
      Length-N int64 np array
    """

    position = np.asarray(position, dtype=np.int32)
    morton_code = np.zeros(len(position), dtype=np.int64)
    coeff = np.asarray([4, 2, 1], dtype=np.int64)
    for b in range(21):
        morton_code |= ((position & (1 << b)) << (2 * b)) @ coeff
    assert morton_code.dtype == np.int64
    return morton_code

def PositionFromMorton(morton_code):
    """Convert int64 Morton code to int32 (x,y,z) positions

    Args:
      morton_code: int64 np array

    Returns:
      Nx3 int32 np array
    """

    morton_code = np.asarray(morton_code, dtype=np.int64)
    position = np.zeros([len(morton_code), 3], dtype=np.int32)
    shift = np.array([2, 1, 0], dtype=np.int64)
    for b in range(21):
        position |= ((morton_code[:, np.newaxis] >> shift[np.newaxis, :]) >> (2 * b)
                     ).astype(np.int32) & (1 << b)
    assert position.dtype == np.int32
    return position

def hash_to_index(hash_val, hash_table):
    if hash_val in hash_table:
        return hash_table[hash_val]
    else:
        return -1
hash_to_index_vec = np.vectorize(hash_to_index)


def create_sparse_structure_from_images(img_height, img_width, window_size, device):

    # CREATE NODES
    xindex, yindex = np.meshgrid(np.arange(img_width), np.arange(img_height))
    xy_location = np.stack([yindex, xindex], axis=2).reshape(-1, 2)
    hash_code = MortonFromPosition(
        np.concatenate([xy_location, np.zeros((xy_location.shape[0], 1))], axis=1)
    )
    order = np.argsort(hash_code)

    ## MUCH REMEMBER ORDER
    xy_location = xy_location[order]
    hash_code = hash_code[order]
    hash_code_map = {code:i for i, code in enumerate(hash_code)}
        
    # ADD EDGES
    m = np.arange(window_size)-window_size//2
    edge_delta = np.array(
        list(itertools.product(m, m)),
        dtype=np.int32)
    max_edge_type = edge_delta.shape[0]

    #
    possible_node_i_indx = np.arange(xy_location.shape[0], dtype=np.int32)[:, np.newaxis] + np.zeros([1, max_edge_type], dtype=np.int32)
    possible_node_i_indx = possible_node_i_indx.flatten()
    possible_edge_types  = np.repeat(np.arange(0, max_edge_type).reshape(1, max_edge_type), xy_location.shape[0], axis=0).flatten()

    #
    possible_node_j_location = xy_location[:, np.newaxis, :] + edge_delta[np.newaxis, :, :]
    possible_node_j_location = possible_node_j_location.reshape([-1, 2])
    possible_node_j_hash = MortonFromPosition(
        np.concatenate([possible_node_j_location, np.zeros((possible_node_j_location.shape[0], 1))], axis=1)
    )
    possible_node_j_indx = hash_to_index_vec(possible_node_j_hash, hash_code_map)

    #
    valid_edges = possible_node_j_indx >= 0
    node_i_indx = possible_node_i_indx[valid_edges]
    node_j_indx = possible_node_j_indx[valid_edges]
    edges_type  = possible_edge_types[valid_edges]
    edges = np.stack([
        node_i_indx, node_j_indx
    ], axis=1)

    ## Control meta information here
    sparse_image_obj = collections.OrderedDict()
    sparse_image_obj['order']          = torch.from_numpy(order).to(device)
    sparse_image_obj["node_locations"] = torch.from_numpy(xy_location).to(device)
    sparse_image_obj['edges']          = torch.from_numpy(edges).to(device)
    sparse_image_obj["edges_type"]     = torch.from_numpy(edges_type).to(device)
    
    return sparse_image_obj


def setup_training_loader(
    path_to_train_data: str,
    unchanged_crops: list,
    target_crops: list,
    train_batch_size: int = 8,
    crop_band_index: int = 18,
    device: str = 'cuda',
    ignore_crops: list = None,
    min_ratio: float = 0.4,
    max_ratio: float = 0.6
) -> DataLoader:
    """
    Automatically sets up the training data loader with all necessary preprocessing steps.
    
    Args:
        path_to_train_data (str): Path to the training data numpy file
        unchanged_crops (list): List of crop IDs that should remain unchanged
        target_crops (list): List of crop IDs to focus on
        train_batch_size (int): Batch size for the training loader
        crop_band_index (int): Index of the crop band in the data
        device (str): Device to load the data on ('cuda' or 'cpu')
        ignore_crops (list, optional): List of crop IDs to ignore when filtering balanced patches
        min_ratio (float): Minimum acceptable ratio of positive to total pixels (default: 0.4)
        max_ratio (float): Maximum acceptable ratio of positive to total pixels (default: 0.6)
        
    Returns:
        DataLoader: Configured training data loader
    """
    # Load training data
    train_data = np.load(path_to_train_data)
    
    # Modify crop labels
    train_data = create_modified_crop_labels(
        train_data,
        unchanged_crops=unchanged_crops,
        other_value=-1,
        crop_band_index=crop_band_index
    )
    
    # Filter balanced patches
    filtered_train_data, _ = filter_balanced_patches(
        train_data,
        target_crops=target_crops,
        ignore_crops=ignore_crops,
        min_ratio=min_ratio,
        max_ratio=max_ratio
    )
    
    # Create dataset and dataloader
    train_dataset = CropBinaryDataset(
        filtered_train_data,
        target_crops=target_crops,
        crop_band_index=crop_band_index,
        device=device
    )
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size)
    
    return train_loader


def setup_validation_loader(
    path_to_valid_data: str,
    unchanged_crops: list,
    target_crops: list,
    crop_band_index: int = 18,
    device: str = 'cuda'
) -> DataLoader:
    """
    Automatically sets up the validation data loader with necessary preprocessing steps.
    Note: Validation loader always uses batch_size=1 and doesn't filter balanced patches.
    
    Args:
        path_to_valid_data (str): Path to the validation data numpy file
        unchanged_crops (list): List of crop IDs that should remain unchanged
        target_crops (list): List of crop IDs to focus on
        crop_band_index (int): Index of the crop band in the data
        device (str): Device to load the data on ('cuda' or 'cpu')
        
    Returns:
        DataLoader: Configured validation data loader with batch_size=1
    """
    # Load validation data
    valid_data = np.load(path_to_valid_data)
    
    # Modify crop labels
    valid_data = create_modified_crop_labels(
        valid_data,
        unchanged_crops=unchanged_crops,
        other_value=-1,
        crop_band_index=crop_band_index
    )
    
    # Create dataset and dataloader (no filtering for validation)
    valid_dataset = CropBinaryDataset(
        valid_data,
        target_crops=target_crops,
        crop_band_index=crop_band_index,
        device=device
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1)  # Always batch_size=1 for validation
    
    return valid_loader

