import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def get_band_names(sensor_type):
    """
    Get the band names based on sensor type.
    
    Args:
        sensor_type (str): Either 'L30' or 'S30'
    
    Returns:
        list: List of band names to process
    """
    if sensor_type == 'L30':
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
    elif sensor_type == 'S30':
        return ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

def extract_julian_date(folder_name):
    """
    Extract Julian date from HLS folder name.
    
    Args:
        folder_name (str): Name of the HLS folder (e.g., 'HLS.S30.T16TBL.2022178T164851.v2.0')
    
    Returns:
        int: Julian date (e.g., 178 for the example above)
        int: Year (e.g., 2022 for the example above)
    
    Example:
        >>> extract_julian_date('HLS.S30.T16TBL.2022178T164851.v2.0')
        (178, 2022)
    """
    try:
        # Split the folder name by dots and get the timestamp part
        parts = folder_name.split('.')
        if len(parts) < 4:
            raise ValueError(f"Invalid folder name format: {folder_name}")
        
        # Get the timestamp part (e.g., '2022178T164851')
        timestamp = parts[3]
        
        # Extract year and Julian date
        year = int(timestamp[:4])
        julian_date = int(timestamp[4:7])
        
        return julian_date, year
        
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract Julian date from folder name: {folder_name}") from e


def load_two_bands(folder_path1, folder_path2, band1='B02', band2='B02'):
    """
    Load one band from each of two HLS data folders.
    
    Args:
        folder_path1 (str or Path): Path to the first HLS data folder
        folder_path2 (str or Path): Path to the second HLS data folder
        band1 (str): Name of the band to load from the first folder (default: 'B02')
        band2 (str): Name of the band to load from the second folder (default: 'B02')
    
    Returns:
        tuple: (band1_raster, band2_raster)
            - band1_raster: rasterio dataset for the band from the first folder
            - band2_raster: rasterio dataset for the band from the second folder
    
    Example:
        >>> folder1 = 'path/to/HLS.S30.T16TBL.2022178T164851.v2.0'
        >>> folder2 = 'path/to/HLS.S30.T16TBL.2022179T164851.v2.0'
        >>> band1_raster, band2_raster = load_two_bands(folder1, folder2, 'B02', 'B03')
        >>> # Access the data
        >>> band1_data = band1_raster.read(1)
        >>> # Get metadata
        >>> crs = band1_raster.crs
        >>> transform = band1_raster.transform
    """
    import rasterio
    from pathlib import Path
    
    folder_path1 = Path(folder_path1)
    folder_path2 = Path(folder_path2)
    
    # Construct paths to band files
    band1_path = folder_path1 / f"{band1}.tif"
    band2_path = folder_path2 / f"{band2}.tif"
    
    # Check if files exist
    if not band1_path.exists():
        raise FileNotFoundError(f"Band file not found: {band1_path}")
    if not band2_path.exists():
        raise FileNotFoundError(f"Band file not found: {band2_path}")
    
    # Open the raster datasets
    band1_raster = rasterio.open(band1_path)
    band2_raster = rasterio.open(band2_path)
    
    # Verify that both bands have the same dimensions
    if band1_raster.shape != band2_raster.shape:
        raise ValueError(f"Band dimensions do not match: {band1_raster.shape} vs {band2_raster.shape}")
    
    return band1_raster, band2_raster

def calculate_crop_percentages(stacked_data_dir="stacked_data"):
    """
    Calculate the percentage of specific crops across the entire dataset.
    
    This function iterates through all stacked data files, extracts the crop data layer
    (19th band), and calculates the percentage of each specified crop type.
    
    Args:
        stacked_data_dir (str): Path to the directory containing stacked data files
    
    Returns:
        dict: Dictionary with crop types as keys and their percentages as values
        dict: Dictionary with crop types as keys and their pixel counts as values
        int: Total number of pixels analyzed
        float: Percentage of pixels that don't belong to any of the specified crop types
    
    Example:
        >>> percentages, counts, total_pixels, other_percentage = calculate_crop_percentages()
        >>> print(f"Corn: {percentages[1]:.2f}%")
        >>> print(f"Soybean: {percentages[5]:.2f}%")
        >>> print(f"Spring Wheat: {percentages[23]:.2f}%")
        >>> print(f"Grassland/Pasture: {percentages[176]:.2f}%")
        >>> print(f"Other land cover types: {other_percentage:.2f}%")
    """
    stacked_data_dir = Path(stacked_data_dir)
    
    # Define the crop types we're interested in
    crop_types = {
        1: "Corn",
        5: "Soybean",
        23: "Spring Wheat",
        176: "Grassland/Pasture"
    }
    
    # Initialize counters
    crop_counts = {crop_id: 0 for crop_id in crop_types.keys()}
    total_pixels = 0
    
    # Find all stacked data files
    stacked_files = list(stacked_data_dir.glob("**/*_stacked.tif"))
    
    if not stacked_files:
        print(f"No stacked data files found in {stacked_data_dir}")
        return {}, {}, 0, 0.0
    
    print(f"Found {len(stacked_files)} stacked data files. Analyzing crop distribution...")
    
    # Process each stacked data file
    for file_path in tqdm(stacked_files, desc="Analyzing crop data"):
        try:
            with rasterio.open(file_path) as src:
                # Read the crop data layer (last band)
                crop_data = src.read(src.count)
                
                # Count pixels for each crop type
                for crop_id in crop_types.keys():
                    crop_counts[crop_id] += np.sum(crop_data == crop_id)
                
                # Update total pixel count
                total_pixels += crop_data.size
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Calculate percentages
    crop_percentages = {}
    for crop_id, count in crop_counts.items():
        if total_pixels > 0:
            percentage = (count / total_pixels) * 100
        else:
            percentage = 0
        crop_percentages[crop_id] = percentage
    
    # Calculate percentage of other pixels (complement of the sum of specified crop percentages)
    other_percentage = 100.0 - sum(crop_percentages.values())
    
    # Print summary
    print("\nCrop Distribution Summary:")
    print(f"Total pixels analyzed: {total_pixels:,}")
    for crop_id, crop_name in crop_types.items():
        print(f"{crop_name} (ID: {crop_id}): {crop_counts[crop_id]:,} pixels ({crop_percentages[crop_id]:.2f}%)")
    print(f"Other land cover types: {other_percentage:.2f}%")
    
    return crop_percentages, crop_counts, total_pixels, other_percentage



def calculate_crop_percentages_from_patches(patches, crop_band_index=18):
    """
    Calculate the percentage of specific crops in a set of patches.
    
    Args:
        patches (numpy.ndarray): Array of patches with shape n*h*w*C
        crop_band_index (int): Index of the band containing crop data (default: 18, which is the 19th band)
        
    Returns:
        dict: Dictionary with crop types as keys and their percentages as values
        dict: Dictionary with crop types as keys and their pixel counts as values
        int: Total number of pixels analyzed
        float: Percentage of pixels that don't belong to any of the specified crop types
    """
    # Define the crop types we're interested in
    crop_types = {
        1: "Corn",
        5: "Soybean",
        23: "Spring Wheat",
        176: "Grassland/Pasture"
    }
    
    # Initialize counters
    crop_counts = {crop_id: 0 for crop_id in crop_types.keys()}
    total_pixels = 0
    
    # Extract the crop data layer from all patches
    crop_data = patches[:, :, :, crop_band_index]
    
    # Count pixels for each crop type
    for crop_id in crop_types.keys():
        crop_counts[crop_id] = np.sum(crop_data == crop_id)
    
    # Update total pixel count
    total_pixels = crop_data.size
    
    # Calculate percentages
    crop_percentages = {}
    for crop_id, count in crop_counts.items():
        if total_pixels > 0:
            percentage = (count / total_pixels) * 100
        else:
            percentage = 0
        crop_percentages[crop_id] = percentage
    
    # Calculate percentage of other pixels (complement of the sum of specified crop percentages)
    other_percentage = 100.0 - sum(crop_percentages.values())
    
    # Print summary
    print("\nCrop Distribution Summary:")
    print(f"Total pixels analyzed: {total_pixels:,}")
    for crop_id, crop_name in crop_types.items():
        print(f"{crop_name} (ID: {crop_id}): {crop_counts[crop_id]:,} pixels ({crop_percentages[crop_id]:.2f}%)")
    print(f"Other land cover types: {other_percentage:.2f}%")
    
    return crop_percentages, crop_counts, total_pixels, other_percentage


def analyze_data_splits(data_dir):
    """
    Analyze the crop distribution in train, validation, and test splits.
    
    Args:
        data_dir (str or Path): Directory containing the .npy files for train, val, and test splits
    """
    data_dir = Path(data_dir)
    
    # Check if the directory exists
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return
    
    # Check if the required files exist
    train_file = data_dir / "train_patches.npy"
    val_file = data_dir / "val_patches.npy"
    test_file = data_dir / "test_patches.npy"
    
    if not all(file.exists() for file in [train_file, val_file, test_file]):
        print("Error: One or more required files are missing.")
        return
    
    # Load the data
    print("Loading data splits...")
    train_patches = np.load(train_file)
    val_patches = np.load(val_file)
    test_patches = np.load(test_file)
    
    # Analyze each split
    print("\n=== TRAIN SET ANALYSIS ===")
    train_percentages, train_counts, train_total, train_other = calculate_crop_percentages_from_patches(train_patches)
    
    print("\n=== VALIDATION SET ANALYSIS ===")
    val_percentages, val_counts, val_total, val_other = calculate_crop_percentages_from_patches(val_patches)
    
    print("\n=== TEST SET ANALYSIS ===")
    test_percentages, test_counts, test_total, test_other = calculate_crop_percentages_from_patches(test_patches)
    
    # Compare distributions
    print("\n=== DISTRIBUTION COMPARISON ===")
    print("Crop Type | Train % | Val % | Test % | Difference (max-min)")
    print("-" * 65)
    
    for crop_id, crop_name in {1: "Corn", 5: "Soybean", 23: "Spring Wheat", 176: "Grassland/Pasture"}.items():
        train_pct = train_percentages[crop_id]
        val_pct = val_percentages[crop_id]
        test_pct = test_percentages[crop_id]
        diff = max(train_pct, val_pct, test_pct) - min(train_pct, val_pct, test_pct)
        print(f"{crop_name:10} | {train_pct:7.2f} | {val_pct:6.2f} | {test_pct:7.2f} | {diff:7.2f}")
    
    # Other land cover types
    train_other_pct = train_other
    val_other_pct = val_other
    test_other_pct = test_other
    diff_other = max(train_other_pct, val_other_pct, test_other_pct) - min(train_other_pct, val_other_pct, test_other_pct)
    print(f"{'Other':10} | {train_other_pct:7.2f} | {val_other_pct:6.2f} | {test_other_pct:7.2f} | {diff_other:7.2f}")
    
    return {
        'train': (train_percentages, train_counts, train_total, train_other),
        'val': (val_percentages, val_counts, val_total, val_other),
        'test': (test_percentages, test_counts, test_total, test_other)
    }


def correct_pred_sign(pred, features, mean_pos_train, mean_neg_train):
    """
    Returns the prediction vector, optionally flipped to match training mean labels.

    Args:
        pred: Tensor of shape [N], values are +1 or -1
        features: Tensor of shape [N, D]
        mean_pos_train: Tensor of shape [D]
        mean_neg_train: Tensor of shape [D]

    Returns:
        corrected_pred: Tensor of shape [N], values are +1 or -1
    """
    def euclidean_dist(x, y):
        return torch.norm(x - y)

    mean_a = features[pred == 1].mean(dim=0)
    mean_b = features[pred == -1].mean(dim=0)

    d1 = euclidean_dist(mean_a, mean_pos_train) + euclidean_dist(mean_b, mean_neg_train)
    d2 = euclidean_dist(mean_a, mean_neg_train) + euclidean_dist(mean_b, mean_pos_train)

    return 1 if d1 <= d2 else -1


def plot_binary_prediction(prediction, title=None, save_path=None):
    """
    Plot a single binary prediction map.
    
    Args:
        prediction (numpy.ndarray): Binary prediction map with values -1 and 1
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(prediction, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(label='Prediction (-1 to 1)')
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_ground_truth(labels, title=None, save_path=None):
    """
    Plot ground truth labels with a custom colormap for specific crop IDs.
    
    Args:
        labels (numpy.ndarray): Ground truth labels with values -1, 1, 5, 23, 176
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
    """
    # Create a custom colormap
    colors = {
        -1: 'red',      # Background/Other
        1: 'green',     # Corn
        5: 'blue',      # Soybean
        23: 'yellow',   # Spring Wheat
        176: 'purple'   # Grassland/Pasture
    }
    
    # Create a custom colormap
    from matplotlib.colors import ListedColormap
    # Create a mapping array for the colormap
    unique_labels = sorted(colors.keys())
    color_list = [colors[label] for label in unique_labels]
    cmap = ListedColormap(color_list)
    
    plt.figure(figsize=(10, 10))
    # Create a normalized colormap
    from matplotlib.colors import BoundaryNorm
    bounds = np.array(unique_labels + [max(unique_labels) + 1]) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.imshow(labels, cmap=cmap, norm=norm)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=f'Crop {label}' if label != -1 else 'Background')
                      for label, color in colors.items()]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
