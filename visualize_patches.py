import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from tqdm import tqdm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

def normalize_band(band, lower_percentile=2, upper_percentile=98):
    """Normalize band data for visualization using percentiles"""
    min_val = np.percentile(band, lower_percentile)
    max_val = np.percentile(band, upper_percentile)
    normalized = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    return normalized

def create_rgb_composite(red_band, green_band, blue_band, lower_percentile=2, upper_percentile=98):
    """Create RGB composite from three bands"""
    r = normalize_band(red_band, lower_percentile, upper_percentile)
    g = normalize_band(green_band, lower_percentile, upper_percentile)
    b = normalize_band(blue_band, lower_percentile, upper_percentile)
    return np.dstack((r, g, b))

def plot_crop_data(crop_data, ax, title="Crop Data Layer"):
    """Plot crop data with a custom colormap"""
    # Create a custom colormap for crop data
    cmap = plt.cm.tab20
    norm = mcolors.Normalize(vmin=np.min(crop_data), vmax=np.max(crop_data))
    
    im = ax.imshow(crop_data, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis('off')
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('Crop Class')
    
    return im

def visualize_patch(patch_path, output_dir):
    """Visualize RGB images and crop data for a patch."""
    try:
        # Open the stacked data file
        with rasterio.open(patch_path) as src:
            # Read all bands
            data = src.read()
            
            # Get metadata
            height, width = data.shape[1], data.shape[2]
            num_bands = data.shape[0]
            
            # Determine number of timestamps (assuming 18 bands per timestamp + 1 CDL band)
            num_timestamps = (num_bands - 1) // 18
            
            # Create a figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.flatten()
            
            # Plot RGB images for the first 3 timestamps (or fewer if not available)
            for i in range(min(3, num_timestamps)):
                # Get bands for this timestamp (B04=red, B03=green, B02=blue)
                # Note: Bands are 0-indexed, so B01 is index 0, B02 is index 1, etc.
                red_idx = i * 18 + 3  # B04 is the 4th band (0-indexed)
                green_idx = i * 18 + 2  # B03 is the 3rd band
                blue_idx = i * 18 + 1  # B02 is the 2nd band
                
                # Create RGB composite
                rgb = create_rgb_composite(
                    data[red_idx], 
                    data[green_idx], 
                    data[blue_idx]
                )
                
                # Plot RGB image
                axes[i].imshow(rgb)
                axes[i].set_title(f'RGB Image - Timestamp {i+1}')
                axes[i].axis('off')
            
            # Plot crop data layer (last band)
            cdl_data = data[-1]
            plot_crop_data(cdl_data, axes[3], "Crop Data Layer")
            
            # Add overall title
            patch_name = Path(patch_path).stem
            plt.suptitle(f'Patch: {patch_name}', fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            output_path = output_dir / f"{patch_name}_visualization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print some statistics about the data
            print(f"Visualization saved to {output_path}")
            print(f"Stacked data shape: {data.shape}")
            print(f"Unique crop classes: {np.unique(cdl_data)}")
            print(f"Number of unique crop classes: {len(np.unique(cdl_data))}")
            
            # Optional: Print the crop class distribution
            unique, counts = np.unique(cdl_data, return_counts=True)
            print("\nCrop class distribution:")
            for u, c in zip(unique, counts):
                print(f"Class {u}: {c} pixels ({c/cdl_data.size*100:.2f}%)")
            
    except Exception as e:
        print(f"Error visualizing {patch_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Process each state
    states = ["illinois", "iowa", "nebraska", "north dakota13", "north dakota14"]
    
    for state in states:
        print(f"\nProcessing visualizations for {state}...")
        
        # Input and output directories
        stacked_data_dir = Path("stacked_data") / state
        visualization_dir = Path("visualizations") / state
        
        # Create output directory
        visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each patch
        for patch_file in tqdm(list(stacked_data_dir.glob("*_stacked.tif")), desc=f"Visualizing {state}"):
            visualize_patch(patch_file, visualization_dir)

if __name__ == "__main__":
    main() 