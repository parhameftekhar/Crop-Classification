import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
from process_data import determine_sensor_type, load_band, sort_timestamps, get_band_names

def load_crop_data(cdl_path, reference_metadata):
    """
    Load and align crop data with the reference band using the notebook snippet approach.
    
    Args:
        cdl_path (str): Path to the crop data file
        reference_metadata (dict): Metadata of the reference band
    
    Returns:
        numpy.ndarray: The aligned crop data
    """
    with rasterio.open(cdl_path) as cdl_src:
        # Get reference dimensions and transform
        ref_height = reference_metadata['height']
        ref_width = reference_metadata['width']
        ref_transform = reference_metadata['transform']
        
        # Get bounding pixel corners of reference band
        ul_x, ul_y = ref_transform * (0, 0)  # top-left corner
        lr_x, lr_y = ref_transform * (ref_width - 1, ref_height - 1)  # bottom-right corner
        
        # Convert world coordinates to pixel indices in CDL
        row_start, col_start = cdl_src.index(ul_x, ul_y)
        row_end, col_end = cdl_src.index(lr_x, lr_y)
        
        # Ensure correct order
        row_min = min(row_start, row_end)
        row_max = max(row_start, row_end)
        col_min = min(col_start, col_end)
        col_max = max(col_start, col_end)
        
        # Read subset of CDL corresponding to reference band's extent
        cdl_data = cdl_src.read(1, window=((row_min, row_max + 1), (col_min, col_max + 1)))
        
        return cdl_data

def stack_bands_with_cdl(state_name, patch_dir, cdl_path):
    """Stack bands from a patch with corresponding CDL data."""
    # Get all timestamp folders
    timestamp_folders = [f for f in os.listdir(patch_dir) 
                        if os.path.isdir(os.path.join(patch_dir, f)) and f.startswith('HLS.')]
    
    if not timestamp_folders:
        print(f"No timestamp folders found in {patch_dir}")
        return None
    
    # Sort timestamp folders in ascending order
    timestamp_folders = sort_timestamps(timestamp_folders)
    
    # Determine sensor type from the first folder
    try:
        sensor_type = determine_sensor_type(timestamp_folders[0])
        band_names = get_band_names(sensor_type)
    except ValueError as e:
        print(f"Error determining sensor type: {str(e)}")
        return None
    
    # Load all bands from all timestamps
    all_bands_data = []
    reference_metadata = None
    
    for timestamp_folder in timestamp_folders:
        folder_path = os.path.join(patch_dir, timestamp_folder)
        
        for band_name in band_names:
            try:
                band_data, metadata = load_band(folder_path, band_name)
                all_bands_data.append(band_data)
                
                # Save the metadata of the first band as reference
                if reference_metadata is None:
                    reference_metadata = metadata
            except Exception as e:
                print(f"Error loading {band_name} from {timestamp_folder}: {str(e)}")
                return None
    
    # Stack all bands
    stacked_bands = np.stack(all_bands_data, axis=0)
    
    # Load and align the crop data
    try:
        cdl_data = load_crop_data(cdl_path, reference_metadata)
        # Add CDL as the last band
        final_stack = np.vstack([stacked_bands, cdl_data[np.newaxis, :, :]])
        
        # Verify the number of bands
        expected_bands = len(timestamp_folders) * len(band_names) + 1  # All bands + CDL
        if final_stack.shape[0] != expected_bands:
            print(f"Warning: Expected {expected_bands} bands, but got {final_stack.shape[0]} bands")
            print(f"Number of timestamps: {len(timestamp_folders)}")
            print(f"Number of bands per timestamp: {len(band_names)}")
            print(f"Total bands: {final_stack.shape[0]}")
        
    except Exception as e:
        print(f"Error loading crop data: {str(e)}")
        return None
    
    return final_stack, reference_metadata['transform'], reference_metadata['crs']

def process_state(state_name):
    """Process all patches for a given state."""
    raw_data_dir = Path("raw_data") / state_name
    cdl_base_dir = Path("cdl_data")
    
    # For all states, including North Dakota
    cdl_path = cdl_base_dir / state_name / "clipped.TIF"
    process_patches_in_dir(raw_data_dir, cdl_path, state_name.lower())

def process_patches_in_dir(patch_dir, cdl_path, state_prefix):
    """Process all patches in a directory."""
    output_dir = Path("stacked_data") / state_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for patch_dir in tqdm(list(patch_dir.glob("patch*")), desc=f"Processing {state_prefix}"):
        try:
            stacked_data, transform, crs = stack_bands_with_cdl(state_prefix, str(patch_dir), str(cdl_path))
            if stacked_data is not None:
                # Save stacked data
                output_path = output_dir / f"{patch_dir.name}_stacked.tif"
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=stacked_data.shape[1],
                    width=stacked_data.shape[2],
                    count=stacked_data.shape[0],
                    dtype=stacked_data.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(stacked_data)
        except Exception as e:
            print(f"Error processing {patch_dir}: {str(e)}")

def main():
    # Process each state
    states = ["illinois", "IOWA", "nebraska", "North Dakota13", "North Dakota14"]
    for state in states:
        print(f"\nProcessing {state}...")
        process_state(state)

if __name__ == "__main__":
    main() 