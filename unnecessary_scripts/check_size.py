import numpy as np
import os

def check_dataset_sizes():
    print("Checking dataset sizes...")
    
    # Paths to the data files
    train_path = '/home/jovyan/project/Crop-Classification/training_data/train_patches.npy'
    val_path = '/home/jovyan/project/Crop-Classification/training_data/val_patches.npy'
    test_path = '/home/jovyan/project/Crop-Classification/training_data/test_patches.npy'
    
    # Check if files exist
    files = {
        'Training': train_path,
        'Validation': val_path,
        'Test': test_path
    }
    
    print("Dataset sizes:")
    print("-" * 50)
    
    total_patches = 0
    for dataset_name, file_path in files.items():
        if os.path.exists(file_path):
            # Load just the shape without loading the full data
            with open(file_path, 'rb') as f:
                # Read the numpy header to get shape info
                version = np.lib.format.read_magic(f)
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                
                num_patches = shape[0]
                patch_height = shape[1]
                patch_width = shape[2]
                num_channels = shape[3]
                
                print(f"{dataset_name:12} set: {num_patches:>8,} patches")
                print(f"             Shape: {shape}")
                print(f"             Patch size: {patch_height} x {patch_width} x {num_channels}")
                print()
                
                total_patches += num_patches
        else:
            print(f"{dataset_name:12} set: File not found!")
    
    print("-" * 50)
    print(f"Total patches: {total_patches:>8,}")
    
    # Calculate percentages
    if total_patches > 0:
        train_data = np.load(train_path, mmap_mode='r')
        val_data = np.load(val_path, mmap_mode='r')
        test_data = np.load(test_path, mmap_mode='r')
        
        train_pct = (train_data.shape[0] / total_patches) * 100
        val_pct = (val_data.shape[0] / total_patches) * 100
        test_pct = (test_data.shape[0] / total_patches) * 100
        
        print(f"\nData distribution:")
        print(f"Training:   {train_pct:5.1f}%")
        print(f"Validation: {val_pct:5.1f}%")
        print(f"Test:       {test_pct:5.1f}%")

if __name__ == "__main__":
    check_dataset_sizes()