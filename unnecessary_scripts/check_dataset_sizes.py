# check_dataset_sizes.py

import numpy as np
import os
from data_manager import setup_training_loader

def check_dataset_sizes():
    print("Checking dataset sizes...")
    
    # Configuration (same as in training scripts)
    TARGET_CROP = 176  # The crop ID we're training to detect
    UNCHANGED_CROPS = [1, 5, 23, 176]  # List of unchanged crops
    
    # Check if data files exist
    train_path = './training_data/train_patches.npy'
    val_path = './training_data/val_patches.npy'
    test_path = './training_data/test_patches.npy'
    
    print(f"Checking data files:")
    print(f"  Train data: {train_path} - {'EXISTS' if os.path.exists(train_path) else 'NOT FOUND'}")
    print(f"  Val data: {val_path} - {'EXISTS' if os.path.exists(val_path) else 'NOT FOUND'}")
    print(f"  Test data: {test_path} - {'EXISTS' if os.path.exists(test_path) else 'NOT FOUND'}")
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("❌ Some data files are missing!")
        return
    
    # Load raw data and check sizes
    print(f"\nRaw dataset sizes (before filtering):")
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)
    
    print(f"  Train patches: {train_data.shape[0]:,} patches")
    print(f"  Val patches: {val_data.shape[0]:,} patches")
    print(f"  Test patches: {test_data.shape[0]:,} patches")
    print(f"  Total patches: {train_data.shape[0] + val_data.shape[0] + test_data.shape[0]:,} patches")
    
    # Check patch dimensions
    print(f"\nPatch dimensions:")
    print(f"  Height: {train_data.shape[1]} pixels")
    print(f"  Width: {train_data.shape[2]} pixels")
    print(f"  Channels: {train_data.shape[3]} bands")
    
    # Check filtered dataset sizes (after applying the same filtering as training)
    print(f"\nFiltered dataset sizes (after applying training filters):")
    
    try:
        # Setup training loader (this applies the same filtering as training)
        train_loader = setup_training_loader(
            path_to_train_data=train_path,
            unchanged_crops=UNCHANGED_CROPS,
            target_crops=[TARGET_CROP],
            train_batch_size=16,
            crop_band_index=18,
            device='cpu',  # Use CPU for this check
            ignore_crops=None,
            min_ratio=0.1,
            max_ratio=0.9
        )
        
        val_loader = setup_training_loader(
            path_to_train_data=val_path,
            unchanged_crops=UNCHANGED_CROPS,
            target_crops=[TARGET_CROP],
            train_batch_size=16,
            crop_band_index=18,
            device='cpu',
            ignore_crops=None,
            min_ratio=0.1,
            max_ratio=0.9
        )
        
        test_loader = setup_training_loader(
            path_to_train_data=test_path,
            unchanged_crops=UNCHANGED_CROPS,
            target_crops=[TARGET_CROP],
            train_batch_size=16,
            crop_band_index=18,
            device='cpu',
            ignore_crops=None,
            min_ratio=0.1,
            max_ratio=0.9
        )
        
        print(f"  Train batches: {len(train_loader):,} batches")
        print(f"  Val batches: {len(val_loader):,} batches")
        print(f"  Test batches: {len(test_loader):,} batches")
        
        # Calculate total samples
        train_samples = len(train_loader) * 16  # batch_size = 16
        val_samples = len(val_loader) * 16
        test_samples = len(test_loader) * 16
        
        print(f"  Train samples: {train_samples:,} samples")
        print(f"  Val samples: {val_samples:,} samples")
        print(f"  Test samples: {test_samples:,} samples")
        print(f"  Total samples: {train_samples + val_samples + test_samples:,} samples")
        
        # Check data distribution
        print(f"\nData distribution:")
        print(f"  Train: {train_samples/(train_samples + val_samples + test_samples)*100:.1f}%")
        print(f"  Val: {val_samples/(train_samples + val_samples + test_samples)*100:.1f}%")
        print(f"  Test: {test_samples/(train_samples + val_samples + test_samples)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error setting up data loaders: {e}")
        return
    
    print(f"\n✅ Dataset size check completed!")

if __name__ == "__main__":
    check_dataset_sizes()
