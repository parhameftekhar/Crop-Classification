import torch
import segmentation_models_pytorch as smp
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleInferenceDataset(Dataset):
    """
    A simple dataset for inference that returns spectral bands only.
    No filtering is applied here to ensure all patches in the split are processed.
    """
    def __init__(self, patches, crop_band_index=18):
        self.patches = patches
        self.crop_band_index = crop_band_index
        # Use only the first 18 spectral bands (0-17)
        self.input_bands = [i for i in range(patches.shape[-1]) if i != crop_band_index]
            
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        # Load one patch from disk (if using mmap) or memory
        patch = self.patches[idx].astype(np.float32)
        features = patch[:, :, self.input_bands]
        
        # Scale features and clip to [0,1]
        features = features * 0.0001
        features = np.clip(features, 0.0, 1.0)
        
        # Change shape from (H, W, C) to (C, H, W) for the model
        features = np.transpose(features, (2, 0, 1))
        return torch.from_numpy(features)

def get_model(checkpoint_path, device):
    """Initializes a Unet logic consistent with training."""
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=18,
        classes=2,
        activation=None,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
    )
    # Load the best state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'unet_binary_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # The four crops identified for the benchmark
    crops = [1, 5, 23, 176]
    splits = {
        'train': './training_data/train_patches.npy',
        'val': './training_data/val_patches.npy',
        'test': './training_data/test_patches.npy'
    }
    
    for split_name, split_path in splits.items():
        logger.info(f"Processing split: {split_name}")
        if not os.path.exists(split_path):
            logger.error(f"  Data split not found: {split_path}")
            continue
            
        # Load the raw patches. mmap_mode='r' prevents loading everything into RAM at once.
        raw_data = np.load(split_path, mmap_mode='r')
        num_samples = raw_data.shape[0]
        h, w = raw_data.shape[1], raw_data.shape[2]
        logger.info(f"  Split contains {num_samples} samples of size {h}x{w}")
        
        dataset = SimpleInferenceDataset(raw_data, crop_band_index=18)
        # Using a small batch size for GPU memory safety
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # Pre-allocate high-dimensional numpy array (N, H, W, 4)
        # We store as int8 to save space while representing +1 and -1 accurately.
        final_stacked_preds = np.zeros((num_samples, h, w, len(crops)), dtype=np.int8)
        
        # Extract and save ground truth labels (from band 18)
        # Mapping consistent with binary_classifiers_outputs: 
        # 0: Other, 1: Corn(1), 2: Soybean(5), 3: Spring Wheat(23), 4: Grassland(176)
        logger.info(f"  Extracting and mapping ground truth labels...")
        raw_gt = raw_data[:, :, :, 18].copy()
        mapped_gt = np.zeros_like(raw_gt, dtype=np.int8)
        for idx, crop_id in enumerate(crops):
            mapped_gt[raw_gt == crop_id] = idx + 1
            
        label_output_name = f'unet_labels_{split_name}.npy'
        label_save_path = os.path.join(output_dir, label_output_name)
        np.save(label_save_path, mapped_gt)
        logger.info(f"  [SUCCESS] Ground truth labels (mapped to 0-4) saved to {label_save_path}")
        logger.info(f"  Unique values in label array: {np.unique(mapped_gt)}")

        for crop_idx, crop in enumerate(crops):
            checkpoint_path = f'checkpoints/benchmark/binary_case/best_unet_model_binary_crop{crop}.pth'
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"    Checkpoint not found for Crop {crop}. Skipping...")
                continue
                
            logger.info(f"    Loading best Unet model for Crop {crop}...")
            model = get_model(checkpoint_path, device)
            
            current_idx = 0
            with torch.no_grad():
                for images in tqdm(loader, desc=f"    Inference [Crop {crop}]"):
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Convert labels: 0 -> -1, 1 -> +1
                    binary_preds = (predicted.float() * 2 - 1).cpu().numpy().astype(np.int8)
                    
                    batch_size = binary_preds.shape[0]
                    final_stacked_preds[current_idx : current_idx + batch_size, :, :, crop_idx] = binary_preds
                    current_idx += batch_size
            
            # Clean up GPU memory before next crop/split
            del model
            torch.cuda.empty_cache()
            
        # Save the multi-crop results for this split
        output_name = f'unet_preds_{split_name}.npy'
        save_path = os.path.join(output_dir, output_name)
        np.save(save_path, final_stacked_preds)
        logger.info(f"  [SUCCESS] All crop outputs for {split_name} stacked and saved to {save_path}")
        logger.info(f"  Final Shape: {final_stacked_preds.shape}")

if __name__ == "__main__":
    main()
