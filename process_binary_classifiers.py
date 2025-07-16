import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress tracking
# Assuming CropClassifierTree and your dataset class are imported or defined here
from model import CropClassifierTree
from torch.utils.data import Dataset
from data_manager import create_modified_crop_labels
from data_manager import setup_training_loader


class CropDataset(Dataset):
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

def get_data_loaders(batch_size=1):
    """
    Create and return data loaders for train, validation, and test sets.
    
    Args:
        batch_size (int): Batch size for the data loaders. Defaults to 1.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_data = np.load('./training_data/train_patches.npy')
    valid_data = np.load('./training_data/val_patches.npy')
    test_data = np.load('./training_data/test_patches.npy')
    
    unchanged_crops = [1, 5, 23, 176]
    train_data = create_modified_crop_labels(train_data, unchanged_crops=unchanged_crops)
    valid_data = create_modified_crop_labels(valid_data, unchanged_crops=unchanged_crops)
    test_data = create_modified_crop_labels(test_data, unchanged_crops=unchanged_crops)

    # Create datasets
    train_dataset = CropDataset(train_data)
    val_dataset = CropDataset(valid_data)
    test_dataset = CropDataset(test_data)

    # Print number of classes
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Label mapping: {train_dataset.label_map}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Print dataset sizes and sample shapes
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def process_and_save_binary_outputs(classifier_tree, output_dir='binary_classifiers_outputs'):
    """
    Iterate over train, validation, and test sets, run binary classifiers,
    concatenate outputs, and save them along with the labels.
    Process images by dividing them into 4 subpatches of 112x112.
    
    Args:
        classifier_tree (CropClassifierTree): The classifier tree instance
        output_dir (str): Directory to save the outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders with batch size 1
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=1)
    
    # Process each dataset
    for dataset_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f'Processing {dataset_name} set...')
        all_eigen_vectors = []
        all_labels = []
        
        with torch.no_grad():
            # Use tqdm to track progress
            for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f'Processing {dataset_name}')):
                # Remove batch dimension since run_binary_classifiers expects (H, W, C)
                image = images.squeeze(0)  # Shape: (H, W, C)
                label = labels.squeeze(0)  # Shape: (H, W)
                
                # Move image to the same device as the model
                image = image.to('cuda')
                
                # Divide image into 4 subpatches of 112x112
                subpatch_eigen_vectors = []
                subpatch_labels = []
                for i in range(2):
                    for j in range(2):
                        start_h = i * 112
                        start_w = j * 112
                        subpatch = image[start_h:start_h+112, start_w:start_w+112, :]
                        subpatch_label = label[start_h:start_h+112, start_w:start_w+112]
                        
                        # Run binary classifiers on the subpatch
                        predictions, eigen_vectors = classifier_tree.run_binary_classifiers(subpatch)
                        
                        # Store eigen vectors for this subpatch
                        concatenated = np.stack([eigen_vectors[crop_id] for crop_id in classifier_tree.crop_order], axis=-1)
                        subpatch_eigen_vectors.append(concatenated)
                        subpatch_labels.append(subpatch_label.numpy())
                
                # Reconstruct full image eigen vectors from subpatches
                full_eigen_vectors = np.zeros((1, 224, 224, len(classifier_tree.crop_order)))
                full_labels = np.zeros((1, 224, 224))
                idx = 0
                for i in range(2):
                    for j in range(2):
                        start_h = i * 112
                        start_w = j * 112
                        full_eigen_vectors[0, start_h:start_h+112, start_w:start_w+112, :] = subpatch_eigen_vectors[idx]
                        full_labels[0, start_h:start_h+112, start_w:start_w+112] = subpatch_labels[idx]
                        idx += 1
                
                all_eigen_vectors.append(full_eigen_vectors)
                all_labels.append(full_labels)
        
        # Concatenate all batches
        dataset_eigen_vectors = np.concatenate(all_eigen_vectors, axis=0)
        dataset_labels = np.concatenate(all_labels, axis=0)
        
        # Save the concatenated eigen vectors
        output_path = os.path.join(output_dir, f'{dataset_name}_eigen_vectors.npy')
        np.save(output_path, dataset_eigen_vectors)
        print(f'Saved {dataset_name} set eigen vectors to {output_path}')
        
        # Save the labels
        labels_path = os.path.join(output_dir, f'{dataset_name}_labels.npy')
        np.save(labels_path, dataset_labels)
        print(f'Saved {dataset_name} set labels to {labels_path}')

if __name__ == '__main__':
    # Initialize your classifier tree
    checkpoints_dir = './checkpoints/v2/'  # Replace with actual path
    crop_order = [-1,1,5,23,176]  # Replace with your actual crop order
    tree = CropClassifierTree(checkpoints_dir='./checkpoints/v2/', 
                   crop_order=crop_order, img_height=112,
                   img_width=112, 
                   window_size=30)

    for crop_id in crop_order:
                
            
        # Setup training loader for this specific crop
        train_loader = setup_training_loader(
            path_to_train_data='./training_data/train_patches.npy',
            unchanged_crops=[1, 5, 23, 176],  # All unchanged crops
            target_crops=[crop_id],           # Current crop is the target
            train_batch_size=16,
            crop_band_index=18,
            device='cuda',
            ignore_crops=None,
            min_ratio=0.1,
            max_ratio=0.9
            )
            
        # Compute feature centers for this classifier
        classifier = tree.classifiers[crop_id]
        positive_center, negative_center = classifier.calculate_feature_centers(train_loader)
        print(positive_center, negative_center)
        
        # Release memory by deleting the train_loader
        del train_loader
    
    # Process and save outputs
    process_and_save_binary_outputs(tree)