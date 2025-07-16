import torch
import os
import numpy as np
from tqdm import tqdm
from model import CropClassifierTree
from data_manager import setup_training_loader

def evaluate_binary_predictions(tree, input_dir='binary_classifier_outputs'):
    """
    Load corrected binary classifier outputs for validation and test sets, combine them using
    tree.combine_binary_predictions, and compute accuracy metrics without saving outputs.
    Apply label mapping to convert tree outputs to corresponding labels.
    
    Args:
        tree (CropClassifierTree): The classifier tree instance
        input_dir (str): Directory containing the corrected binary outputs and labels
    """
    # Label mapping for converting tree outputs
    label_map = {
        -1: 0,  # background
        1: 1,   # corn
        5: 2,   # soybean
        23: 3,  # spring wheat
        176: 4  # grassland/pasture
    }
    
    # Process only validation and test datasets
    for dataset_name in ['val', 'test']:
        print(f'Evaluating predictions for {dataset_name} set...')
        
        # Load corrected binary outputs and labels
        binary_outputs_path = os.path.join(input_dir, f'{dataset_name}_binary_outputs.npy')
        labels_path = os.path.join(input_dir, f'{dataset_name}_labels.npy')
        
        if not os.path.exists(binary_outputs_path) or not os.path.exists(labels_path):
            print(f'Missing files for {dataset_name} set. Skipping...')
            continue
        
        binary_outputs = np.load(binary_outputs_path)
        labels = np.load(labels_path)
        
        print(f'Loaded {dataset_name} binary outputs with shape: {binary_outputs.shape}')
        print(f'Loaded {dataset_name} labels with shape: {labels.shape}')
        
        accuracy_list = []
        
        # Process each image in the dataset
        for idx in tqdm(range(binary_outputs.shape[0]), desc=f'Evaluating {dataset_name} predictions'):
            # Extract predictions and label for this image
            image_predictions = binary_outputs[idx]  # Shape: (H, W, num_crops)
            image_label = labels[idx]  # Shape: (H, W)
            
            # Convert predictions to dictionary format expected by combine_binary_predictions
            predictions_dict = {
                crop_id: image_predictions[:, :, i] for i, crop_id in enumerate(tree.crop_order)
            }
            
            # Combine predictions
            combined_pred = tree.combine_binary_predictions(predictions_dict)
            
            # Apply label mapping to convert tree outputs to corresponding labels
            mapped_pred = np.vectorize(label_map.get)(combined_pred)
            
            # Calculate accuracy for the full image
            accuracy = (mapped_pred == image_label).sum() / combined_pred.size
            accuracy_list.append(accuracy)
        
        # Compute and print average accuracy for the dataset
        avg_accuracy = np.mean(accuracy_list)
        print(f'Average accuracy for {dataset_name} set: {avg_accuracy:.4f}')

if __name__ == '__main__':
    # Initialize your classifier tree
    checkpoints_dir = './checkpoints/v2/'  # Path to checkpoints
    crop_order = [1, 5, 23, 176]  # Your crop order
    tree = CropClassifierTree(
        checkpoints_dir=checkpoints_dir,
        crop_order=crop_order,
        img_height=112,
        img_width=112,
        window_size=30
    )
    
    # Calculate feature centers for each classifier
    for crop_id in crop_order:
        print(f'Calculating feature centers for crop {crop_id}...')
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
        print(f'Feature centers for crop {crop_id}: Positive={positive_center.shape}, Negative={negative_center.shape}')
    
    # Evaluate predictions without saving
    evaluate_binary_predictions(tree) 