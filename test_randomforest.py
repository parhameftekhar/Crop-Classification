import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import gc
import joblib

def evaluate_random_forest(input_dir='binary_classifiers_outputs', window_size=3):
    """
    Load a trained RandomForestClassifier and evaluate it on validation and test sets.
    Features are enriched by considering a neighborhood around each pixel.
    
    Args:
        input_dir (str): Directory containing the binary outputs, labels, and saved model
        window_size (int): Size of the neighborhood window around each pixel (e.g., 3 for a 3x3 window)
    """
    # Load the saved model
    model_path = os.path.join(input_dir, 'best_random_forest_model.joblib')
    if not os.path.exists(model_path):
        print(f'Model file not found at {model_path}. Aborting...')
        return
    
    best_clf = joblib.load(model_path)
    print(f'Loaded model from: {model_path}')
    half_window = window_size // 2
    
    # Evaluate on validation set
    print('Evaluating model on validation set...')
    val_binary_path = os.path.join(input_dir, 'val_eigen_vectors.npy')
    val_labels_path = os.path.join(input_dir, 'val_labels.npy')
    
    if not os.path.exists(val_binary_path) or not os.path.exists(val_labels_path):
        print('Missing validation data files. Skipping validation...')
    else:
        val_binary = np.load(val_binary_path)
        val_labels = np.load(val_labels_path)
        
        print(f'Loaded validation binary outputs with shape: {val_binary.shape}')
        print(f'Loaded validation labels with shape: {val_labels.shape}')
        
        # Enrich features for validation set for all pixels in selected images
        num_samples_val, height_val, width_val, num_features_val = val_binary.shape
        enriched_val_features = []
        enriched_val_labels = []
        
        print(f'Enriching validation features with a {window_size}x{window_size} window for all pixels in selected images...')
        for i in tqdm(range(num_samples_val), desc='Processing validation samples'):
            for h in range(half_window, height_val - half_window):
                for w in range(half_window, width_val - half_window):
                    window_features = val_binary[i, h-half_window:h+half_window+1, w-half_window:w+half_window+1, :]
                    window_features_flat = window_features.reshape(-1)
                    enriched_val_features.append(window_features_flat)
                    enriched_val_labels.append(val_labels[i, h, w])
        
        X_val = np.array(enriched_val_features)
        y_val = np.array(enriched_val_labels)
        
        print(f'Enriched validation data shape: {X_val.shape} for features and {y_val.shape} for labels')
        
        # Clear validation data to free memory
        del val_binary, val_labels, enriched_val_features, enriched_val_labels
        gc.collect()
        
        # Predict and calculate accuracy on validation set
        y_val_pred = best_clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f'Validation set accuracy with model: {val_accuracy:.4f}')
    
    # Evaluate on test set
    print('Evaluating model on test set...')
    test_binary_path = os.path.join(input_dir, 'test_eigen_vectors.npy')
    test_labels_path = os.path.join(input_dir, 'test_labels.npy')
    
    if not os.path.exists(test_binary_path) or not os.path.exists(test_labels_path):
        print('Missing test data files. Skipping test evaluation...')
    else:
        test_binary = np.load(test_binary_path)
        test_labels = np.load(test_labels_path)
        
        print(f'Loaded test binary outputs with shape: {test_binary.shape}')
        print(f'Loaded test labels with shape: {test_labels.shape}')
        
        # Enrich features for test set for all pixels in selected images
        num_samples_test, height_test, width_test, num_features_test = test_binary.shape
        enriched_test_features = []
        enriched_test_labels = []
        
        print(f'Enriching test features with a {window_size}x{window_size} window for all pixels in selected images...')
        for i in tqdm(range(num_samples_test), desc='Processing test samples'):
            for h in range(half_window, height_test - half_window):
                for w in range(half_window, width_test - half_window):
                    window_features = test_binary[i, h-half_window:h+half_window+1, w-half_window:w+half_window+1, :]
                    window_features_flat = window_features.reshape(-1)
                    enriched_test_features.append(window_features_flat)
                    enriched_test_labels.append(test_labels[i, h, w])
        
        X_test = np.array(enriched_test_features)
        y_test = np.array(enriched_test_labels)
        
        print(f'Enriched test data shape: {X_test.shape} for features and {y_test.shape} for labels')
        
        # Clear test data to free memory
        del test_binary, test_labels, enriched_test_features, enriched_test_labels
        gc.collect()
        
        # Predict and calculate accuracy on test set
        y_test_pred = best_clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f'Test set accuracy with model: {test_accuracy:.4f}')

if __name__ == '__main__':
    # Run the evaluation with a 3x3 window
    evaluate_random_forest(window_size=3) 