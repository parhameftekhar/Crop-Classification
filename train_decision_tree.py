import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import gc
import joblib

def train_and_evaluate_random_forest(input_dir='binary_classifiers_outputs', subsample_ratio=0.1, window_size=3, image_subsample_ratio=0.1):
    """
    Train a RandomForestClassifier on a subsample of the training set binary outputs with hyperparameter tuning,
    evaluate on validation set to pick the best model, and test on the test set.
    Features are enriched by considering a neighborhood around each pixel.
    
    Args:
        input_dir (str): Directory containing the corrected binary outputs and labels
        subsample_ratio (float): Ratio of enriched training data to use for subsampling after enrichment (0.0 to 1.0)
        window_size (int): Size of the neighborhood window around each pixel (e.g., 3 for a 3x3 window)
        image_subsample_ratio (float): Ratio of images to subsample before feature enrichment (0.0 to 1.0)
    """
    # Label mapping for reference (though not used directly in predictions here)
    label_map = {
        -1: 0,  # background
        1: 1,   # corn
        5: 2,   # soybean
        23: 3,  # spring wheat
        176: 4  # grassland/pasture
    }
    
    # Load training data
    train_binary_path = os.path.join(input_dir, 'train_eigen_vectors.npy')
    train_labels_path = os.path.join(input_dir, 'train_labels.npy')
    
    if not os.path.exists(train_binary_path) or not os.path.exists(train_labels_path):
        print('Missing training data files. Aborting...')
        return
    
    train_binary = np.load(train_binary_path)
    train_labels = np.load(train_labels_path)
    
    print(f'Loaded training binary outputs with shape: {train_binary.shape}')
    print(f'Loaded training labels with shape: {train_labels.shape}')
    
    # Subsample images before feature enrichment
    num_samples = train_binary.shape[0]
    subsample_image_count = int(num_samples * image_subsample_ratio)
    image_indices = np.random.choice(num_samples, size=subsample_image_count, replace=False)
    train_binary = train_binary[image_indices]
    train_labels = train_labels[image_indices]
    
    print(f'Subsampled training images to: {train_binary.shape} (using {image_subsample_ratio*100}% of images)')
    
    # Enrich features by considering a neighborhood around each pixel for all pixels in selected images
    num_samples, height, width, num_features = train_binary.shape
    half_window = window_size // 2
    enriched_features = []
    enriched_labels = []
    
    print(f'Enriching features with a {window_size}x{window_size} window for all pixels in selected images...')
    for i in tqdm(range(num_samples), desc='Processing training samples'):
        for h in range(half_window, height - half_window):
            for w in range(half_window, width - half_window):
                # Extract the neighborhood window for this pixel
                window_features = train_binary[i, h-half_window:h+half_window+1, w-half_window:w+half_window+1, :]
                # Flatten the window features
                window_features_flat = window_features.reshape(-1)
                enriched_features.append(window_features_flat)
                enriched_labels.append(train_labels[i, h, w])
    
    X_train_full = np.array(enriched_features)
    y_train_full = np.array(enriched_labels)
    
    print(f'Enriched training data shape: {X_train_full.shape} for features and {y_train_full.shape} for labels')
    
    # Clear original data to free memory
    del train_binary, train_labels, enriched_features, enriched_labels
    gc.collect()
    
    # Subsample the enriched training data
    total_pixels = X_train_full.shape[0]
    subsample_size = int(total_pixels * subsample_ratio)
    indices = np.random.choice(total_pixels, size=subsample_size, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    
    print(f'Subsampled enriched training data to: {X_train.shape} for features and {y_train.shape} for labels (using {subsample_ratio*100}% of data)')
    
    # Clear full data to free memory
    del X_train_full, y_train_full
    gc.collect()
    
    # Define parameter grid for GridSearchCV - simplified to reduce memory usage
    param_grid = {
        'n_estimators': [50],
        'max_depth': [None, 10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'criterion': ['gini']
    }
    
    # Initialize the classifier
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform GridSearchCV with limited parallel jobs
    print('Performing hyperparameter tuning with GridSearchCV on subsampled data...')
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    
    print('Hyperparameter tuning completed.')
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_:.4f}')
    
    # Get the best model
    best_clf = grid_search.best_estimator_
    
    # Save the best model
    model_save_path = os.path.join(input_dir, 'best_random_forest_model.joblib')
    joblib.dump(best_clf, model_save_path)
    print(f'Best model saved to: {model_save_path}')
    
    # Evaluate on validation set
    print('Evaluating best model on validation set...')
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
        print(f'Validation set accuracy with best model: {val_accuracy:.4f}')
    
    # Evaluate on test set with the best model
    print('Evaluating best model on test set...')
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
        print(f'Test set accuracy with best model: {test_accuracy:.4f}')

if __name__ == '__main__':
    # Run the training and evaluation with a 3x3 window and reduced subsampling ratios
    train_and_evaluate_random_forest(window_size=3, image_subsample_ratio=0.5, subsample_ratio=0.2) 