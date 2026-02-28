import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import gc
import joblib

def evaluate_random_forest(input_dir='unet_binary_outputs', window_size=3):
    """
    Load a trained RandomForestClassifier and evaluate it on validation and test sets.
    """
    model_path = os.path.join(input_dir, 'best_random_forest_model_unet.joblib')
    if not os.path.exists(model_path):
        print(f'Model file not found at {model_path}. Aborting...')
        return
    
    best_clf = joblib.load(model_path)
    print(f'Loaded model from: {model_path}')
    half_window = window_size // 2
    
    for split in ['val', 'test']:
        print(f'\nEvaluating on {split} set...')
        binary_path = os.path.join(input_dir, f'unet_preds_{split}.npy')
        labels_path = os.path.join(input_dir, f'unet_labels_{split}.npy')
        
        if not os.path.exists(binary_path) or not os.path.exists(labels_path):
            print(f'Missing files for {split} split. Skipping...')
            continue
            
        binary_data = np.load(binary_path)
        labels_data = np.load(labels_path)
        
        num_samples, height, width, num_features = binary_data.shape
        enriched_features = []
        enriched_labels = []
        
        for i in tqdm(range(num_samples), desc=f'Processing {split} samples'):
            for h in range(half_window, height - half_window):
                for w in range(half_window, width - half_window):
                    window_features = binary_data[i, h-half_window:h+half_window+1, w-half_window:w+half_window+1, :]
                    enriched_features.append(window_features.flatten())
                    enriched_labels.append(labels_data[i, h, w])
        
        X = np.array(enriched_features)
        y = np.array(enriched_labels)
        
        print(f'Predicting for {split} set...')
        y_pred = best_clf.predict(X)
        
        acc = accuracy_score(y, y_pred)
        print(f'{split.capitalize()} accuracy: {acc:.4f}')
        print(f'{split.capitalize()} classification report:')
        print(classification_report(y, y_pred))
        
        del binary_data, labels_data, X, y, y_pred
        gc.collect()

if __name__ == '__main__':
    evaluate_random_forest(window_size=3)
