

import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from mne.io import read_raw_edf
import warnings
import os
warnings.filterwarnings('ignore')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass


def predict_seizure(edf_path, model_path=None, scaler_path=None):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths relative to the script directory if not provided
    if model_path is None:
        model_path = os.path.join(script_dir, 'epilepsy_eegnet_model.h5')
    if scaler_path is None:
        scaler_path = os.path.join(script_dir, 'epilepsy_scaler.joblib')
    
    # Verify that the required files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found at: {edf_path}")

    # Load the model and scaler
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    raw = read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    
    window_samples = 256  
    max_channels = 23
    stride = 256  
    batch_size = 128  
    
    if data.shape[0] > max_channels:
        data = data[:max_channels, :]
    elif data.shape[0] < max_channels:
        padding = np.zeros((max_channels - data.shape[0], data.shape[1]))
        data = np.vstack([data, padding])
    
    n_samples = data.shape[1]
    windows = []
    
    for start_idx in range(0, n_samples - window_samples, stride):
        end_idx = start_idx + window_samples
        window = data[:, start_idx:end_idx]
        windows.append(window)
    
    if len(windows) == 0:
        return 0
    
    windows = np.array(windows)  
    
    windows_flat = windows.reshape(len(windows), -1)
    windows_normalized = scaler.transform(windows_flat).reshape(len(windows), max_channels, window_samples, 1)
    
    predictions = model.predict(windows_normalized, batch_size=batch_size, verbose=0)
    
    seizure_count = np.sum(predictions > 0.5)
    total_count = len(predictions)
    
    seizure_ratio = seizure_count / total_count if total_count > 0 else 0
    return "seizure" if seizure_ratio > 0.1 else "no seizure"
