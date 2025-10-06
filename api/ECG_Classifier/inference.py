import wfdb
import numpy as np
import os
from scipy.signal import butter, lfilter
from tensorflow.keras.models import load_model
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

_model = None

def predict_ecg_arrhythmia(record_name, pn_dir, model_path='ecg_full_model.h5'):
    global _model
    
    if _model is None:
        _model = load_model(model_path)
    model = _model
    
    def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, signal)
    
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    except Exception as e:
        return f"Error loading record: {str(e)}"
    
    signal = record.p_signal
    fs = record.fs  
    

    if signal.shape[1] != 12:
        return "Error: The model expects 12-lead ECG data."
    if signal.shape[0] != 5000:
        return "Error: The model expects ECG data of length 5000 samples (10 seconds at 500 Hz)."
    if fs != 500:
        return "Error: The model expects data sampled at 500 Hz."
    
    for ch in range(12):
        signal[:, ch] = bandpass_filter(signal[:, ch], fs=fs)
    
    for ch in range(12):
        std = np.std(signal[:, ch])
        if std > 0:
            signal[:, ch] = (signal[:, ch] - np.mean(signal[:, ch])) / std
    
    signal = np.expand_dims(signal, axis=0)  
    
    prediction = model.predict(signal, verbose=0)
    pred_class_idx = np.argmax(prediction, axis=1)[0]
    
    classes = {
        0: 'Atrial Fibrillation/Flutter (AFIB)',
        1: 'Hypertrophy (HYP)',
        2: 'Myocardial Infarction (MI)',
        3: 'Normal Sinus Rhythm',
        4: 'Other Abnormality',
        5: 'ST-T Changes (STTC)' 
    }
    
    # Get confidence score
    confidence = prediction[0][pred_class_idx] * 100
    return {'predicted_class':f"{classes.get(pred_class_idx, 'Unknown')}", 'confidience':f"{confidence:.2f}%"}