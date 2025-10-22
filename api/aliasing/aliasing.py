import librosa
import numpy as np
import random

SR = 16000 #hz
TARGET_LEN = 8 #seconds

# Helper functions for lazy loading
def load_and_pad_audio(path, sr=SR, target_len=TARGET_LEN):
    """Load audio file and pad/truncate to target length."""
    y, _ = librosa.load(path, sr=sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y

def degrade_audio(y, sr_high=SR, downsample_factor=5, add_noise=True, noise_level=0.005, 
                  naive_aliasing=True):
    """
    Apply degradation effects to simulate low-quality audio.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Input audio waveform
    sr_high : int
        Original sample rate (default: 16000 Hz)
    downsample_factor : int
        Factor by which to downsample (default: 5, meaning 16000/5 = 3200 Hz)
    add_noise : bool
        Whether to add Gaussian noise (default: True)
    noise_level : float
        Standard deviation of Gaussian noise (default: 0.005)
    naive_aliasing : bool
        If True, use naive downsampling (NO anti-aliasing filter - creates frequency folding)
        If False, use librosa's filtered resampling (cleaner, but loses high frequencies)
        (default: True)
    
    Returns:
    --------
    degraded : numpy.ndarray
        Degraded audio with aliasing and optional noise
    """
    
    # 1. ALIASING: Downsample then upsample to introduce aliasing artifacts
    if naive_aliasing:
        # NAIVE DOWNSAMPLING: NO anti-aliasing filter - creates TRUE frequency folding
        # This creates harsh "unnatural robotic sounds" from frequency folding
        y_low = y[::downsample_factor]  # Naive decimation - NO filtering!
        
        # Upsample by zero-padding (also naive - creates imaging artifacts)
        y_up = np.zeros(len(y))
        y_up[::downsample_factor] = y_low[:len(y_up[::downsample_factor])]
        
        # Ensure same length
        if len(y_up) < len(y):
            y_up = np.pad(y_up, (0, len(y) - len(y_up)), mode='constant')
        else:
            y_up = y_up[:len(y)]
    else:
        # FILTERED RESAMPLING: librosa includes anti-aliasing filter
        # Cleaner degradation, but permanently loses high-frequency information
        sr_low = sr_high // downsample_factor
        y_low = librosa.resample(y, orig_sr=sr_high, target_sr=sr_low)
        y_up = librosa.resample(y_low, orig_sr=sr_low, target_sr=sr_high)
        
        # Ensure same length after resampling
        if len(y_up) < len(y):
            y_up = np.pad(y_up, (0, len(y) - len(y_up)), mode='constant')
        else:
            y_up = y_up[:len(y)]
    
    degraded = y_up.copy()
    
    # 2. ADDITIVE NOISE: Add Gaussian noise to simulate environmental/recording noise
    if add_noise:
        noise = np.random.normal(0, noise_level, len(degraded))
        degraded = degraded + noise
    
    # Clip to prevent overflow
    degraded = np.clip(degraded, -1.0, 1.0)
    
    return degraded
