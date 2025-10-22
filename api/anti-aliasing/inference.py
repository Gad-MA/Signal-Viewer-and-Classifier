import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE = 16000
AUDIO_DURATION = 15.0  
class LightweightAudioAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, 9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, 9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, 9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.ConvTranspose1d(64, 1, 9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottleneck(e3)
        
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        out = self.final(d1)
        return out

def restore_audio(audio: np.ndarray, model: nn.Module, device: str) -> np.ndarray:

    chunk_size = int(AUDIO_DURATION * SAMPLE_RATE)
    restored_chunks = []
    
    
    with torch.no_grad():
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            original_chunk_len = len(chunk)
            
            if original_chunk_len < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - original_chunk_len), 'constant')
            
            chunk_max = np.max(np.abs(chunk))
            if chunk_max > 1e-6:
                chunk_normalized = chunk / chunk_max
            else:
                chunk_normalized = chunk  
            
            chunk_tensor = torch.tensor(chunk_normalized, dtype=torch.float32)
            chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            restored_chunk_tensor = model(chunk_tensor)
            restored_chunk_normalized = restored_chunk_tensor.cpu().squeeze().numpy()
            
            if chunk_max > 1e-6:
                restored_chunk = restored_chunk_normalized * chunk_max
            else:
                restored_chunk = restored_chunk_normalized
            
            restored_chunks.append(restored_chunk[:original_chunk_len])
    
    restored_audio = np.concatenate(restored_chunks)
    return restored_audio

def denoise_audio(audio: np.ndarray, sample_rate: int, device: str) -> np.ndarray:

    try:
        from denoiser import pretrained
        
        denoiser_model = pretrained.dns64().to(device)
        denoiser_model.eval()
        
        if sample_rate != denoiser_model.sample_rate:
            audio_resampled = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=denoiser_model.sample_rate
            )
        else:
            audio_resampled = audio
        
        wav_tensor = torch.from_numpy(audio_resampled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            denoised_tensor = denoiser_model(wav_tensor)[0]
        
        denoised_audio = denoised_tensor.cpu().numpy().squeeze()
        
        if sample_rate != denoiser_model.sample_rate:
            denoised_audio = librosa.resample(
                denoised_audio,
                orig_sr=denoiser_model.sample_rate,
                target_sr=sample_rate
            )
        
        return denoised_audio
        
    except ImportError:
        print("     To enable denoising, install with: pip install -U denoiser")
        return audio
    except Exception as e:
        print(f"     ⚠️  Warning: Denoising failed with error: {e}")
        print("     Continuing with restored audio without denoising...")
        return audio


def restore_and_clean_audio(
    input_path: str,
    output_path: str,
    model_path: str = "best_model.pth" # you can change that 
) -> np.ndarray:


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found at '{input_path}'")
    
    original_audio, original_sr = sf.read(input_path)
    
    if original_sr >= 16000:
        print("   No aliasing detected")
        sf.write(output_path, original_audio, original_sr)
        return original_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LightweightAudioAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    

    
    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    

    
    restored_audio = restore_audio(audio, model, device)
    

    try:
        final_audio = denoise_audio(restored_audio, SAMPLE_RATE, device)
    except Exception as e:
        final_audio = restored_audio
    

    sf.write(output_path, final_audio, SAMPLE_RATE)

    return final_audio


# ============================================================
# EXAMPLE USAGE
# ============================================================

# if __name__ == "__main__":
#     print("Example: Restore and clean audio with denoising")
#     cleaned_audio = restore_and_clean_audio(
#         input_path="/content/downsampled.wav",
#         output_path="output.wav"
#     )
#     print(f"\nReturned audio shape: {cleaned_audio.shape}")