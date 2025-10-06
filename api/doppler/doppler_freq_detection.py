!pip install torchcrepe
!pip install soundfile


import torch
import torchaudio
import torchcrepe
import pandas as pd
import matplotlib.pyplot as plt

# =============== CONFIG ====================
audio_path = "./doppler_effect.wav"     # <-- change this to your file
fmin, fmax = 50, 550              # typical for human voice
model = "full"                    # or "tiny" for faster inference
hop_length = None                 # auto = 5 ms hop (~200 fps)
device = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

# Load audio file
audio, sr = torchaudio.load(audio_path)
audio = torch.mean(audio, dim=0, keepdim=True)  # convert to mono

# Define hop_length for ~5ms
if hop_length is None:
    hop_length = int(sr / 200)

# Move to device
audio = audio.to(device)

# Predict pitch and periodicity
with torch.no_grad():
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model=model,
        return_periodicity=True,
        batch_size=2048,
        device=device
    )

# Filter and threshold (optional smoothing)
periodicity = torchcrepe.filter.median(periodicity, win_length=3)
pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)
pitch = torchcrepe.filter.mean(pitch, win_length=3)

# Convert hop indices to time (seconds)
times = torch.arange(pitch.shape[-1]) * hop_length / sr

# Move back to CPU and flatten
times = times.cpu().numpy()
pitch = pitch.cpu().numpy().flatten()
periodicity = periodicity.cpu().numpy().flatten()

# Save to CSV
df = pd.DataFrame({"time_sec": times, "frequency_Hz": pitch, "periodicity": periodicity})
df.to_csv("pitch_output.csv", index=False)
print("✅ Pitch prediction saved to pitch_output.csv")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(times, pitch, label="Pitch (Hz)")
plt.title("Estimated Pitch Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.legend()
plt.show()
