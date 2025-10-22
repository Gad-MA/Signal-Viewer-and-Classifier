import os
import sys
import torch
import torchaudio
import subprocess


def install_dependencies():
    try:
        import model
    except ImportError:
        subprocess.run(["git", "clone", "https://github.com/JaesungHuh/voice-gender-classifier.git"], check=True)
        subprocess.run(["pip", "install", "torch", "torchaudio", "huggingface_hub"], check=True)
        subprocess.run(["pip", "install", "-r", "voice-gender-classifier/requirements.txt"], check=True)


def load_model():
    sys.path.append("voice-gender-classifier")
    from model import ECAPA_gender
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
    model.to(device)
    model.eval()
    return model, device


def load_audio(path, target_sample_rate=16000):
    waveform, sr = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # convert stereo → mono
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)
    return waveform, target_sample_rate


def predict_gender(model, device, audio_path):
    with torch.no_grad():
        try:
            output = model.predict(audio_path, device=device)
            return output
        except AttributeError:
            waveform, sr = load_audio(audio_path)
            waveform = waveform.to(device)
            logits = model(waveform)
            gender_idx = torch.argmax(logits, dim=-1).item()
            label_map = {0: "female", 1: "male"}
            return label_map.get(gender_idx, str(gender_idx))

def infer_gender_from_audio(audio_path):
    install_dependencies()
    model, device = load_model()

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    prediction = predict_gender(model, device, audio_path)
    return prediction
