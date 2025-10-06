import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Audio preprocessing
DESIRED_SR = 16000   # YAMNet requires 16 kHz audio

def load_wav_16k_mono(path, desired_sr=DESIRED_SR):
    """
    Loads an audio file, converts to mono and resamples to 16kHz.
    """
    wav, sr = librosa.load(path, sr=None, mono=True)   # load audio
    if sr != desired_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=desired_sr)
    return wav

# Load YAMNet from TensorFlow Hub
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# Load the trained classifier
classifier = tf.keras.models.load_model("yamnet_drone_classifier.h5")

def process_drone_audio(audio_path: str) -> str:
    """
    Takes an audio file path, runs drone sound detection,
    and returns 'Drone detected' or 'No drone detected'.
    """
    # Load and preprocess audio
    wav = load_wav_16k_mono(audio_path)

    # YAMNet expects float32 waveform
    scores, embeddings, spectrogram = yamnet_model(tf.convert_to_tensor(wav, dtype=tf.float32))

    # Average embeddings across time
    embedding_mean = np.mean(embeddings, axis=0)
    embedding_mean = embedding_mean.reshape(1, -1)

    # Predict using the trained classifier
    prediction = classifier.predict(embedding_mean)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map to labels (0 = drone, 1 = noise)
    if predicted_class == 0:
        return "Drone detected"
    else:
        return "No drone detected"
