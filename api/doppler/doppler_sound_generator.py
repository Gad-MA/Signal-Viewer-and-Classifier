import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

sample_rate = 44100 # (Sample/second)

noise_amplitude = 0.0

def doppler_effect_with_distance(source_velocity, d, half_simulation_duration, source_freq, sound_velocity=343):
    """
    Model the Doppler effect for a source moving past an observer at distance d.

    Parameters:
    source_velocity : speed of the source (m/s)
    d : distance of closest approach (m)
    source_freq : emitted frequency (Hz) 
    sound_velocity : speed of sound (m/s)
    sample_rate: (sample/second)
    """

    # Create time array centered on closest approach
    t = np.linspace(-half_simulation_duration, half_simulation_duration,
                    int(sample_rate * half_simulation_duration * 2))

    # Calculate distance between source and observer at each time
    R = np.sqrt((source_velocity * t)**2 + d**2)

    # Calculate radial velocity (component along line of sight)
    # Derivative of R with respect to time
    v_radial = (source_velocity**2 * t) / \
        np.sqrt((source_velocity * t)**2 + d**2)

    # Apply Doppler formula
    # When source is approaching (v_radial < 0), use v_sound / (v_sound - |v_radial|)
    # When source is receding (v_radial > 0), use v_sound / (v_sound + v_radial)
    f_observed = source_freq * (sound_velocity / (sound_velocity + v_radial))

    return t, R, v_radial, f_observed


def doppler_effect_wav_generator(source_velocity, source_freq, normal_distance, half_simulation_duration):
    time, distance, _, f_observed = doppler_effect_with_distance(
        source_velocity=source_velocity, d=normal_distance, half_simulation_duration=half_simulation_duration, source_freq=source_freq)
    
    # Generate multiple frequency components for a more realistic car sound
    waveform = np.zeros_like(time, dtype=float)
    
    # Base engine frequency and its harmonics with increased amplitudes
    frequencies = [source_freq, source_freq*2, source_freq*3, source_freq*4]
    amplitudes = [2.0, 1.0, 0.6, 0.4]  # Increased amplitudes for harmonics
    
    for freq, amp in zip(frequencies, amplitudes):
        _, _, _, f_obs = doppler_effect_with_distance(
            source_velocity=source_velocity, d=normal_distance, 
            half_simulation_duration=half_simulation_duration, source_freq=freq)
        waveform += amp * np.sin(2 * np.pi * np.cumsum(f_obs) / sample_rate)
    
    # Add some noise for realism (reduced amplitude)
    noise = np.random.normal(0, noise_amplitude, len(time))
    waveform += noise
    
    # Apply distance-based amplitude modulation
    amplitude_mod = 1 / (1 + (distance / normal_distance)**2)
    waveform *= amplitude_mod
    
    # Normalize waveform to the range [-1, 1]
    waveform = waveform / np.max(np.abs(waveform))
    
    # Add a slight fade in/out to prevent clicking
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    waveform[:fade_samples] *= fade_in
    waveform[-fade_samples:] *= fade_out
    
    # Boost the volume and convert to 16-bit PCM format
    volume_boost = 1  # Increase volume
    waveform = waveform * volume_boost
    # Clip any values that exceed [-1, 1] to prevent distortion
    waveform = np.clip(waveform, -1, 1)
    waveform_integers = np.int16(waveform * 32767)

    # Save the waveform as a WAV file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_file = os.path.join(script_dir, "doppler_effect.wav")
    write(wav_file, sample_rate, waveform_integers)
    print(f"Wav file generated at: {wav_file}")