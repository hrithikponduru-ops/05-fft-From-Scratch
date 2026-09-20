import numpy as np
from scipy.io import wavfile

# 1. Setup audio parameters
sample_rate = 44100  # Standard audio sample rate (Hz)
duration = 2.0       # Duration in seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 2. Define frequencies for a C-Major chord (C4, E4, G4)
freq_c = 261.63
freq_e = 329.63
freq_g = 392.00

# 3. Generate the clean chord
wave_c = np.sin(2 * np.pi * freq_c * t)
wave_e = np.sin(2 * np.pi * freq_e * t)
wave_g = np.sin(2 * np.pi * freq_g * t)
clean_chord = wave_c + wave_e + wave_g

# Normalize clean audio to prevent clipping
clean_chord = clean_chord / np.max(np.abs(clean_chord))

# 4. Generate white noise and combine it with the clean chord
noise_level = 0.5
noise = np.random.normal(0, noise_level, clean_chord.shape)
noisy_chord = clean_chord + noise

# Normalize noisy audio
noisy_chord = noisy_chord / np.max(np.abs(noisy_chord))

# 5. Convert to 16-bit PCM format (standard for .wav files)
clean_chord_16bit = np.int16(clean_chord * 32767)
noisy_chord_16bit = np.int16(noisy_chord * 32767)

# 6. Save the files to your current directory
wavfile.write("chord.wav", sample_rate, clean_chord_16bit)
wavfile.write("chord_noisy.wav", sample_rate, noisy_chord_16bit)

print("Files 'chord.wav' and 'chord_noisy.wav' generated successfully!")