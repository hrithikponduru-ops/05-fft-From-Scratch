"""Synthesised test signals and WAV files, using only the standard library `wave`."""

import wave
from pathlib import Path

import numpy as np

INT16_MAX = 32767

def synthesize_tone(
    freq: float, duration: float, sample_rate: int, harmonics: tuple[float, ...] = (1.0, 0.5, 0.25)
) -> np.ndarray:
    """A note with overtones: sum_j h_j sin(2 pi (j+1) f t).

    A pure sine sounds like a tuning fork. Adding the 2nd and 3rd harmonics
    at decreasing amplitude gives something closer to a plucked string.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return sum(h * np.sin(2 * np.pi * (j + 1) * freq * t) for j, h in enumerate(harmonics))

def synthesize_chord(
    freqs: tuple[float, ...], duration: float, sample_rate: int,
    harmonics: tuple[float, ...] = (1.0, 0.5, 0.25), peak: float = 0.9,
) -> np.ndarray:
    """Several tones at once, scaled so the loudest sample is `peak` (< 1 avoids clipping)."""
    chord = sum(synthesize_tone(f, duration, sample_rate, harmonics) for f in freqs)
    return chord * (peak / np.max(np.abs(chord)))

def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise scaled to the requested signal-to-noise ratio in dB."""
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / 10 ** (snr_db / 10)
    return x + rng.normal(scale=np.sqrt(noise_power), size=len(x))

def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """10 log10(signal power / noise power), noise being whatever differs from clean."""
    noise = noisy - clean
    return float(10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2)))

def write_wav(path: Path, x: np.ndarray, sample_rate: int) -> None:
    """16-bit mono PCM. Samples outside [-1, 1] are clipped, so scale first."""
    samples = (np.clip(x, -1.0, 1.0) * INT16_MAX).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as floats in [-1, 1]. Stereo is averaged to mono."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * f.getsampwidth()}-bit")
        channels, sample_rate = f.getnchannels(), f.getframerate()
        raw = f.readframes(f.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(float) / INT16_MAX
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate