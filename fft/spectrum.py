"""Things you can do once you have an FFT: spectra, peaks, notes, filtering."""

import numpy as np

from fft.fft import fft_iterative, ifft, pad_to_power_of_two

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_HZ = 440.0
A4_MIDI = 69


def frequencies(n: int, sample_rate: float) -> np.ndarray:
    """Frequency of bins 0 .. N/2: k * sample_rate / N. Bin N/2 is the Nyquist limit."""
    return np.arange(n // 2 + 1) * sample_rate / n


def magnitude_spectrum(x: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a real signal.

    A real signal has X_{N-k} = conj(X_k), so bins above N/2 repeat the ones
    below and are dropped. Amplitudes are scaled so a unit-amplitude cosine
    at a bin frequency reads 1.0: divide by N, then double every bin except
    DC and Nyquist, whose energy is not split across two bins.
    """
    padded = pad_to_power_of_two(np.asarray(x, dtype=float))
    n = len(padded)
    X = fft_iterative(padded)[: n // 2 + 1]
    mags = np.abs(X) / n
    scale = np.full(len(mags), 2.0)
    scale[0] = 1.0
    if n % 2 == 0:
        scale[-1] = 1.0
    return frequencies(n, sample_rate), mags * scale


def find_peaks(freqs: np.ndarray, mags: np.ndarray, n_peaks: int, min_separation_hz: float) -> np.ndarray:
    """Greedy peak picking: take the tallest bin, mask its neighbourhood, repeat."""
    remaining = np.array(mags, dtype=float)
    peaks = []
    for _ in range(n_peaks):
        i = int(np.argmax(remaining))
        if remaining[i] <= 0:
            break
        peaks.append(freqs[i])
        remaining = np.where(np.abs(freqs - freqs[i]) < min_separation_hz, 0.0, remaining)
    return np.array(peaks)


def frequency_to_note_name(freq: float) -> str:
    """Nearest equal-tempered note. Each semitone multiplies frequency by 2^(1/12)."""
    semitones_from_a4 = int(round(12 * np.log2(freq / A4_HZ)))
    midi = A4_MIDI + semitones_from_a4
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def convolve_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear convolution via the convolution theorem: FFT(x * y) = FFT(x) FFT(y).

    The theorem gives *circular* convolution, so both inputs are zero-padded
    to at least len(x) + len(y) - 1 first, which is where a linear convolution
    stops wrapping around.
    """
    x, y = np.asarray(x), np.asarray(y)
    full_length = len(x) + len(y) - 1
    n = len(pad_to_power_of_two(np.zeros(full_length)))
    X = fft_iterative(np.concatenate([x, np.zeros(n - len(x))]))
    Y = fft_iterative(np.concatenate([y, np.zeros(n - len(y))]))
    result = ifft(X * Y)[:full_length]
    if np.isrealobj(x) and np.isrealobj(y):
        return result.real
    return result


def lowpass_filter(x: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """Brick-wall low-pass: zero every bin whose frequency exceeds the cutoff.

    Both the positive-frequency bin k and its mirror N - k must be zeroed to
    keep the result real. A hard cut in frequency is a sinc in time, so the
    output rings near sharp edges; see the README.
    """
    x = np.asarray(x, dtype=float)
    padded = pad_to_power_of_two(x)
    n = len(padded)
    k = np.arange(n)
    freq = np.where(k <= n // 2, k, k - n) * sample_rate / n  # signed frequency per bin
    X = fft_iterative(padded)
    X_filtered = np.where(np.abs(freq) <= cutoff_hz, X, 0.0)
    return ifft(X_filtered).real[: len(x)]


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window 0.5 (1 - cos(2 pi k / N)). Overlap-adds to 1 at 50 % hop."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def spectrogram(
    x: np.ndarray, sample_rate: float, window_size: int, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform magnitudes: frame, window, FFT, repeat.

    Returns (frame_times, freqs, S) with S[f, t] the magnitude of frequency
    f in frame t. The window tapers each frame to zero at its edges so the
    frame boundaries do not smear energy across every frequency.
    """
    x = np.asarray(x, dtype=float)
    window = hann_window(window_size)
    n_frames = 1 + (len(x) - window_size) // hop
    starts = np.arange(n_frames) * hop
    frames = np.stack([x[s: s + window_size] * window for s in starts])
    S = np.stack([np.abs(fft_iterative(frame)[: window_size // 2 + 1]) for frame in frames], axis=1)
    times = (starts + window_size / 2) / sample_rate
    return times, frequencies(window_size, sample_rate), S