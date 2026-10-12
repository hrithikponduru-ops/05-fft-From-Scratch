"""Spectrum tools built on the hand-written FFT: convolution, peaks, notes, filtering."""

import numpy as np
import pytest

from fft.audio import synthesize_tone
from fft.spectrum import (
    convolve_fft,
    find_peaks,
    frequency_to_note_name,
    hann_window,
    lowpass_filter,
    magnitude_spectrum,
    spectrogram,
)


@pytest.mark.parametrize("lengths", [(37, 50), (1, 1), (100, 3), (64, 64)])
def test_convolution_theorem_matches_direct_convolution(lengths):
    rng = np.random.default_rng(sum(lengths))
    x, y = rng.normal(size=lengths[0]), rng.normal(size=lengths[1])
    np.testing.assert_allclose(convolve_fft(x, y), np.convolve(x, y), atol=1e-9)


def test_pure_tone_peak_is_at_its_frequency():
    sample_rate, duration = 8000, 1.0
    x = synthesize_tone(440.0, duration, sample_rate, harmonics=(1.0,))
    freqs, mags = magnitude_spectrum(x, sample_rate)
    bin_width = freqs[1] - freqs[0]
    assert abs(freqs[np.argmax(mags)] - 440.0) <= bin_width


def test_find_peaks_returns_distinct_fundamentals():
    sample_rate = 8000
    t = np.arange(8192) / sample_rate
    x = np.sin(2 * np.pi * 440 * t) + 0.8 * np.sin(2 * np.pi * 660 * t) + 0.6 * np.sin(2 * np.pi * 550 * t)
    freqs, mags = magnitude_spectrum(x, sample_rate)
    peaks = find_peaks(freqs, mags, n_peaks=3, min_separation_hz=20.0)
    assert sorted(np.round(peaks / 10) * 10) == [440, 550, 660]


@pytest.mark.parametrize("freq, name", [(440.0, "A4"), (261.63, "C4"), (880.0, "A5"),
                                        (554.37, "C#5"), (659.25, "E5"), (27.5, "A0")])
def test_frequency_to_note_name(freq, name):
    assert frequency_to_note_name(freq) == name


def test_lowpass_removes_high_keeps_low():
    """Bin-aligned frequencies so the brick-wall filter is exact."""
    sample_rate = 16384
    t = np.arange(sample_rate) / sample_rate  # exactly 1 s, N = 16384
    low = np.sin(2 * np.pi * 440 * t)
    high = np.sin(2 * np.pi * 5000 * t)
    filtered = lowpass_filter(low + high, sample_rate, cutoff_hz=2000.0)
    np.testing.assert_allclose(filtered, low, atol=1e-9)
    freqs, mags = magnitude_spectrum(filtered, sample_rate)
    assert mags[np.argmin(np.abs(freqs - 5000))] < 1e-9


def test_hann_window_overlap_adds_to_constant():
    """The periodic Hann window at 50 % hop satisfies the COLA property."""
    n, hop = 64, 32
    w = hann_window(n)
    total = np.zeros(n + 4 * hop)
    for start in range(0, len(total) - n + 1, hop):
        total[start:start + n] += w
    interior = total[n - hop: len(total) - n + hop]
    np.testing.assert_allclose(interior, 1.0, atol=1e-12)


def test_spectrogram_tracks_a_frequency_step():
    sample_rate = 8000
    t = np.arange(4000) / sample_rate
    x = np.concatenate([np.sin(2 * np.pi * 500 * t), np.sin(2 * np.pi * 1500 * t)])
    times, freqs, S = spectrogram(x, sample_rate, window_size=512, hop=128)
    assert S.shape == (len(freqs), len(times))
    first, last = freqs[np.argmax(S[:, 0])], freqs[np.argmax(S[:, -1])]
    assert abs(first - 500) < 20 and abs(last - 1500) < 20