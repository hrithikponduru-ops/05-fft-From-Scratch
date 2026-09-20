"""Apply the hand-written FFT to audio: read notes off a chord, denoise, spectrogram, timing.

The audio is SYNTHESISED (a chord built from sine waves with harmonics) so the
project has no external data dependency. To analyse a real recording instead,
replace `synthesize_chord(...)` with `read_wav(Path("your_file.wav"))`.

Run from this folder: python run.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fft.audio import add_noise, snr_db, synthesize_chord, write_wav
from fft.dft import dft_naive
from fft.fft import fft_iterative, fft_recursive
from fft.spectrum import find_peaks, frequency_to_note_name, lowpass_filter, magnitude_spectrum, spectrogram

SAMPLE_RATE = 44_100
DURATION = 2.0
A_MAJOR = (440.0, 554.37, 659.25)  # A4, C#5, E5
C_MAJOR_ARPEGGIO = (261.63, 329.63, 392.00, 523.25)  # C4, E4, G4, C5
NOISE_SNR_DB = 0.0
CUTOFF_HZ = 2000.0
N_PEAKS = 8  # 3 fundamentals + 5 distinct harmonics (A5 and E6 coincide with the 2nd harmonics)
TIMING_SIZES = [2 ** p for p in range(6, 15)]
SEED = 0
AUDIO = Path("audio")
FIGURES = Path("figures")
BLUE, GREEN, RED, GREY = "#2563eb", "#059669", "#dc2626", "#9ca3af"

def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("NOTE: all audio below is synthesised; see the docstring to use a real recording.\n")

    chord = synthesize_chord(A_MAJOR, DURATION, SAMPLE_RATE)
    write_wav(AUDIO / "chord.wav", chord, SAMPLE_RATE)
    freqs, mags = magnitude_spectrum(chord, SAMPLE_RATE)
    print(f"chord: {len(chord)} samples, padded to {2 * (len(freqs) - 1)}, bin width {freqs[1]:.3f} Hz")
    peaks = find_peaks(freqs, mags, n_peaks=N_PEAKS + 1, min_separation_hz=15.0)
    print(f"{N_PEAKS} tallest peaks (fundamentals first, then harmonics):")
    for f in peaks[:N_PEAKS]:
        print(f"  {f:8.2f} Hz -> {frequency_to_note_name(f)}")
    fundamentals = sorted(peaks[:3])
    print(f"detected chord: {[frequency_to_note_name(f) for f in fundamentals]}")
    leak = peaks[N_PEAKS]
    print(f"peak number {N_PEAKS + 1} is at {leak:.2f} Hz with amplitude "
          f"{mags[np.argmin(np.abs(freqs - leak))]:.4f}: not a note but a leakage sidelobe "
          f"of the {peaks[0]:.0f} Hz fundamental (see README, numerical lessons)\n")
    peaks = peaks[:N_PEAKS]
    fig_waveform(chord)
    fig_spectrum(freqs, mags, peaks)

    noisy = add_noise(chord, NOISE_SNR_DB, rng)
    denoised = lowpass_filter(noisy, SAMPLE_RATE, CUTOFF_HZ)
    write_wav(AUDIO / "chord_noisy.wav", noisy, SAMPLE_RATE)
    write_wav(AUDIO / "chord_denoised.wav", denoised, SAMPLE_RATE)
    print(f"SNR before low-pass: {snr_db(chord, noisy):6.2f} dB")
    print(f"SNR after  low-pass: {snr_db(chord, denoised):6.2f} dB  (cutoff {CUTOFF_HZ:.0f} Hz)\n")
    fig_denoise(chord, noisy, denoised)

    fig_spectrogram()
    timing_table()

def fig_waveform(chord):
    t = np.arange(len(chord)) / SAMPLE_RATE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
    ax1.plot(t, chord, color=BLUE, lw=0.4)
    ax1.set_xlabel("time (s)")
    ax1.set_title("A-major chord, all 2 seconds")
    zoom = slice(0, int(0.02 * SAMPLE_RATE))
    ax2.plot(t[zoom] * 1000, chord[zoom], color=BLUE, lw=1.2)
    ax2.set_xlabel("time (ms)")
    ax2.set_title("First 20 ms: three notes plus harmonics, summed")
    for ax in (ax1, ax2):
        ax.set_ylabel("amplitude")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "waveform.png", dpi=150)
    plt.close(fig)

def fig_spectrum(freqs, mags, peaks):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    keep = freqs <= 2500
    ax.plot(freqs[keep], mags[keep], color=BLUE, lw=1)
    for i, f in enumerate(sorted(peaks)):
        m = mags[np.argmin(np.abs(freqs - f))]
        ax.annotate(f"{frequency_to_note_name(f)}\n{f:.0f} Hz", xy=(f, m), xytext=(0, 8 + 22 * (i % 2)),
                    textcoords="offset points", ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.set_ylim(0, mags.max() * 1.35)
    ax.set_title("Spectrum of the chord from the hand-written FFT: the notes are the peaks")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "spectrum.png", dpi=150)
    plt.close(fig)

def fig_denoise(chord, noisy, denoised):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    t_ms = np.arange(len(chord)) / SAMPLE_RATE * 1000
    zoom = slice(int(0.5 * SAMPLE_RATE), int(0.5 * SAMPLE_RATE) + int(0.01 * SAMPLE_RATE))
    for col, (name, sig, color) in enumerate((("clean", chord, BLUE), ("noisy, 0 dB SNR", noisy, RED),
                                              (f"low-passed at {CUTOFF_HZ:.0f} Hz", denoised, GREEN))):
        f, m = magnitude_spectrum(sig, SAMPLE_RATE)
        keep = f <= 6000
        axes[0, col].plot(f[keep], m[keep], color=color, lw=0.7)
        axes[0, col].set_title(name)
        axes[0, col].set_xlabel("frequency (Hz)")
        axes[0, col].axvline(CUTOFF_HZ, color=GREY, ls="--", lw=0.8)
        axes[1, col].plot(t_ms[zoom], sig[zoom], color=color, lw=1)
        axes[1, col].set_xlabel("time (ms)")
        axes[1, col].set_ylim(-1.6, 1.6)
    axes[0, 0].set_ylabel("amplitude spectrum")
    axes[1, 0].set_ylabel("waveform, 10 ms window")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Denoising by zeroing frequency bins above the cutoff (dashed line)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "denoise.png", dpi=150)
    plt.close(fig)

def fig_spectrogram():
    sample_rate, note_len, window, hop = 8000, 0.5, 1024, 128
    x = np.concatenate([synthesize_chord((f,), note_len, sample_rate, peak=0.8) for f in C_MAJOR_ARPEGGIO])
    times, freqs, S = spectrogram(x, sample_rate, window, hop)
    fig, ax = plt.subplots(figsize=(10, 4))
    keep = freqs <= 1800
    ax.pcolormesh(times, freqs[keep], 20 * np.log10(S[keep] + 1e-6), cmap="magma", shading="auto", vmin=-40)
    for i, (f, name) in enumerate(zip(C_MAJOR_ARPEGGIO, ("C4", "E4", "G4", "C5"))):
        ax.text((i + 0.5) * note_len, f + 90, f"{name}   {f:.0f} Hz", ha="center", fontsize=8, color="white")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Spectrogram of a C-major arpeggio (window {window}, hop {hop}): \n"
                 "each note is a fundamental plus two harmonics", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "spectrogram.png", dpi=150)
    plt.close(fig)

def best_time(fn, x, repeats):
    return min(_timed(fn, x) for _ in range(repeats))

def _timed(fn, x):
    t0 = time.perf_counter()
    fn(x)
    return time.perf_counter() - t0

def timing_table():
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{'N':>6} {'naive O(N^2)':>14} {'recursive':>11} {'iterative':>11}")
    for n in TIMING_SIZES:
        x = rng.normal(size=n)
        row = (n, best_time(dft_naive, x, 1), best_time(fft_recursive, x, 3), best_time(fft_iterative, x, 3))
        rows.append(row)
        print(f"{row[0]:>6} {row[1]:>13.5f}s {row[2]:>10.5f}s {row[3]:>10.5f}s")
    rows = np.array(rows)
    for name, col in (("naive", 1), ("recursive", 2), ("iterative", 3)):
        slope = np.polyfit(np.log(rows[-5:, 0]), np.log(rows[-5:, col]), 1)[0]
        print(f"log-log slope over the five largest N, {name:>9}: {slope:.2f}")
    fig_complexity(rows)

def fig_complexity(rows):
    n = rows[:, 0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(n, rows[:, 1], "o-", color=RED, label="naive DFT, O(N^2)")
    ax.loglog(n, rows[:, 2], "s-", color=GREEN, label="recursive FFT")
    ax.loglog(n, rows[:, 3], "^-", color=BLUE, label="iterative FFT")
    ax.loglog(n, rows[:, 1] * (n / n[-1]) ** 2, ":", color=RED, lw=1, label="slope 2 reference")
    ax.loglog(n, rows[:, 3] * (n * np.log2(n)) / (n[-1] * np.log2(n[-1])), ":", color=BLUE, lw=1,
              label="N log N reference")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("seconds")
    ax.set_title("Run time of the three transforms")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "complexity.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()