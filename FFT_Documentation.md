# Fast Fourier Transform from Scratch

A Complete Implementation, Verification, and Applications Guide

---

## Overview

A radix-2 Cooley–Tukey FFT written in pure NumPy, derived from the definition of the discrete Fourier transform, verified against the $O(N^2)$ definition, against `numpy.fft`, and against four theorems, then used to read the notes off a chord, denoise it, and draw a spectrogram. `numpy.fft` appears only in the tests.

| Dimension | Details |
|-----------|---------|
| **Mathematics** | Complex numbers, roots of unity, orthogonality, divide and conquer, Parseval's theorem, the convolution theorem |
| **Verification** | 78 tests: matches `numpy.fft.fft` to $10^{-10}$ for $N = 1, \dots, 1024$; measured cost slope 2.1 for the naive sum and below 1 for the FFT |
| **Result** | The three notes of an A-major chord identified from the spectrum; 0 dB to 10.5 dB signal-to-noise by zeroing bins |

---

## 1. The Idea in Plain Words

A sound wave is a list of numbers, one per sample: 44,100 of them for each second of CD-quality audio. The **discrete Fourier transform** (DFT) asks a different question of the same list: *how much of each frequency is in it?* It answers by taking the dot product of the signal with a sampled cosine and sine at every frequency the samples can represent.

Written out, that is $N$ dot products each of length $N$: $N^2$ multiplications. For one second of audio, about two billion. The **fast Fourier transform** (FFT) gets the identical answer with about $N \log_2 N$ multiplications: roughly 700,000. It does this by noticing that the DFT of a signal can be assembled from the DFTs of its even-indexed and odd-indexed halves, and that the assembly costs only $N$ operations. Apply the same idea to each half, and to each quarter, and so on.

### DFT Basis Functions

Each output $X_k$ is the dot product of the signal with one of these rows. Row $k$ is a cosine (real part) and a negative sine (imaginary part) that complete $k$ cycles over the $N$ samples.

---

## 2. Worked Example: An FFT of Length 8 by Hand

Take the signal

$$
x = \begin{bmatrix} 1 & 2 & 3 & 4 & 0 & 0 & 0 & 0 \end{bmatrix}
$$

Run `python worked_example.py` to see every number below printed by the code.

### 2.1 Split into Evens and Odds

$$
x_{\text{even}} = \begin{bmatrix} 1 & 3 & 0 & 0 \end{bmatrix}, \qquad
x_{\text{odd}} = \begin{bmatrix} 2 & 4 & 0 & 0 \end{bmatrix}
$$

### 2.2 Transform Each Half with the Length-4 DFT Matrix

With $\omega_4 = e^{-2\pi i/4} = -i$, the entry in row $k$, column $j$ of the DFT matrix is $\omega_4^{kj}$:

$$
F_4 = \begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & -i & -1 & i \\
1 & -1 & 1 & -1 \\
1 & i & -1 & -i
\end{bmatrix}
$$

Multiplying,

$$
E = F_4\, x_{\text{even}} = \begin{bmatrix} 4 \\ 1 - 3i \\ -2 \\ 1 + 3i \end{bmatrix}, \qquad
O = F_4\, x_{\text{odd}} = \begin{bmatrix} 6 \\ 2 - 4i \\ -2 \\ 2 + 4i \end{bmatrix}
$$

Check one entry by hand: $E_1 = 1 \cdot 1 + 3 \cdot (-i) + 0 + 0 = 1 - 3i$.

### 2.3 The Twiddle Factors

With $\omega_8 = e^{-2\pi i/8}$, the first four powers are

$$
\omega_8^0 = 1, \qquad
\omega_8^1 = \tfrac{1}{\sqrt 2}(1 - i) \approx 0.7071 - 0.7071i, \qquad
\omega_8^2 = -i, \qquad
\omega_8^3 = \tfrac{1}{\sqrt 2}(-1 - i) \approx -0.7071 - 0.7071i
$$

### 2.4 The Butterflies

Each pair $(E_k, O_k)$ produces two outputs:

$$
X_k = E_k + \omega_8^k\, O_k, \qquad X_{k+4} = E_k - \omega_8^k\, O_k
$$

| $k$ | $\omega_8^k O_k$ | $X_k$ | $X_{k+4}$ |
|---|---|---|---|
| 0 | $6$ | $X_0 = 10$ | $X_4 = -2$ |
| 1 | $\tfrac{1}{\sqrt 2}(1-i)(2-4i) = \tfrac{1}{\sqrt 2}(-2-6i) \approx -1.4142 - 4.2426i$ | $X_1 \approx -0.4142 - 7.2426i$ | $X_5 \approx 2.4142 + 1.2426i$ |
| 2 | $(-i)(-2) = 2i$ | $X_2 = -2 + 2i$ | $X_6 = -2 - 2i$ |
| 3 | $\tfrac{1}{\sqrt 2}(-1-i)(2+4i) = \tfrac{1}{\sqrt 2}(2-6i) \approx 1.4142 - 4.2426i$ | $X_3 \approx 2.4142 - 1.2426i$ | $X_7 \approx -0.4142 + 7.2426i$ |

Eight outputs from two length-4 transforms and four complex multiplications. Computing the same eight numbers from the definition takes $64$ multiplications.

### 2.5 Two Checks

$X_0 = 10$ is the sum of the samples, as it must be, since row 0 of the DFT matrix is all ones. And Parseval's theorem holds:

$$
\sum_{n=0}^{7} \lvert x_n \rvert^2 = 1 + 4 + 9 + 16 = 30, \qquad
\frac{1}{8} \sum_{k=0}^{7} \lvert X_k \rvert^2 = \frac{240}{8} = 30
$$

The code agrees with these numbers to $10^{-15}$.

---

## 3. Derivations

### 3.1 The DFT and Its Matrix

For a signal $x_0, \dots, x_{N-1}$ and $\omega = e^{-2\pi i/N}$,

$$
X_k = \sum_{n=0}^{N-1} x_n\, \omega^{kn}, \qquad k = 0, \dots, N-1
$$

As a matrix, $X = F x$ with $F_{kn} = \omega^{kn}$. The powers of $\omega$ are the $N$-th roots of unity, equally spaced around the unit circle.

The columns of $F$ are orthogonal. Take two columns $j$ and $m$ and form the inner product

$$
\sum_k \overline{\omega^{kj}}\, \omega^{km} = \sum_k \omega^{k(m-j)}
$$

Let $r = \omega^{m-j}$. If $j = m$ then $r = 1$ and the sum is $N$. Otherwise $r \ne 1$ but $r^N = 1$, and the geometric series gives

$$
\sum_{k=0}^{N-1} r^k = \frac{1 - r^N}{1 - r} = \frac{1 - 1}{1 - r} = 0
$$

So $F^{*} F = N I$, where $F^{*}$ is the conjugate transpose.

### 3.2 The Inverse Transform

From $F^{*} F = N I$, the inverse is $F^{-1} = \tfrac{1}{N} F^{*}$, that is

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k\, \omega^{-kn}
$$

The only differences from the forward transform are the sign of the exponent and the factor $1/N$. Conjugating flips the sign of the exponent, which gives the trick used in the implementation: $\operatorname{IFFT}(X) = \tfrac{1}{N} \overline{\operatorname{FFT}(\overline{X})}$.

One piece of code serves both directions.

### 3.3 The Even–Odd Split

Separate the sum into even and odd indices, $n = 2m$ and $n = 2m + 1$:

$$
X_k = \sum_{m=0}^{N/2-1} x_{2m}\, \omega^{2mk} + \omega^{k} \sum_{m=0}^{N/2-1} x_{2m+1}\, \omega^{2mk}
$$

Since $\omega^2 = e^{-2\pi i/(N/2)}$ is the root of unity for length $N/2$, both sums are length-$N/2$ DFTs. Call them $E_k$ and $O_k$:

$$
X_k = E_k + \omega^k O_k
$$

$E_k$ and $O_k$ are periodic with period $N/2$, and $\omega^{N/2} = e^{-\pi i} = -1$, so

$$
\omega^{k + N/2} = -\omega^k
\quad\Longrightarrow\quad
X_{k+N/2} = E_k - \omega^k O_k
$$

This is the **divide-and-conquer recurrence**: each length-$N$ DFT costs $2N$ plus the cost of two length-$N/2$ DFTs. Unroll it as a tree with $\log_2 N$ levels, each costing $N$, for a total of $N \log_2 N$.

---

## 4. Verification & Tests

### Test Coverage

| Test | Criterion | Tolerance |
|------|-----------|-----------|
| Matches `numpy.fft.fft` | For random complex input, $N = 1, \dots, 1024$ | $10^{-10}$ |
| Matches naive $O(N^2)$ DFT | All length-8 and length-16 real and complex inputs | $10^{-10}$ |
| Parseval's theorem | $\sum_n \lvert x_n \rvert^2 = \tfrac{1}{N} \sum_k \lvert X_k \rvert^2$ | $10^{-12}$ |
| Orthogonality | $\cos(2\pi k_0 n / N)$ has energy only in bins $k_0$ and $N - k_0$ | $10^{-9}$ |
| Power-of-two requirement | Lengths $3, 6, 12, 1000$ raise `ValueError` | Exact |
| Convolution theorem | `convolve_fft` matches `np.convolve` for four length pairs | $10^{-9}$ |
| Peak finding | A 440 Hz tone peaks within one bin of 440 Hz | One bin |
| Note identification | $440 \to$ A4, $261.63 \to$ C4, $880 \to$ A5, $27.5 \to$ A0 | Exact |
| Low-pass filtering | Removes a 5 kHz sine and leaves a 440 Hz sine untouched | $10^{-9}$ |

### Test Results

```
python -m pytest tests/ -q
78 passed in 7.12s
```

### Numerical Accuracy

Maximum disagreement with `numpy.fft.fft` on random complex input:

| $N$ | Iterative | Recursive |
|-----|-----------|-----------|
| $1\,024$ | $7.2 \times 10^{-14}$ | $6.6 \times 10^{-14}$ |
| $16\,384$ | $4.3 \times 10^{-13}$ | $4.4 \times 10^{-13}$ |
| $131\,072$ | $1.6 \times 10^{-12}$ | $1.7 \times 10^{-12}$ |

---

## 5. Results

All audio in this project is **synthesised** from sine waves so that it has no external data dependency. `run.py` explains how to substitute a real recording with `read_wav`.

### 5.1 Reading the Notes Off a Chord

An A-major chord (A4, C♯5, E5) with second and third harmonics at half and quarter amplitude, 2 seconds at 44.1 kHz, saved as `audio/chord.wav`.

The FFT of the 88,200 samples (padded to 131,072) and a greedy peak finder give:

| Peak | Frequency | Note | What It Is |
|------|-----------|------|-----------|
| 1 | 440.08 Hz | A4 | Fundamental |
| 2 | 554.48 Hz | C♯5 | Fundamental |
| 3 | 659.12 Hz | E5 | Fundamental |
| 4 | 1318.57 Hz | E6 | 2nd harmonic of E5 |
| 5 | 1108.62 Hz | C♯6 | 2nd harmonic of C♯5 |
| 6 | 879.83 Hz | A5 | 2nd harmonic of A4 |
| 7 | 1663.10 Hz | G♯6 | 3rd harmonic of C♯5 |
| 8 | 1977.69 Hz | B6 | 3rd harmonic of E5 |

The three loudest peaks are the three notes. The harmonics fall exactly where the synthesiser put them, and every frequency is within one bin (0.34 Hz) of the true value. The 3rd harmonic of A4 at 1320 Hz coincides with the 2nd harmonic of E5 at 1318.5 Hz, which is why an A-major chord has eight distinct peaks rather than nine, and why A and E sound consonant together.

### 5.2 Denoising

White noise is added at 0 dB signal-to-noise ratio (noise power equal to signal power). Every frequency bin above 2 kHz is set to zero and the signal transformed back.

| | SNR |
|---|---|
| Noisy | $-0.01$ dB |
| After low-pass at 2 kHz | $10.45$ dB |

White noise spreads its power evenly over all 22,050 Hz of bandwidth. Keeping 2,000 Hz keeps $2000 / 22050 = 9.1\,\%$ of the noise power, a reduction of $10 \log_{10}(1/0.091) = 10.4$ dB, which is what was measured. The chord itself lives entirely below 2 kHz and is untouched.

### 5.3 Spectrogram

A C-major arpeggio (C4, E4, G4, C5, half a second each) at 8 kHz, analysed with a 1024-sample Hann window and a 128-sample hop. Each note shows as three horizontal lines: fundamental, 2nd harmonic, 3rd harmonic.

The vertical bands at the note changes are real, not artefacts of the plot: the waveform jumps discontinuously from one note to the next, and a jump contains every frequency. A click is broadband.

### 5.4 Computational Cost

**Best of three runs, random real input, single CPU core:**

| $N$ | Naive $O(N^2)$ | Recursive FFT | Iterative FFT |
|-----|---|---|---|
| 64 | 0.29 ms | 0.41 ms | 0.08 ms |
| 256 | 1.07 ms | 1.67 ms | 0.12 ms |
| 1,024 | 10.0 ms | 6.7 ms | 0.22 ms |
| 4,096 | 117 ms | 29.1 ms | 0.52 ms |
| 16,384 | 4,490 ms | 144 ms | 2.15 ms |

**Log-log slope over the five largest sizes:**

- **Naive sum:** 2.13 (theoretical: 2)
- **Recursive FFT:** 1.11 (theoretical: slightly above 1)
- **Iterative FFT:** 0.86 (theoretical: slightly above 1)

At $N = 16\,384$ the iterative FFT is **2,000 times faster** than the definition.

---

## 6. Numerical Lessons Learned

### Spectral Leakage: A Ninth Peak That Is Not a Note

The peak finder's ninth-tallest peak sits at 424 Hz, amplitude 0.0016, labelled G♯4. There is no G♯ in the chord. The 2-second signal is zero-padded to 131,072 samples, which is the same as multiplying an infinite signal by a rectangular window. The spectrum of a rectangle is a sinc, so every true peak grows a comb of sidelobes on either side. Each fundamental in the spectrum figure has a visible skirt for this reason. The fix in real work is to taper the signal with a window (Hann, Hamming) before transforming; the spectrogram does exactly this and its lines are clean. Left untapered here on purpose, because the sidelobe is worth seeing.

### Recursion Is Slower Than the Naive Sum Until About $N = 512$

A length-$N$ recursive FFT makes $2N - 1$ Python function calls, and each call costs a few microseconds regardless of how little arithmetic it does. Below $N = 512$ that overhead matches or exceeds the $N^2$ arithmetic of the naive sum, which runs as a single vectorised dot product per output. The iterative version does $\log_2 N$ vectorised stages with no function calls and wins at every size. The asymptotic argument is right, but constants decide who wins on small inputs.

### Brick-Wall Filtering Rings

Zeroing every bin above the cutoff is multiplying the spectrum by a rectangle. By the convolution theorem that is convolving the signal with a sinc, whose tails extend far in both directions. Near any sharp change in the signal, the filtered output oscillates. The test in `tests/test_spectrum.py` avoids this by using frequencies that fall exactly on bins and a signal whose length is a power of two, so the filter is exact. The claim "the chord is untouched" holds because the chord's components all fall well below the cutoff and have no sharp edges within the window.

### Why the DFT Matrix Is Fine at $N = 8$ and Impossible at $N = 16\,384$

The matrix has $N^2$ complex entries at 16 bytes each: 1 KB for the worked example, 4.3 GB for the largest timing size. The naive reference therefore computes one row at a time, indexing the $N$ precomputed powers of $\omega$ with $(kn) \bmod N$, which is $O(N)$ memory and still $O(N^2)$ time. Using $\omega^N = 1$ to reduce the exponent is the same fact that makes the even–odd split work.

### 16-Bit Audio Clips

A WAV file stores integers in $[-32768, 32767]$. Three tones of unit amplitude sum to a peak near 3, which would be clipped to 1 and sound distorted. The chord is scaled to a peak of 0.9 before writing, and the noisy version, whose samples exceed 1 in magnitude about 2 % of the time (maximum 1.76), is clipped and slightly distorted on disk (the analysis uses the unclipped floats).

---

## 7. How to Run

```bash
pip install -r ../requirements.txt

python -m pytest tests/ -v     # 78 tests, about 7 seconds

python worked_example.py       # the length-8 FFT with every number printed

python run.py                  # chord analysis, denoising, spectrogram, timing; about 20 s

python make_figures.py         # the three conceptual diagrams
```

To analyse your own recording, save it as 16-bit PCM WAV and in `run.py` replace `synthesize_chord(...)` with `read_wav(Path("your_file.wav"))`, which returns the samples and the sample rate.

---

## 8. Code Map

| File | Contents |
|------|----------|
| `fft/dft.py` | `dft_matrix`, `dft_naive`, `idft_naive`: the $O(N^2)$ definition, used as the reference |
| `fft/fft.py` | `fft_recursive`, `fft_iterative`, `bit_reverse_indices`, `ifft`, `pad_to_power_of_two` |
| `fft/spectrum.py` | `magnitude_spectrum`, `find_peaks`, `frequency_to_note_name`, `convolve_fft`, `lowpass_filter`, `hann_window`, `spectrogram` |
| `fft/audio.py` | `synthesize_tone`, `synthesize_chord`, `add_noise`, `snr_db`, `write_wav`, `read_wav` |
| `tests/test_fft.py` | The transform against NumPy, the definition, and five theorems |
| `tests/test_spectrum.py` | Convolution, peaks, notes, filtering, windows, spectrogram |
| `tests/test_complexity.py` | Measured cost slopes |
| `worked_example.py` | Section 2 worked example |
| `run.py` | Section 5 applications |
| `make_figures.py` | Sections 1, 3 figures |

---

**Documentation Version:** 1.0  
**Last Updated:** September 2026  
**Format:** Markdown with LaTeX Mathematics (MathJax compatible)
