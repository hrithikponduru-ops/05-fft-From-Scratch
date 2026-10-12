"""Radix-2 Cooley-Tukey FFT, two ways: recursive and iterative.

Both rely on one identity. Split the DFT sum into even- and odd-indexed
terms. Each half is a DFT of length N/2 (call them E_k and O_k), and with
omega = exp(-2 pi i / N):

    X_k       = E_k + omega^k O_k
    X_{k + N/2} = E_k - omega^k O_k      because omega^{N/2} = -1

One pair of half-size transforms therefore yields *all* N outputs. Doing
this recursively costs T(N) = 2 T(N/2) + O(N), which solves to O(N log N).
"""

import numpy as np

def _check_power_of_two(n: int) -> None:
    if n < 1 or n & (n - 1):
        raise ValueError(f"radix-2 FFT needs a power-of-two length, got {n}")

def pad_to_power_of_two(x: np.ndarray) -> np.ndarray:
    """Zero-pad x to the next power of two (unchanged if already one)."""
    x = np.asarray(x)
    n = len(x)
    target = 1 << max(n - 1, 0).bit_length() if n > 1 else 1
    return np.concatenate([x, np.zeros(target - n, dtype=x.dtype)])

def fft_recursive(x: np.ndarray) -> np.ndarray:
    """The textbook recursion: transform the evens, transform the odds, combine."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    _check_power_of_two(n)
    if n == 1:
        return x.copy()
    evens = fft_recursive(x[0::2])
    odds = fft_recursive(x[1::2])
    twiddles = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    t = twiddles * odds
    return np.concatenate([evens + t, evens - t])

def bit_reverse_indices(n: int) -> np.ndarray:
    """Permutation that sorts indices by their binary digits read backwards.

    Recursing on evens/odds and unwinding is the same as sorting the input
    by reversed index bits. For N = 8: 0 1 2 3 4 5 6 7 -> 0 4 2 6 1 5 3 7.
    """
    bits = n.bit_length() - 1
    idx = np.arange(n)
    reversed_idx = np.zeros(n, dtype=int)
    for b in range(bits):
        reversed_idx |= ((idx >> b) & 1) << (bits - 1 - b)
    return reversed_idx

def fft_iterative(x: np.ndarray) -> np.ndarray:
    """Same arithmetic as the recursion, without the recursion.

    Permute the input into bit-reversed order, then run log2(N) stages of
    butterflies. Stage s combines blocks of size 2^s. Each stage is written
    as one vectorised operation over all blocks at once, working on a fresh
    array rather than mutating in place, so the input is never modified.
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    _check_power_of_two(n)
    a = x[bit_reverse_indices(n)]
    size = 2
    while size <= n:
        half = size // 2
        twiddles = np.exp(-2j * np.pi * np.arange(half) / size)
        blocks = a.reshape(-1, size)
        top = blocks[:, :half]
        bottom = blocks[:, half:] * twiddles
        a = np.concatenate([top + bottom, top - bottom], axis=1).reshape(-1)
        size *= 2
    return a

def ifft(X: np.ndarray) -> np.ndarray:
    """Inverse via the conjugation trick: ifft(X) = conj(fft(conj(X))) / N.

    The inverse DFT differs from the forward one only in the sign of the
    exponent and a factor 1/N. Conjugating the input flips the sign, and
    conjugating the output flips it back.
    """
    X = np.asarray(X, dtype=complex)
    return np.conj(fft_iterative(np.conj(X))) / len(X)