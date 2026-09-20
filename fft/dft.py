"""The discrete Fourier transform straight from its definition: O(N^2).

X_k = sum_{n=0}^{N-1} x_n omega^{k n},   omega = exp(-2 pi i / N)

This is the *reference* the fast algorithm is tested against. It is slow on
purpose: one length-N dot product per output, N outputs.
"""

import numpy as np

def dft_matrix(n: int) -> np.ndarray:
    """The N x N matrix F with F[k, j] = omega^{k j}. Used in the worked example."""
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)

def dft_naive(x: np.ndarray) -> np.ndarray:
    """X_k = sum_n x_n omega^{k n}, computed one k at a time.

    The exponent k*n is reduced mod N because omega^N = 1, so only the N
    distinct powers of omega are ever needed. This also keeps memory O(N):
    the full N x N matrix would need 4 GB at N = 16384.
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 0:
        return x.copy()
    omega_powers = np.exp(-2j * np.pi * np.arange(n) / n)
    idx = np.arange(n)
    return np.array([x @ omega_powers[(k * idx) % n] for k in range(n)])

def idft_naive(X: np.ndarray) -> np.ndarray:
    """x_n = (1/N) sum_k X_k omega^{-k n}: the conjugate transform divided by N."""
    X = np.asarray(X, dtype=complex)
    n = len(X)
    if n == 0:
        return X.copy()
    return np.conj(dft_naive(np.conj(X))) / n