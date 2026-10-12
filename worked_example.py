"""One FFT of length 8, every intermediate number printed.

Run from this folder: python worked_example.py
The README walks through the same numbers by hand.
"""

import numpy as np

from fft.dft import dft_matrix, dft_naive
from fft.fft import fft_recursive

np.set_printoptions(precision=4, suppress=True, linewidth=100)

X_INPUT = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])


def fmt(z: complex) -> str:
    return f"{z.real:+.4f} {z.imag:+.4f}j"


def main() -> None:
    x = X_INPUT
    n = len(x)
    print(f"x = {x}\n")

    print("Step 1: split into even- and odd-indexed samples")
    evens, odds = x[0::2], x[1::2]
    print(f"  x_even = {evens}    x_odd = {odds}\n")

    print("Step 2: the length-4 DFT matrix, omega_4 = exp(-2 pi i / 4) = -i")
    F4 = dft_matrix(4)
    for row in F4:
        print("  [ " + "  ".join(fmt(z) for z in row) + " ]")
    print()

    print("Step 3: transform each half    E = F4 x_even,    O = F4 x_odd")
    E, O = F4 @ evens, F4 @ odds
    for k in range(4):
        print(f"  E_{k} = {fmt(E[k])}    O_{k} = {fmt(O[k])}")
    print()

    print("Step 4: twiddle factors omega_8^k = exp(-2 pi i k / 8), k = 0..3")
    w = np.exp(-2j * np.pi * np.arange(4) / n)
    for k in range(4):
        print(f"  omega^{k} = {fmt(w[k])}")
    print()

    print("Step 5: butterflies    X_k = E_k + omega^k O_k,    X_{k+4} = E_k - omega^k O_k")
    X = np.empty(n, dtype=complex)
    for k in range(4):
        t = w[k] * O[k]
        X[k], X[k + 4] = E[k] + t, E[k] - t
        print(f"  omega^{k} O_{k} = {fmt(t)}")
        print(f"  X_{k} = {fmt(X[k])}    X_{k + 4} = {fmt(X[k + 4])}")
    print()

    print("Check against the O(N^2) definition and the recursive code")
    print(f"  max |X - dft_naive(x)|     = {np.max(np.abs(X - dft_naive(x))):.2e}")
    print(f"  max |X - fft_recursive(x)| = {np.max(np.abs(X - fft_recursive(x))):.2e}\n")

    print("Parseval: sum |x_n|^2 = (1/N) sum |X_k|^2")
    print(f"  sum |x_n|^2       = {np.sum(np.abs(x) ** 2):.4f}")
    print(f"  (1/8) sum |X_k|^2 = {np.sum(np.abs(X) ** 2) / n:.4f}")

if __name__ == "__main__":
    main()