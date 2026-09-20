"""The hand-written FFT against the definition, NumPy, and four theorems.

numpy.fft is a *reference* here. It is never imported by the fft package.
"""

import numpy as np
import pytest

from fft.dft import dft_naive, idft_naive
from fft.fft import fft_iterative, fft_recursive, ifft, pad_to_power_of_two

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def random_signal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) + 1j * rng.normal(size=n)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fft", [fft_recursive, fft_iterative])
def test_matches_numpy(fft, n):
    x = random_signal(n, seed=n)
    np.testing.assert_allclose(fft(x), np.fft.fft(x), atol=1e-10)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fft", [fft_recursive, fft_iterative])
def test_matches_naive_definition(fft, n):
    """An independent reference: the O(N^2) sum written straight from the formula."""
    x = random_signal(n, seed=100 + n)
    np.testing.assert_allclose(fft(x), dft_naive(x), atol=1e-9)


def test_naive_inverse_round_trip():
    x = random_signal(64, seed=3)
    np.testing.assert_allclose(idft_naive(dft_naive(x)), x, atol=1e-12)


@pytest.mark.parametrize("n", [1, 8, 256, 1024])
def test_ifft_round_trip(n):
    x = random_signal(n, seed=7)
    np.testing.assert_allclose(ifft(fft_iterative(x)), x, atol=1e-12)


def test_parseval():
    """Energy in time equals energy in frequency (divided by N)."""
    x = random_signal(512, seed=11)
    X = fft_iterative(x)
    assert np.sum(np.abs(x) ** 2) == pytest.approx(np.sum(np.abs(X) ** 2) / len(x), rel=1e-12)


def test_linearity():
    x, y = random_signal(128, seed=1), random_signal(128, seed=2)
    a, b = 2.0 - 1.0j, -0.5 + 3.0j
    np.testing.assert_allclose(
        fft_iterative(a * x + b * y), a * fft_iterative(x) + b * fft_iterative(y), atol=1e-10
    )


def test_shift_theorem():
    """Delaying x by m samples multiplies X_k by exp(-2 pi i k m / N)."""
    n, m = 64, 5
    x = random_signal(n, seed=5)
    shifted = np.roll(x, m)  # shifted[j] = x[j - m]
    k = np.arange(n)
    expected = fft_iterative(x) * np.exp(-2j * np.pi * k * m / n)
    np.testing.assert_allclose(fft_iterative(shifted), expected, atol=1e-10)


@pytest.mark.parametrize("n", [3, 6, 12, 1000])
@pytest.mark.parametrize("fft", [fft_recursive, fft_iterative])
def test_non_power_of_two_rejected(fft, n):
    with pytest.raises(ValueError):
        fft(np.zeros(n))


def test_pad_to_power_of_two():
    padded = pad_to_power_of_two(np.arange(5.0))
    assert len(padded) == 8
    np.testing.assert_array_equal(padded[:5], np.arange(5.0))
    assert np.all(padded[5:] == 0)
    assert len(pad_to_power_of_two(np.arange(8.0))) == 8


def test_pure_cosine_lands_in_two_bins():
    """cos(2 pi k0 n / N) has energy only at k0 and N - k0, each of magnitude N/2."""
    n, k0 = 256, 17
    x = np.cos(2 * np.pi * k0 * np.arange(n) / n)
    X = fft_iterative(x)
    mags = np.abs(X)
    assert mags[k0] == pytest.approx(n / 2, rel=1e-10)
    assert mags[n - k0] == pytest.approx(n / 2, rel=1e-10)
    others = np.delete(mags, [k0, n - k0])
    assert np.max(others) < 1e-9