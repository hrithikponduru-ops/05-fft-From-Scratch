"""The whole point of the FFT: O(N log N) instead of O(N^2), measured not asserted."""

import time

import numpy as np

from fft.dft import dft_naive
from fft.fft import fft_iterative

SIZES = [2 ** p for p in range(10, 15)]  # 1024 .. 16384


def best_of(fn, x, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        best = min(best, time.perf_counter() - t0)
    return best


def loglog_slope(sizes, times):
    slope, _ = np.polyfit(np.log(sizes), np.log(times), 1)
    return slope


def test_naive_is_quadratic_and_fft_is_not():
    rng = np.random.default_rng(0)
    signals = {n: rng.normal(size=n) for n in SIZES}
    naive = [best_of(dft_naive, signals[n], repeats=1) for n in SIZES]
    fast = [best_of(fft_iterative, signals[n]) for n in SIZES]
    assert loglog_slope(SIZES, naive) > 1.7, naive
    assert loglog_slope(SIZES, fast) < 1.5, fast
    assert fast[-1] < naive[-1] / 20