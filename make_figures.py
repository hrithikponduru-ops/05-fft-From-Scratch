"""Conceptual diagrams for README.md (data-independent). Writes to figures/.

Run from this folder: python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, BLUE, GREEN, RED, GREY = "#1f2937", "#2563eb", "#059669", "#dc2626", "#9ca3af"

def fig_roots_of_unity():
    n = 8
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(circle), np.sin(circle), color=GREY, lw=0.8)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.axvline(0, color=GREY, lw=0.5)
    for k in range(n):
        color = BLUE if k < n // 2 else RED
        ax.plot([0, w[k].real], [0, w[k].imag], color=color, lw=1.5)
        ax.plot(w[k].real, w[k].imag, "o", color=color, ms=7)
        ax.text(w[k].real * 1.2, w[k].imag * 1.2, f"$\\omega^{{{k}}}$", ha="center", va="center", fontsize=12, color=color)
    ax.annotate("", xy=(w[5].real, w[5].imag), xytext=(w[1].real, w[1].imag),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1, ls="--"))
    ax.text(0.08, -0.32, "$\\omega^{5} = -\\omega^{1}$", fontsize=11, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.set_title("The 8th roots of unity, $\\omega = e^{-2\\pi i/8}$\nopposite points differ only by sign")
    fig.tight_layout()
    fig.savefig("figures/roots_of_unity.png", dpi=150)
    plt.close(fig)

def fig_butterfly():
    n = 8
    fig, ax = plt.subplots(figsize=(11, 5.2))
    stages = 3
    x_in, x_out = 0.0, 1.0
    xs = np.linspace(x_in + 0.18, x_out - 0.12, stages + 1)
    order = [0, 4, 2, 6, 1, 5, 3, 7]  # bit-reversed input order
    y = lambda i: n - 1 - i  # noqa: E731

    for i, idx in enumerate(order):
        ax.text(x_in, y(i), f"$x_{{{{{idx}}}}}$", ha="right", va="center", fontsize=12)
        ax.text(x_out, y(i), f"$X_{{{{{i}}}}}$", ha="left", va="center", fontsize=12)
        ax.plot([x_in + 0.02, xs[0]], [y(i), y(i)], color=GREY, lw=1)
        ax.plot([xs[-1], x_out - 0.02], [y(i), y(i)], color=GREY, lw=1)

    for s in range(stages):
        size = 2 ** (s + 1)
        half = size // 2
        x0, x1 = xs[s], xs[s + 1]
        for block in range(0, n, size):
            for j in range(half):
                top, bot = block + j, block + j + half
                ax.plot([x0, x1], [y(top), y(top)], color=BLUE, lw=1.2)
                ax.plot([x0, x1], [y(bot), y(bot)], color=BLUE, lw=1.2)
                ax.plot([x0, x1], [y(bot), y(top)], color=RED, lw=1, alpha=0.8)
                ax.plot([x0, x1], [y(top), y(bot)], color=RED, lw=1, alpha=0.8)
                ax.text(x0 + 0.012, y(bot) + 0.08, f"$\\omega_{{{{{size}}}}}^{{{{{j}}}}}$", fontsize=8, color=RED)
        ax.text((x0 + x1) / 2, n - 0.3, f"stage {s + 1}\nblocks of {size}", ha="center", fontsize=9, color=INK)

    ax.text((x_in + xs[0]) / 2, n - 0.3, "bit-reversed\ninput order", ha="center", fontsize=9, color=INK)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.6, n + 0.4)
    ax.axis("off")
    ax.set_title("Iterative FFT for N = 8: three stages of butterflies (red = multiply by a twiddle, then add / subtract)")
    fig.tight_layout()
    fig.savefig("figures/butterfly.png", dpi=150)
    plt.close(fig)

def fig_basis_functions():
    n = 32
    t = np.arange(n)
    fig, axes = plt.subplots(2, 4, figsize=(13, 4.6), sharex=True, sharey=True)
    for col, k in enumerate((0, 1, 2, 5)):
        basis = np.exp(-2j * np.pi * k * t / n)
        fine = np.linspace(0, n - 1, 400)
        axes[0, col].plot(fine, np.cos(2 * np.pi * k * fine / n), color=GREY, lw=0.8)
        axes[0, col].stem(t, basis.real, linefmt=BLUE, markerfmt="o", basefmt=" ")
        axes[1, col].plot(fine, -np.sin(2 * np.pi * k * fine / n), color=GREY, lw=0.8)
        axes[1, col].stem(t, basis.imag, linefmt=GREEN, markerfmt="o", basefmt=" ")
        axes[0, col].set_title(f"k = {k}: real part, $\\cos(2\\pi k n / N)$", fontsize=9)
        axes[1, col].set_title(f"k = {k}: imaginary part, $-\\sin(2\\pi k n / N)$", fontsize=9)
        axes[1, col].set_xlabel("sample n")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-1.3, 1.3)
    fig.suptitle(f"Rows of the DFT matrix for N = {n}: X_k is the dot product of the signal with row k", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/basis_functions.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    fig_roots_of_unity()
    fig_butterfly()
    fig_basis_functions()
    print("wrote figures/")