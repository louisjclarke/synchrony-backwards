"""Two explainer figures: why locking hides the couplings, and why imaginary coherency ignores zero delay.

Run:  python explainers.py      (a second or two)
Writes figures/fig_line_of_best_fit.png and figures/fig_arrows.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BLUE, RED, GREY = "#2a6f97", "#c8553d", "#9a9a9a"


def line_of_best_fit(path):
    """Two coupled oscillators. Oscillator 1's speed is a straight line in x = sin(theta2 - theta1):
    intercept omega1, slope K. While they fall into step the points trace the line; once locked they
    pile up at one point, and lines of any slope fit equally well."""
    w1, w2, K, dt, T = 1.0, 1.6, 0.5, 0.01, 40
    th = np.zeros((int(T / dt), 2))
    th[0] = [0.0, -2.8]
    for k in range(len(th) - 1):
        a, b = th[k]
        th[k + 1] = [a + dt * (w1 + K * np.sin(b - a)), b + dt * (w2 + K * np.sin(a - b))]
    t = np.arange(len(th)) * dt
    rng = np.random.default_rng(3)
    x = np.sin(th[:, 1] - th[:, 0])
    y = np.gradient(th[:, 0], dt) + rng.normal(0, 0.02, len(t))   # measured speed, with a little noise
    early, late, step = t < 8, t > 30, 10

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    a1.scatter(x[early][::step], y[early][::step], s=10, color=BLUE, alpha=0.7)
    slope, icpt = np.polyfit(x[early], y[early], 1)
    xx = np.linspace(-1.05, 1.05, 2)
    a1.plot(xx, icpt + slope * xx, color=RED, lw=2)
    a1.text(-1.0, 1.42, f"slope = {slope:.2f}  (true K = {K})\nintercept = {icpt:.2f}  (true ω = {w1})",
            fontsize=9.5)
    a1.set(xlabel=r"$\sin(\theta_2-\theta_1)$", ylabel="speed of oscillator 1",
           title="While falling into step", xlim=(-1.1, 1.1))
    a2.scatter(x[late][::step], y[late][::step], s=10, color=BLUE, alpha=0.7)
    x0, y0 = x[late].mean(), y[late].mean()
    for s, ls in [(0.5, "-"), (-0.6, "--"), (2.0, ":")]:
        a2.plot(xx, y0 + s * (xx - x0), color=RED, lw=1.6, ls=ls)
    a2.set(xlabel=r"$\sin(\theta_2-\theta_1)$", title="Once locked: every line fits",
           xlim=(-1.1, 1.1), ylim=(0.2, 1.7))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def arrows(path):
    """Phase-difference arrows on the Argand diagram. Zero-delay mixing puts every arrow on the real
    axis (or pointing backwards along it), so the average has no imaginary part; a delay tilts them."""
    rng = np.random.default_rng(7)
    cases = [("Same source, no delay", np.r_[np.zeros(22), np.full(8, np.pi)]),
             ("Real interaction, with a delay", rng.normal(1.0, 0.45, 30))]
    th = np.linspace(0, 2 * np.pi, 300)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, (title, d) in zip(axes, cases):
        ax.plot(np.cos(th), np.sin(th), color="#cccccc", lw=1)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.axvline(0, color="#bbbbbb", lw=0.8)
        for a in d:
            ax.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8, alpha=0.6, mutation_scale=8))
        m = np.exp(1j * d).mean()
        ax.annotate("", xy=(m.real, m.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=3, mutation_scale=16))
        if abs(m.imag) > 0.05:
            ax.plot([m.real, m.real], [0, m.imag], ls="--", color=BLUE, lw=1.8)
            ax.text(m.real + 0.07, m.imag / 2 - 0.1, "imaginary\npart", color=BLUE, fontsize=10, va="center")
        else:
            ax.text(-0.95, -0.4, "every arrow points along\nthe real axis, so the\nimaginary part is 0",
                    color=BLUE, fontsize=10)
        ax.text(1.12, 0.04, "real", fontsize=9, color="#777777")
        ax.text(0.04, 1.1, "imaginary", fontsize=9, color="#777777")
        ax.set(xlim=(-1.25, 1.35), ylim=(-1.25, 1.25), aspect="equal")
        ax.axis("off")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    line_of_best_fit(ROOT / "figures" / "fig_line_of_best_fit.png")
    arrows(ROOT / "figures" / "fig_arrows.png")
    print("saved figures/fig_line_of_best_fit.png and figures/fig_arrows.png")


if __name__ == "__main__":
    main()
