"""Part A: synchrony hides its cause, and sparse regression finds the real links.

Run:  python part_a.py      (a few seconds)
Writes results/part_a.json and figures/fig_part_a.png, and the Part A table in README.md.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report import save_json, sci, update_readme
from synchrony import fit, make_network, simulate, stlsq, truth_row

ROOT = Path(__file__).resolve().parent
DT = 0.01
NOISE = 0.05       # measurement noise on the phases, radians
THRESHOLD = 0.5    # sparse regression drops couplings smaller than this


def main():
    rng = np.random.default_rng(1)
    K, omega = make_network()
    n = len(omega)

    # 1. One long run: the network locks, so the phase differences freeze.
    _, long = simulate(rng.uniform(0, 2 * np.pi, n), K, omega, 200)
    _, cond_locked = fit([long[-2000:]], 0, DT)

    # 2. Twenty short runs from random phases, each capturing the fall into step.
    runs = [simulate(rng.uniform(0, 2 * np.pi, n), K, omega, 3)[1] for _ in range(20)]
    fits = [fit(runs, i, DT) for i in range(n)]
    true = np.array([truth_row(K, omega, i) for i in range(n)])
    found = np.array([beta for beta, _ in fits])
    max_error = np.abs(found - true).max()
    cond_restarts = np.median([c for _, c in fits])

    # 3. With noise: plain least squares against sparse regression.
    noisy = [th + rng.normal(0, NOISE, th.shape) for th in runs]
    kept = false_sparse = false_plain = 0
    for i in range(n):
        real = truth_row(K, omega, i)[1:] > 0
        b_sparse = stlsq(noisy, i, DT, threshold=THRESHOLD)
        b_plain, _ = fit(noisy, i, DT)
        kept += int((b_sparse[1:] != 0)[real].sum())
        false_sparse += int((b_sparse[1:] != 0)[~real].sum())
        false_plain += int((b_plain[1:] != 0)[~real].sum())
    links = int((K > 0).sum())
    non_links = n * (n - 1) - links

    res = dict(oscillators=n, real_links=links, cond_locked=float(cond_locked),
               cond_restarts_median=float(cond_restarts), max_coupling_error=float(max_error),
               noise_sd=NOISE, threshold=THRESHOLD, sparse_kept=kept,
               sparse_false=false_sparse, plain_nonzero_non_links=false_plain, non_links=non_links)
    save_json(res, ROOT / "results" / "part_a.json")

    print(f"condition number: one locked run {cond_locked:.3g}, twenty restarts {cond_restarts:.3g}")
    print(f"largest error in any coupling with restarts: {max_error:.4f}")
    print(f"sparse regression kept {kept} of {links} real links and added {false_sparse} false ones")
    print(f"plain least squares left {false_plain} of {non_links} non-links non-zero")

    update_readme("part-a", f"""
| | |
|---|---|
| condition number of X, one locked run | {sci(cond_locked)} |
| condition number of X, twenty restarts (median) | {cond_restarts:.2g} |
| largest error in any recovered coupling, twenty restarts | {max_error:.4f} |
| sparse regression with {NOISE} rad noise: real links kept | {kept} of {links} |
| sparse regression with {NOISE} rad noise: false links | {false_sparse} |
| plain least squares with {NOISE} rad noise: non-links left non-zero | {false_plain} of {non_links} |
""")

    # Figure: the network locking (left) and the recovered couplings (right).
    t = np.arange(len(long)) * DT
    r = np.abs(np.exp(1j * long).mean(axis=1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(t, r, color="#2a6f97")
    a1.set(xlabel="time", ylabel="order parameter r", ylim=(0, 1.02),
           title="One long run: the network locks")
    lim = [true[:, 1:].min() - 0.3, true[:, 1:].max() + 0.3]
    a2.plot(lim, lim, color="#999999", lw=1)
    a2.scatter(true[:, 1:].ravel(), found[:, 1:].ravel(), s=18, color="#c8553d")
    a2.set(xlabel="true coupling", ylabel="recovered coupling",
           title="Twenty restarts: every coupling recovered")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig_part_a.png", dpi=150)


if __name__ == "__main__":
    main()
