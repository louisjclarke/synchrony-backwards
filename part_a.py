"""Part A: synchrony hides its cause, and sparse regression finds the real links.

Run:  python part_a.py
"""
import numpy as np
from synchrony import make_network, simulate, fit, stlsq, truth_row

rng = np.random.default_rng(1)
dt = 0.01
K, omega = make_network()
n = len(omega)

# One long run: the network locks, so the phase differences freeze.
_, long = simulate(rng.uniform(0, 2 * np.pi, n), K, omega, 200)
_, cond_locked = fit([long[-2000:]], 0, dt)

# Twenty short runs from random phases: each captures the fall into step.
runs = [simulate(rng.uniform(0, 2 * np.pi, n), K, omega, 3)[1] for _ in range(20)]
errs, conds = [], []
for i in range(n):
    beta, c = fit(runs, i, dt)
    errs.append(np.abs(beta - truth_row(K, omega, i)).max())
    conds.append(c)
print(f"condition number: one locked run {cond_locked:.3g}, twenty restarts {np.median(conds):.3g}")
print(f"largest error in any coupling with restarts: {max(errs):.4f}")

# Noise: plain least squares against sparse regression.
noisy = [th + rng.normal(0, 0.05, th.shape) for th in runs]
kept = false_sparse = false_plain = 0
for i in range(n):
    real = truth_row(K, omega, i)[1:] > 0
    b_sparse = stlsq(noisy, i, dt)
    b_plain, _ = fit(noisy, i, dt)
    kept += (b_sparse[1:] != 0)[real].sum()
    false_sparse += (b_sparse[1:] != 0)[~real].sum()
    false_plain += (np.abs(b_plain[1:]) > 0)[~real].sum()
print(f"sparse regression kept {kept} of {int((K > 0).sum())} real links and added {false_sparse} false ones")
print(f"plain least squares left {false_plain} non-links non-zero")

# Figure: the network locking (left) and the couplings recovered from restarts (right).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_long = np.arange(len(long)) * dt
r = np.abs(np.exp(1j * long).mean(axis=1))
true_all, found_all = [], []
for i in range(n):
    beta, _ = fit(runs, i, dt)
    true_all.append(truth_row(K, omega, i)[1:])
    found_all.append(beta[1:])
true_all, found_all = np.concatenate(true_all), np.concatenate(found_all)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(t_long, r, color="#2a6f97")
a1.set_xlabel("time")
a1.set_ylabel("order parameter r (1 = perfect step)")
a1.set_title("One long run: the network locks")
lim = [true_all.min() - 0.3, true_all.max() + 0.3]
a2.plot(lim, lim, color="#999999", lw=1)
a2.scatter(true_all, found_all, s=18, color="#c8553d")
a2.set_xlabel("true coupling")
a2.set_ylabel("recovered coupling")
a2.set_title("Twenty restarts: every coupling recovered")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_part_a.png", dpi=150)
print("saved figures/fig_part_a.png")
