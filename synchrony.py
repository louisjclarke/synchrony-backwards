"""Recovering who is coupled to whom from rhythms alone (Kuramoto model).

Part A of the project: simulate a network of coupled oscillators, then
recover its couplings by least squares and by sparse regression (SINDy's
sequentially thresholded least squares).
"""
import numpy as np
from scipy.integrate import solve_ivp


def make_network(n=8, p=0.3, strength=5.0, seed=0):
    """A random sparse network: K[i, j] is how strongly j pulls on i."""
    r = np.random.default_rng(seed)
    A = (r.random((n, n)) < p).astype(float)
    np.fill_diagonal(A, 0)
    K = strength * A * r.uniform(0.3, 0.8, (n, n))
    omega = r.normal(0, 1.0, n)
    return K, omega


def kuramoto_rhs(t, theta, K, omega):
    """Speed of every oscillator: omega_i + sum_j K[i, j] sin(theta_j - theta_i)."""
    D = theta[None, :] - theta[:, None]
    return omega + (K * np.sin(D)).sum(axis=1)


def simulate(theta0, K, omega, t_end, dt=0.01):
    """Run the model forward. Returns times and phases, shape (time, n)."""
    t = np.arange(0, t_end, dt)
    sol = solve_ivp(kuramoto_rhs, (0, t[-1]), theta0, t_eval=t,
                    args=(K, omega), rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T


def design(theta, i, dt):
    """Linear system for oscillator i: columns [1, sin(theta_j - theta_i)], target = speed."""
    y = np.gradient(theta[:, i], dt)
    others = [j for j in range(theta.shape[1]) if j != i]
    cols = [np.ones(len(theta))] + [np.sin(theta[:, j] - theta[:, i]) for j in others]
    return np.column_stack(cols), y


def fit(runs, i, dt):
    """Least squares over several runs. Returns coefficients and the condition number."""
    Xs, ys = zip(*[design(th, i, dt) for th in runs])
    X, y = np.vstack(Xs), np.concatenate(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, np.linalg.cond(X)


def stlsq(runs, i, dt, threshold=0.5, iters=10):
    """Sparse regression: zero small couplings, refit on the survivors, repeat."""
    Xs, ys = zip(*[design(th, i, dt) for th in runs])
    X, y = np.vstack(Xs), np.concatenate(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(iters):
        keep = np.abs(beta) >= threshold
        keep[0] = True
        beta = np.zeros_like(beta)
        beta[keep], *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
    return beta


def truth_row(K, omega, i):
    """The true coefficients for oscillator i, in the same order as design()."""
    return np.r_[omega[i], np.delete(K[i], i)]
