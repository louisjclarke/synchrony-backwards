import numpy as np

from synchrony import make_network, simulate, fit, stlsq, truth_row

DT = 0.01


def _runs(K, omega, rng, count=20, t_end=3):
    n = len(omega)
    return [simulate(rng.uniform(0, 2 * np.pi, n), K, omega, t_end)[1] for _ in range(count)]


def test_recovers_couplings():
    """With restarts and no noise, every coupling comes back to within 0.01."""
    K, omega = make_network()
    runs = _runs(K, omega, np.random.default_rng(1))
    for i in range(len(omega)):
        beta, _ = fit(runs, i, DT)
        assert np.abs(beta - truth_row(K, omega, i)).max() < 0.01


def test_locked_run_is_ill_conditioned():
    """Synchrony hides its cause: a locked run is far worse conditioned than restarts."""
    K, omega = make_network()
    rng = np.random.default_rng(1)
    _, long = simulate(rng.uniform(0, 2 * np.pi, len(omega)), K, omega, 200)
    _, cond_locked = fit([long[-2000:]], 0, DT)
    _, cond_restarts = fit(_runs(K, omega, rng), 0, DT)
    assert cond_locked > 1000 * cond_restarts


def test_sparse_regression_drops_non_links():
    """With 0.05 rad of noise, sparse regression keeps every real link and no false ones."""
    K, omega = make_network()
    rng = np.random.default_rng(1)
    noisy = [th + rng.normal(0, 0.05, th.shape) for th in _runs(K, omega, rng)]
    for i in range(len(omega)):
        real = truth_row(K, omega, i)[1:] > 0
        found = stlsq(noisy, i, DT)[1:] != 0
        assert np.array_equal(found, real)
