import json

import mne
import numpy as np

from eeg import REAR, SFREQ, alpha_phase, download, nolte_check
from part_b import run

OTHERS = ["Fz", "Cz", "C3", "C4"]


def _raw(rng, closed, seconds=61):
    """Fake EEG: a shared 10 Hz source at the rear electrodes, three times stronger with eyes closed,
    plus a second source that reaches each electrode with a small delay."""
    t = np.arange(0, seconds, 1 / SFREQ)
    amp = 3.0 if closed else 1.0
    s1 = amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
    s2 = amp * np.sin(2 * np.pi * 10.5 * t + rng.uniform(0, 6))
    data = rng.normal(size=(len(REAR) + len(OTHERS), t.size))
    for k in range(len(REAR)):
        data[k] += s1 + 0.5 * np.roll(s2, 2 * (k % 3))
    info = mne.create_info(REAR + OTHERS, SFREQ, "eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose=False)


def test_alpha_phase_tracks_a_10_hz_rhythm():
    """The Hilbert phase of a pure 10 Hz signal advances by 2 pi x 10 radians a second."""
    t = np.arange(0, 20, 1 / SFREQ)
    data = np.tile(np.sin(2 * np.pi * 10 * t), (len(REAR), 1)) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(REAR, SFREQ, "eeg"), verbose=False)
    phase, _ = alpha_phase(raw)
    middle = slice(int(2 * SFREQ), int(18 * SFREQ))  # ignore filter edges
    rate = np.polyfit(t[middle], phase[0, middle], 1)[0]
    assert abs(rate / (2 * np.pi) - 10) < 0.05


def test_imaginary_coherency_ignores_zero_delay_mixing():
    """Nolte et al. (2004): mixing without delay cannot fake a lagged interaction."""
    no_delay, delayed = nolte_check()
    assert no_delay < 0.05
    assert delayed > 0.5


def test_whole_analysis_runs_on_synthetic_volunteers(tmp_path):
    """End to end on ten fake volunteers, one of whom fails to load and must be skipped."""
    rng = np.random.default_rng(0)

    def loader(s):
        if s == 4:
            raise RuntimeError("simulated download failure")
        return [_raw(rng, closed=False), _raw(rng, closed=True)]

    res = run(subjects=range(1, 11), loader=loader, out=tmp_path)
    assert res["volunteers"] == 9 and res["skipped"] == [4]
    assert res["alpha_power_ratio_median"] > 5
    assert res["accuracy"]["alpha power"]["mean"] > 0.9
    for f in ["results/part_b.json", "results/eeg_features.npz",
              "figures/fig_eyes_closed.png", "figures/fig_accuracy.png"]:
        assert (tmp_path / f).exists()
    assert json.loads((tmp_path / "results/part_b.json").read_text())["volunteers"] == 9
    cached = run(out=tmp_path, cached=True)
    assert cached["accuracy"] == res["accuracy"]


def test_download_runs_in_parallel_and_reports_failures(monkeypatch):
    """Every volunteer is requested once, and a failed download is reported, not fatal."""
    asked = []

    def fake_load(s, runs, **kw):
        asked.append(s)
        if s == 3:
            raise OSError("simulated network failure")
        return [f"S{s:03d}R{r:02d}.edf" for r in runs]

    monkeypatch.setattr(mne.datasets.eegbci, "load_data", fake_load)
    assert download(range(1, 11), workers=4) == [3]
    assert sorted(asked) == list(range(1, 11))
