"""Synchrony in resting EEG: phases, pairwise measures and features (Part B).

Data: PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al., 2004;
Goldberger et al., 2000). Run 1 is one minute resting with eyes open and
run 2 is one minute with eyes closed.
"""
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import mne
import numpy as np
import pooch
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import hilbert

# Electrodes over the back of the head, where the alpha rhythm is strongest.
REAR = ["PO3", "POz", "PO4", "O1", "Oz", "O2", "P1", "P2"]
PAIRS = np.tril_indices(len(REAR), -1)  # the 28 electrode pairs
SFREQ = 160.0                           # samples per second
ALPHA = (8.0, 12.0)                     # alpha band, Hz
EPOCH_SECONDS = 2.0


# PhysioNet's own copy on AWS Open Data is much faster than physionet.org, which is the fallback.
SOURCES = ["https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/",
           "https://physionet.org/files/eegmmidb/1.0.0/"]
RUNS = [1, 2]  # 1 = resting with eyes open, 2 = resting with eyes closed


def data_dir():
    """The folder where MNE keeps this dataset, so mne.datasets.eegbci.load_data finds the files."""
    root = mne.get_config("MNE_DATASETS_EEGBCI_PATH", mne.get_config("MNE_DATA"))
    root = Path(root).expanduser() if root else Path.home() / "mne_data"
    return root / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"


def checksums():
    """The published SHA-256 checksum of every file in the dataset, as shipped with MNE."""
    text = files("mne").joinpath("data", "eegbci_checksums.txt").read_text()
    return dict(line.split() for line in text.splitlines() if line.strip())


def fetch(s, dest=None, sums=None):
    """Download volunteer s's two recordings, unless already present.

    Each file is checked against its published checksum, whichever server it comes from,
    and a stalled connection gives up after 60 seconds rather than hanging.
    """
    dest, sums = dest or data_dir(), sums or checksums()
    for r in RUNS:
        part = f"S{s:03d}/S{s:03d}R{r:02d}.edf"
        if (dest / part).exists():
            continue
        error = None
        for url in SOURCES:
            try:
                pooch.retrieve(url + part, known_hash=f"sha256:{sums[part]}",
                               fname=Path(part).name, path=dest / Path(part).parent,
                               downloader=pooch.HTTPDownloader(timeout=60))
                error = None
                break
            except Exception as e:
                error = e
        if error is not None:
            raise error


def download(subjects, workers=8, dest=None):
    """Fetch every volunteer's recordings, several at a time.

    Returns the volunteers whose files could not be fetched from either server.
    """
    sums = checksums()

    def get(s):
        try:
            fetch(s, dest, sums)
        except Exception as e:
            print(f"download failed for volunteer {s}: {e}", flush=True)
            return s
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [s for s in pool.map(get, subjects) if s is not None]


def load_subject(s):
    """Volunteer s as [eyes open, eyes closed], with standard electrode names, at 160 Hz.

    A few volunteers were recorded at a different rate, so they are resampled.
    """
    raws = []
    for f in mne.datasets.eegbci.load_data(s, RUNS, update_path=False):
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)  # e.g. "O1.." becomes "O1"
        raw.set_montage("standard_1005")
        if raw.info["sfreq"] != SFREQ:
            raw.resample(SFREQ)
        raws.append(raw)
    return raws


def alpha_phase(raw, picks=REAR):
    """Phase and amplitude of the alpha rhythm at each electrode.

    The Hilbert transform turns each filtered signal into a point circling
    the origin of the complex plane: its angle is the phase and its distance
    from the origin is the amplitude.
    """
    x = raw.copy().pick(picks).filter(*ALPHA, verbose=False).get_data()
    z = hilbert(x, axis=1)
    return np.unwrap(np.angle(z), axis=1), np.abs(z)


def pair_measures(raw):
    """Alpha-band phase-locking value and |imaginary coherency| for the 28 rear pairs.

    The phase-locking value counts any steady phase relation, including the
    zero-delay agreement produced when neighbouring electrodes pick up the
    same source. Imaginary coherency ignores zero-delay mixing (Nolte et al., 2004).
    """
    ep = mne.make_fixed_length_epochs(raw, duration=EPOCH_SECONDS, preload=True,
                                      verbose=False).pick(REAR)
    plv, imc = spectral_connectivity_epochs(ep, method=["plv", "imcoh"], fmin=ALPHA[0],
                                            fmax=ALPHA[1], faverage=True, verbose=False)
    return (plv.get_data(output="dense")[:, :, 0][PAIRS],
            np.abs(imc.get_data(output="dense")[:, :, 0][PAIRS]))


def features(raw):
    """Phase-locking values (28), |imaginary coherency| (28) and log alpha power for one recording."""
    plv, imc = pair_measures(raw)
    _, amp = alpha_phase(raw)
    return plv, imc, float(np.log(np.mean(amp ** 2)))


def nolte_check(seed=0, seconds=60):
    """|Imaginary coherency| for two channel pairs that share one 10 Hz source.

    The first pair mixes it with no delay, the second with a quarter-period
    delay. Nolte's result says the first should be near 0 and the second clearly not.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / SFREQ)
    src = np.sin(2 * np.pi * 10 * t) + 0.5 * rng.normal(size=t.size)
    n = int(EPOCH_SECONDS * SFREQ)

    def noise():
        return 0.3 * rng.normal(size=t.size)

    def imcoh(a, b):
        data = np.stack([a, b]).reshape(2, -1, n).transpose(1, 0, 2)  # (epochs, channels, samples)
        c = spectral_connectivity_epochs(data, method="imcoh", sfreq=SFREQ, fmin=ALPHA[0],
                                         fmax=ALPHA[1], faverage=True, verbose=False)
        return float(abs(c.get_data(output="dense")[1, 0, 0]))

    a = src + noise()
    return imcoh(a, 0.7 * src + noise()), imcoh(a, np.roll(src, 4) + noise())
