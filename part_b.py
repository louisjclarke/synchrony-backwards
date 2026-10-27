"""Part B: does brain synchrony change when the eyes close, once zero-delay mixing is ignored?

Run:  python part_b.py            (about 20 minutes; downloads about 300 MB from PhysioNet the first time)
      python part_b.py --cached   (seconds; reuses results/eeg_features.npz)
Writes results/part_b.json, results/eeg_features.npz, figures/fig_eyes_closed.png and
figures/fig_accuracy.png, and the Part B table in README.md.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg import features, load_subject, nolte_check
from report import save_json, update_readme

ROOT = Path(__file__).resolve().parent
VOLUNTEERS = range(1, 110)


def collect(subjects, loader):
    """Features for every volunteer; any volunteer whose download or processing fails is skipped."""
    F = {k: [] for k in ("plv", "imc", "pow", "label", "group")}
    skipped = []
    for s in subjects:
        try:
            rows = [features(raw) for raw in loader(s)]  # [eyes open, eyes closed]
        except Exception as e:
            skipped.append(int(s))
            print(f"skipped volunteer {s}: {e}")
            continue
        for label, (plv, imc, pw) in enumerate(rows):  # 0 = eyes open, 1 = eyes closed
            for k, v in zip(F, (plv, imc, pw, label, s)):
                F[k].append(v)
        print(f"volunteer {s} done")
    return {k: np.array(v) for k, v in F.items()}, skipped


def per_volunteer(M, F, subs):
    """Each volunteer's average over the 28 pairs (or their log power), eyes open and eyes closed."""
    o = np.array([M[(F["group"] == s) & (F["label"] == 0)].mean() for s in subs])
    c = np.array([M[(F["group"] == s) & (F["label"] == 1)].mean() for s in subs])
    return o, c


def analyse(F, skipped, figures_dir):
    subs = [s for s in np.unique(F["group"]) if set(F["label"][F["group"] == s]) == {0, 1}]
    N = len(subs)
    plv_o, plv_c = per_volunteer(F["plv"], F, subs)
    imc_o, imc_c = per_volunteer(F["imc"], F, subs)
    pow_o, pow_c = per_volunteer(F["pow"], F, subs)

    # Paired test: each volunteer is compared only with themselves.
    imc_p = wilcoxon(imc_c, imc_o).pvalue
    plv_p = wilcoxon(plv_c, plv_o).pvalue

    # Classifier tested on volunteers it has never seen: folds split by volunteer.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = GroupKFold(n_splits=5)
    y, g = F["label"], F["group"]
    acc = {name: cross_val_score(clf, X, y, groups=g, cv=cv)
           for name, X in [("coherency", F["imc"]),
                           ("alpha power", F["pow"].reshape(-1, 1)),
                           ("both", np.column_stack([F["imc"], F["pow"]]))]}

    nolte_none, nolte_delayed = nolte_check()
    res = dict(
        volunteers=N, skipped=skipped,
        alpha_power_ratio_median=float(np.median(np.exp(pow_c - pow_o))),
        nolte_check=dict(no_delay=nolte_none, delayed=nolte_delayed),
        eyes_open_mean=dict(plv=float(plv_o.mean()), imcoh=float(imc_o.mean())),
        eyes_closed_mean=dict(plv=float(plv_c.mean()), imcoh=float(imc_c.mean())),
        imcoh_higher_closed=int((imc_c > imc_o).sum()), imcoh_wilcoxon_p=float(imc_p),
        plv_higher_closed=int((plv_c > plv_o).sum()), plv_wilcoxon_p=float(plv_p),
        accuracy={k: dict(mean=float(a.mean()), folds=[float(x) for x in a]) for k, a in acc.items()},
    )

    # Figure 1: every volunteer, eyes open against eyes closed.
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (o, c, name) in zip(axes, [(plv_o, plv_c, "phase-locking value"),
                                       (imc_o, imc_c, "|imaginary coherency|")]):
        for i in range(N):
            ax.plot([0, 1], [o[i], c[i]], color="grey", alpha=0.3)
        ax.plot([0, 1], [np.median(o), np.median(c)], color="#c8553d", lw=3, label="median")
        ax.set_xticks([0, 1], ["eyes open", "eyes closed"])
        ax.set_title(f"{name}, rear electrode pairs")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_eyes_closed.png", dpi=150)

    # Figure 2: classifier accuracy on unseen volunteers.
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(acc)
    ax.bar(names, [100 * acc[k].mean() for k in names], color="#2a6f97")
    for i, k in enumerate(names):
        ax.plot([i] * len(acc[k]), 100 * acc[k], "o", color="black", ms=4)
    ax.axhline(50, ls="--", color="grey", label="chance")
    ax.set(ylabel="accuracy on unseen volunteers (%)", ylim=(0, 105),
           title="Eyes open or closed?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_accuracy.png", dpi=150)
    plt.close("all")
    return res


def run(subjects=VOLUNTEERS, loader=load_subject, out=ROOT, cached=False):
    """The whole analysis. Writes results and figures under `out` and returns the results."""
    path = out / "results" / "eeg_features.npz"
    if cached:
        data = np.load(path)
        F, skipped = {k: data[k] for k in ("plv", "imc", "pow", "label", "group")}, list(data["skipped"])
    else:
        mne.set_log_level("ERROR")
        F, skipped = collect(subjects, loader)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, skipped=np.array(skipped, dtype=int), **F)
    res = analyse(F, [int(s) for s in skipped], out / "figures")
    save_json(res, out / "results" / "part_b.json")
    return res


def summary(r):
    a = r["accuracy"]
    pct = lambda k: f"{100 * a[k]['mean']:.0f}% (folds {100 * min(a[k]['folds']):.0f}-{100 * max(a[k]['folds']):.0f}%)"
    return f"""
| | |
|---|---|
| volunteers analysed | {r['volunteers']} |
| alpha power, eyes closed ÷ eyes open (median) | {r['alpha_power_ratio_median']:.1f} times |
| Nolte check, \\|imaginary coherency\\| with no delay / with delay | {r['nolte_check']['no_delay']:.3f} / {r['nolte_check']['delayed']:.3f} |
| eyes open, rear pairs: phase-locking value / \\|imaginary coherency\\| | {r['eyes_open_mean']['plv']:.2f} / {r['eyes_open_mean']['imcoh']:.2f} |
| \\|imaginary coherency\\| higher with eyes closed | {r['imcoh_higher_closed']} of {r['volunteers']} volunteers (Wilcoxon p = {r['imcoh_wilcoxon_p']:.2g}) |
| phase-locking value higher with eyes closed | {r['plv_higher_closed']} of {r['volunteers']} volunteers (Wilcoxon p = {r['plv_wilcoxon_p']:.2g}) |
| classifier accuracy on unseen volunteers: coherency | {pct('coherency')} |
| classifier accuracy on unseen volunteers: alpha power alone | {pct('alpha power')} |
| classifier accuracy on unseen volunteers: both | {pct('both')} |
"""


if __name__ == "__main__":
    results = run(cached="--cached" in sys.argv)
    table = summary(results)
    print(table)
    update_readme("part-b", table)
