# Synchrony backwards

![tests](https://github.com/YOUR-USERNAME/synchrony-backwards/actions/workflows/tests.yml/badge.svg)
[![Open Part B in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/synchrony-backwards/blob/main/notebooks/part_b_brain_rhythms.ipynb)

Most models of synchronisation run forwards: given who is coupled to whom, predict the rhythm. This project runs one backwards. Given only the rhythms, can you work out the couplings, and when can that answer be trusted?

It started with Steven Strogatz's two-line model of Romeo and Juliet, led to the Kuramoto model of coupled oscillators, and ends with open EEG recordings of 109 volunteers.

## The model

In the Kuramoto model on a network, oscillator *i* has its own natural frequency ω<sub>i</sub> and is pulled by the oscillators it is coupled to:

dθ<sub>i</sub>/dt = ω<sub>i</sub> + Σ<sub>j</sub> K<sub>ij</sub> sin(θ<sub>j</sub> − θ<sub>i</sub>)

The equation is nonlinear in the phases but **linear in the unknowns** ω<sub>i</sub> and K<sub>ij</sub>. So if the phases are recorded, each time step gives one linear equation, and stacking them gives a linear system **X β = y**:

- each row of X is [1, sin(θ<sub>1</sub> − θ<sub>i</sub>), …, sin(θ<sub>n</sub> − θ<sub>i</sub>)] at one moment,
- y is the measured rate of change of θ<sub>i</sub>,
- β = (ω<sub>i</sub>, K<sub>i1</sub>, …, K<sub>in</sub>) is what we want.

It is solved by least squares, one oscillator at a time.

## Part A: simulations (`part_a.py`)

An 8-oscillator random network with 18 real links, simulated with SciPy's `solve_ivp`.

**1. Synchrony hides its cause.** Run the network for a long time and it locks: the phase differences stop changing, so the columns of X become (almost) multiples of one another and the matrix is close to singular. Restarting the network from random phases twenty times, and keeping only the short stretch where it falls into step, fixes this.

| | condition number of X |
|---|---|
| one locked run | 3.16 × 10<sup>13</sup> |
| twenty restarts (median over oscillators) | 8.2 |

With restarts, least squares recovered every coupling, with a largest error of **0.0028**.

**2. Sparse regression finds the real links.** With measurement noise (0.05 rad) added, plain least squares gives every non-link a small non-zero value. Sequentially thresholded least squares, the algorithm behind SINDy (Brunton et al., 2016), repeatedly sets coefficients below 0.5 to zero and refits.

| | real links kept | false links |
|---|---|---|
| sparse regression | 18 of 18 | 0 |
| plain least squares | 18 of 18 | 38 non-links left non-zero |

The exact figures depend on the random draws: my first version, written as I learned Python, gave 3.09 × 10<sup>13</sup> against 9.5 and a largest error of 0.0041, with the same conclusions.

![Part A](figures/fig_part_a.png)

*Left: one long run locks within seconds, after which the data carry almost no information about the couplings. Right: couplings recovered from twenty restarts against the true values.*

## Part B: brain rhythms (`notebooks/part_b_brain_rhythms.ipynb`)

Open data from the PhysioNet EEG Motor Movement/Imagery Dataset: 109 volunteers, 64 electrodes, one minute resting with eyes open and one with eyes closed. The analysis uses the alpha rhythm (8 to 12 Hz) at eight electrodes over the back of the head, where alpha is strongest.

1. **Phases.** Band-pass filter to 8 to 12 Hz, then take the Hilbert transform. Each electrode's signal becomes a point circling the origin of the complex plane, and its angle is the phase.
2. **The old problem returns.** Neighbouring electrodes pick up the same brain sources at the same instant (volume conduction), so their phases move together whether or not anything is coupled. The phase-locking value counts this zero-delay agreement; imaginary coherency does not.
3. **Nolte's check.** Mixing without delay gives a real cross-spectrum, so its imaginary part is zero. The notebook confirms this on synthetic signals before trusting it on real ones.
4. **Eyes open against eyes closed.** Phase-locking value and |imaginary coherency| for all 28 electrode pairs in 2-second windows, compared within each volunteer with a Wilcoxon signed-rank test.
5. **A classifier tested on strangers.** Logistic regression on the 28 coherency values, with five-fold cross-validation split by volunteer (`GroupKFold`), so nobody is in both the training and the test set. A baseline using alpha power alone shows whether synchrony adds anything beyond the rhythm's strength.

### Results

<!-- Fill from results.json after running the notebook -->

| | |
|---|---|
| volunteers analysed | RESULT |
| eyes open, rear pairs: phase-locking value vs \|imaginary coherency\| | RESULT |
| \|imaginary coherency\| higher with eyes closed | RESULT of RESULT volunteers (Wilcoxon p = RESULT) |
| classifier accuracy on unseen volunteers, coherency | RESULT % |
| classifier accuracy, alpha power alone | RESULT % |
| classifier accuracy, both | RESULT % |

![Eyes open against eyes closed](figures/fig_eyes_closed.png)
![Classifier accuracy](figures/fig_accuracy.png)

## Next: larger networks

For large networks, Restrepo, Ott and Hunt (2005) showed that the coupling strength at which synchrony first appears is set by the largest eigenvalue of the network's adjacency matrix. The next step is to test how well the backwards method works on bigger networks either side of that threshold.

## Running it

```bash
pip install -r requirements.txt
python part_a.py          # Part A: prints the results above and saves figures/fig_part_a.png
pytest -q                 # the tests GitHub runs on every push
```

Part B is easiest in Google Colab (button at the top; *Runtime > Run all*, about 30 minutes, most of it downloading data).

| file | what it does |
|---|---|
| `synchrony.py` | simulation, the linear system, least squares and sparse regression |
| `part_a.py` | Part A results and figure |
| `notebooks/part_b_brain_rhythms.ipynb` | Part B, the EEG analysis |
| `tests/test_core.py` | checks that couplings are recovered, that locked data are ill-conditioned, and that sparse regression drops non-links |
| `results/results.json` | Part B numbers, written by the notebook |

## References

- Strogatz, S. H. (1988). Love affairs and differential equations. *Mathematics Magazine*, 61(1), 35.
- Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators. *Physica D*, 143, 1–20.
- Brunton, S. L., Proctor, J. L. and Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937.
- Nolte, G. et al. (2004). Identifying true brain interaction from EEG data using the imaginary part of coherency. *Clinical Neurophysiology*, 115(10), 2292–2307.
- Restrepo, J. G., Ott, E. and Hunt, B. R. (2005). Onset of synchronization in large networks of coupled oscillators. *Physical Review E*, 71, 036151.
- Schalk, G. et al. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
- Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

## Acknowledgement

I wrote Part A myself while learning Python, using exercises set by Claude, an AI assistant. The Part B notebook and the tests were written with Claude's help; I ran the analysis, checked each step and can explain every result.

## Licence

MIT. See `LICENSE`.
