# Simulations for Murphy (2026)

Code to reproduce the two simulations

> Murphy, E. (2026). Formal properties of language as constraints on neural
dynamics. arXiv.

Both simulations are self-contained. Stimuli are generated from seeded random 
number generators.

**Simulation A — composition laws.** How non-associative is the entropy-optimized
Rényi gate, as a function of inverse temperature β and Rényi order α; and can a
decoder recover syntactic bracketing from a composed root representation better
under the gate than under its rivals?

**Simulation B — phase codes.** Does embedding depth show up in cortical dynamics
as an angular phase address or as a containment topology over the cross-frequency
coupling graph, and what distinguishes them experimentally?

---

## Quick start

```bash
git clone https://github.com/ElliotMurphy91/NeuralAdmissibilityProgram.git
cd NeuralAdmissibilityProgram
pip install -r requirements.txt

python run_all.py        # both simulations + all three figures
python test_sims.py      # 21 analytic self-checks, a few seconds
```

Or one at a time:

```bash
python run_all.py --sim a     # composition laws  -> Figures S1, S2
python run_all.py --sim b     # phase codes       -> Figure S3
python run_all.py --figs      # build figures from existing .npz
```

## Contents

| File | Role |
|---|---|
| `gate_core.py` | Sim A. Gabor atoms, the Rényi gate, the offset-corrected gate, the rival laws, tree builders. |
| `run_sim.py` | Sim A driver → `results.npz` |
| `make_figs.py` | Sim A figures → `Figure_S1`, `Figure_S2` |
| `phase_core.py` | Sim B. Signal synthesis, 1/f noise, both decoders. |
| `run_phase.py` | Sim B driver → `results_phase.npz` |
| `make_fig3.py` | Sim B figure → `Figure_S3` |
| `run_all.py` | Runs the above in order |
| `test_sims.py` | Self-checks on every analytic property the argument relies on |

## Simulation A — composition laws

Lexical atoms are Gabor wavelets on a 128-point compact domain, zero-mean and
unit-variance, with carrier frequencies drawn uniformly from 6–26 cycles. The
gate

```
f ⊕_{S_α,β} g  =  min_{λ∈[0,1]} { λf + (1−λ)g − S_α(λ)/β }
```

is evaluated pointwise by grid minimization over 1,201 knots. Five laws are
compared:

| Name | Form | Commutative | Associative |
|---|---|---|---|
| `gate` | `min_λ {λu + (1−λ)v − S_α(λ)/β}` | yes | **no** |
| `gate_norm` | `λ*u + (1−λ*)v`, offset-corrected | yes | **no** |
| `additive` | `u + v` | yes | yes |
| `tensor` | `R_L u + R_R v`, orthonormal roles scaled 1/√2 | no | no |
| `rnn` | `tanh(W_u u + W_v v)` | no | no |

Bracketing contrasts are height-matched three-terminal rotations, `((A B) C)`
versus `(A (B C))`, embedded *d*−1 levels below the root on a right-branching
spine (*d* = 1…5). The two conditions share height, node count, terminal
identity, terminal order and leaf-depth multiset, and differ only in the
bracketing of the deepest triple. Order contrasts use the same spine with
`(A B)` versus `(B A)` at the deepest pair. Sixty independent lexicons per cell;
nearest-centroid decoding with 120 training and 120 test trials per condition.

`test_sims.py` verifies the shape matching directly rather than taking it on
trust, along with the tropical limit, the Shannon point, the β→0 plateau
|κ|·‖h−f‖ with κ = 1 − (3/2)λ\*, and the fact that the offset-corrected gate is
*not* associative at α → 1 — the caveat that the two α claims in the manuscript
hold of different gates.

## Simulation B — phase codes

Sixteen channels synthesized at 500 Hz over 4 s trials. Seven contents occupy
seven orthonormal spatial patterns; every trial contains all seven and only the
assignment of content to chain position varies, so terminals are matched across
depths. The probe content is fixed and its depth *d* ∈ {1…6} decoded; chance is
1/6.

Under the **angular** model a δ–θ control field at 4 Hz carries a nested β
carrier, and the constituent at depth *d* emits an 80 Hz γ burst at the β trough
falling at phase Φ(*d*) = *d*·Δφ. Because the troughs *are* the addresses, their
number is exactly the carrier ratio f_β/f_δθ — Δφ is an identity.

Under the **containment** model each content carries an intrinsic low-frequency
carrier at a distinct frequency (2.5–8.5 Hz in 1 Hz steps), and a node's γ
amplitude is modulated by the carriers of all its ancestors with weight decaying
geometrically in distance. Depth is read from the transitive reduction of the
thresholded, direction-oriented Tort modulation-index matrix.

Both are covered by self-checks:

- **Noise is calibrated band by band**, not broadband. 1/f noise carries almost
  no power at 80 Hz, so a nominal broadband −20 dB leaves a γ read-out
  essentially clean and a broadband power calculation is wrong by tens of dB.
- **The common-carrier control is unidentifiable by algebra not by noise.** A
  sum of same-frequency sinusoids is a single sinusoid, so the ancestor set is
  not encoded at all. `test_sims.py` checks this identity to machine precision.

## Expected output

Sim A prints, among other things:

```
kappa = 0.115879  lam* = 0.589414
assoc monotone in beta? increases = 0 / 39
low-beta assoc corr with ||h-f|| = 1.0  with ||g-f|| = 0.036
alpha-1 log-log slope = 0.994

BRACKETING accuracy @ 0 dB (rows=laws, cols=depth 1..5)
  gate       0.633 0.596 0.580 0.558 0.547
  gate_norm  0.819 0.745 0.720 0.678 0.657
  additive   0.506 0.496 0.500 0.499 0.497
  tensor     1.000 1.000 1.000 0.998 0.972
  rnn        1.000 1.000 0.996 0.943 0.799
```

Sim B prints:

```
a4 d*= 4 acc@0dB [0.43 0.45 1.   1.   0.49 0.51]
a7 d*= 7 acc@0dB [1. 1. 1. 1. 1. 1.]
cont acc@0dB [1.   0.97 0.87 0.69 0.51 0.45]
cont common-freq acc @ +5 dB: mean 0.162
ratio sweep: {2: 0.32, 3: 0.51, 4: 0.67, 5: 0.87, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
```

### Figure map

| Script output | Manuscript figure |
|---|---|
| `Figure_S1` | [Figure N — fill in] |
| `Figure_S2` | [Figure N — fill in] |
| `Figure_S3` | [Figure N — fill in] |

## Reproducibility

All fixed structure derives from `np.random.default_rng(20260715)` (lexicons,
role operators, recurrent weights) and `default_rng(11)` (spatial patterns);
per-cell noise and stimulus draws are seeded per cell, so any single cell
reproduces on its own. Changing the seeds changes the realized lexicon or
pattern set but not the qualitative results.

Produced with Python 3.11, NumPy 2.x, SciPy 1.x, scikit-learn 1.x, Matplotlib
3.x on Linux. `default_rng` (PCG64) is stable across platforms and versions, so
the printed numbers should be bit-identical elsewhere.

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
```

Python ≥ 3.9.

## Citation

If you use the gate implementation itself, please also cite Marcolli & Berwick 
(2026), which introduces the entropy-optimized gate on function spaces.
