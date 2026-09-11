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
python test_sims.py      # 30 analytic self-checks, a few seconds
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
| `gate_core.py` | Sim A. Gabor atoms, the Rényi gate, the offset-corrected gate, the rival laws (incl. fixed mean-pooling), tree builders, the balanced four-leaf contrast. |
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

is evaluated pointwise by grid minimization over 1,201 knots. Nine laws are
compared:

| Name | Form | Commutative | Associative | Mixture |
|---|---|---|---|---|
| `gate` | `min_λ {λu + (1−λ)v − S_α(λ)/β}` | yes | **no** | content-dependent |
| `gate_norm` | `λ*u + (1−λ*)v`, offset-corrected | yes | **no** | content-dependent |
| `mean_pool` | `½(u + v)` — `gate_norm` with λ frozen at ½ | yes | **no** | fixed |
| `additive` | `u + v` | yes | yes | — |
| `tensor` | `R_L u + R_R v`, orthonormal roles scaled 1/√2 | no | no | — |
| `rnn` | `tanh(W_u u + W_v v)` | no | no | — |
| `sat_mean` | `tanh(g(u+v)/2)/tanh(g)`, g = 2 | yes | **no** | saturating, sum-only |
| `sat_tied` | `tanh(W(u+v)/2)`, W orthogonal — `rnn` with W_u = W_v | yes | **no** | saturating, sum-only |
| `meld` | **Meld**: `tanh(κ W [λ*u + (1−λ*)v])`, κ = 2 — the corrected gate's soft-min mixture, tied orthogonal weights, saturating transfer | yes | **no** | content-dependent, saturating |

`mean_pool` is the control for what the entropy term adds. It is commutative,
non-associative and order-blind, so it shares the gate's two-dimensional
signature (bracketing decodable, daughter order at chance) and in fact decodes
the three-terminal rotation better than the corrected gate at β = 1. What it
lacks is a content-dependent mixture, and the contrast that exposes this is the
**balanced four-leaf pair** `((A B)(C D))` versus `((A C)(B D))`: same terminals,
same leaf-depth multiset, same content-to-depth map. Any content-independent
mixture maps both to the same function (under `mean_pool` both are
`(A+B+C+D)/4`), so the pair is indistinguishable at every SNR; the corrected
gate separates it at intermediate β only, since λ* → ½ as β → 0 restores
mean-pooling and the tropical limit is associative. This is also the pair on
which the corrected gate's faithfulness demonstrably fails at β → 0. It is not
a matched-string contrast — `((A C)(B D))` has no linearisation as `A B C D` —
so on neural data the discriminator is the leaf-mixture readout (manuscript
§9.2), which is shape-fixed under `mean_pool` and content-dependent under the
gate.

`sat_mean` and `sat_tied` are the commutative members of the recurrent family:
a rate unit into which both daughters enter through the same synapses. They lie
outside the pointwise translation-equivariant class `g(x,y) = (x+y)/2 + h(|x−y|)`
to which both entropy gates belong, and this matters. In that class, boundedness
under nesting forces idempotence (`h(0) = 0`), and all members agree to second
order in signal amplitude on trees with the same content-to-depth map, so the
four-leaf pair is separated only at fourth order (`test_sims.py` checks the ε⁴
scaling). A saturating law escapes this: the cubic term of its transfer recovers
pairing, and it decodes the four-leaf pair at 0.97 against the corrected gate's
0.77, while remaining order-blind. What discriminates a sum-only law from the
gate is sensitivity to daughter *disagreement* at fixed sum — `phi(u+v)` cannot
see `|u−v|`, the soft-min gate can — and `test_sims.py` checks that too.

**Meld** (the proposed neural binding operation; Merge is the formal operation, Meld a candidate realization of its binding factor) combines the two escape routes. Its mixture is the
corrected gate's pointwise soft minimum (so it penalises daughter disagreement
and is content-dependent), both daughters then pass through the same orthogonal
weights (commutativity) and a saturating transfer (bounded, not idempotent,
pairing recovered at third order). It reduces to `sat_tied` as β → 0 and to a
saturated tied minimum as β → ∞, and unlike the corrected gate it has no
temperature window: bracketing on the rotation decodes at 1.000 and the
four-leaf pair at ≥ 0.99 for every β from 0.03 to 30. On the depth ladder it
decodes 1.000/1.000/0.998/0.970/0.922, order at chance — within reach of the
ordered role-filler law (0.972 at d = 5) while admissible.

The β trade-off (bracketing decodability of the *d* = 1 rotation against
inverse temperature) is run on the corrected gate, which is the gate the
manuscript recommends; the unnormalised gate is swept alongside for reference
only, since its accumulated entropy offset separates the two bracketings on its
own and diverges as β → 0, confounding its sweep at every temperature. The
sweep's decode seeds are those of the *d* = 1 ladder cell, so its β = 1 row
reproduces the ladder value exactly.

`run_sim.py` also reports d′ = ‖m₁ − m₂‖/σ for every cell. Each law attenuates
the deepest triple differently on its way to the root (≈λ* per level for the
gates and `mean_pool`, 0.707 for `tensor`, a saturating nonlinearity for `rnn`)
while noise is scaled to the whole response, so the cross-law ordering at depth
is partly a statement about per-level gain; d′ makes that explicit.

Bracketing contrasts are height-matched three-terminal rotations, `((A B) C)`
versus `(A (B C))`, embedded *d*−1 levels below the root on a right-branching
spine (*d* = 1…5). The two conditions share height, node count, terminal
identity, terminal order and leaf-depth multiset, and differ only in the
bracketing of the deepest triple. Order contrasts use the same spine with
`(A B)` versus `(B A)` at the deepest pair. Sixty independent lexicons per cell;
nearest-centroid decoding with 120 training and 120 test trials per condition.

`test_sims.py` verifies the shape matching directly rather than taking it on
trust, along with the tropical limit, the Shannon point, the β→0 plateau
|κ|·‖h−f‖ with κ = 1 − (3/2)λ\*, the fact that the offset-corrected gate is
*not* associative at α → 1 — the caveat that the two α claims in the manuscript
hold of different gates — and the corrected gate's known failures of
faithfulness: it is idempotent (`G(u,u) = u`, so self-merge collapses; the
unnormalised gate is not, precisely because `x ⊕ x = x − log2/β`), it collapses
the four-leaf pair at β → 0 and β → ∞, and `mean_pool` collapses that pair at
every β.

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
  mean_pool  0.997 0.928 0.736 0.581 0.525
  sat_mean   0.985 0.931 0.828 0.719 0.595
  sat_tied   1.000 0.998 0.976 0.832 0.654
  meld       1.000 1.000 0.998 0.970 0.922
ORDER accuracy @ 0 dB
  gate       0.499 0.506 0.498 0.502 0.494
  gate_norm  0.505 0.500 0.499 0.495 0.503
  additive   0.506 0.496 0.500 0.499 0.497
  tensor     1.000 1.000 0.997 0.967 0.891
  rnn        1.000 0.997 0.945 0.834 0.701
  mean_pool  0.499 0.502 0.498 0.500 0.499
  sat_mean   0.502 0.502 0.497 0.495 0.502
  sat_tied   0.507 0.497 0.497 0.507 0.506
  meld       0.495 0.498 0.500 0.502 0.499
d' @ 0 dB, bracketing (||m1-m2||/sigma)
  gate         1.11   0.89   0.80   0.69   0.60
  gate_norm    2.22   1.77   1.60   1.41   1.19
  additive     0.00   0.00   0.00   0.00   0.00
  tensor      16.30  11.66   8.18   5.88   4.15
  rnn         15.93   9.47   5.61   3.50   2.11
  mean_pool    6.41   3.34   1.70   0.85   0.43
  sat_mean     6.47   3.94   2.54   1.62   0.87
  sat_tied    14.78   7.72   4.29   2.26   1.22
  meld        14.41   9.08   6.36   4.29   3.33

FOUR-LEAF ((AB)(CD)) vs ((AC)(BD)) @ 0 dB:  acc / d' / noise-free rel. separation
  gate       0.570 /   0.75 / 0.0223
  gate_norm  0.767 /   1.87 / 0.1358
  additive   0.492 /   0.00 / 0.0000
  tensor     1.000 /  11.42 / 1.0099
  rnn        1.000 /  12.06 / 1.0690
  mean_pool  0.497 /   0.00 / 0.0000
  sat_mean   0.971 /   4.19 / 0.3734
  sat_tied   0.966 /   3.96 / 0.3531
  meld       1.000 /   9.11 / 0.8026
four-leaf, corrected gate vs beta @ 0 dB:
  beta= 0.03: acc 0.510   rel. sep 0.0010
  beta=  0.1: acc 0.503   rel. sep 0.0012
  beta=  0.3: acc 0.518   rel. sep 0.0250
  beta=  0.5: acc 0.827   rel. sep 0.1996
  beta=  1.0: acc 0.753   rel. sep 0.1358
  beta=  3.0: acc 0.525   rel. sep 0.0292
  beta= 10.0: acc 0.500   rel. sep 0.0056
  beta= 30.0: acc 0.498   rel. sep 0.0011

BETA SWEEP, d=1 rotation @ 0 dB (corrected gate | unnormalised gate | Meld) ; Meld four-leaf
  beta= 0.03: 0.997 | 0.919 | 1.000 ; 0.990
  beta=  0.1: 0.997 | 0.917 | 1.000 ; 0.990
  beta=  0.3: 0.997 | 0.900 | 1.000 ; 0.994
  beta=  0.5: 0.991 | 0.842 | 1.000 ; 0.998
  beta=  1.0: 0.819 | 0.633 | 1.000 ; 1.000
  beta=  3.0: 0.532 | 0.504 | 1.000 ; 1.000
  beta= 10.0: 0.505 | 0.498 | 1.000 ; 1.000
  beta= 30.0: 0.505 | 0.499 | 1.000 ; 1.000
  significance threshold 0.553
```

(The four-leaf accuracies at β = 1 differ between the law table, 0.767, and the
β sweep, 0.753, only through the per-cell decode seed; the noise-free
separation, 0.1358, is identical.)

Sim B prints:

```
a4 d*= 4 acc@0dB [0.43 0.45 1.   1.   0.49 0.51]
a7 d*= 7 acc@0dB [1. 1. 1. 1. 1. 1.]
cont acc@0dB [1.   0.97 0.87 0.69 0.51 0.45]
cont common-freq acc @ +5 dB: mean 0.162
ratio sweep: {2: 0.32, 3: 0.51, 4: 0.67, 5: 0.87, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
```

### Known code / manuscript discrepancies

- The per-item significance threshold is 0.553 here (0.5 + 1.645·√(0.25/240));
  the manuscript quotes 0.558.
- The unnormalised gate's *d* = 1 bracketing accuracy at β = 0.1 is 0.917 here;
  the manuscript quotes 0.912. (The manuscript's β sweep should in any case be
  read off the corrected-gate column above, per the revision.)

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
