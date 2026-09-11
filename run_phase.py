"""Angular vs containment depth codes.  Writes results_phase.npz."""
import numpy as np
from phase_core import (patterns, angular_trial, angular_decode, n_addresses,
                        containment_trial, containment_decode, pac_matrix,
                        DEPTHS, F_DTH, TAX, NT, FS)

RNG = np.random.default_rng(20260715)
W = patterns(np.random.default_rng(11))
SNRS = np.array([-20., -15., -12., -9., -6., -3., 0., 5.])
NTRI = 150
TAU = 0.01
out = {}


def confusion(fn, snr, n, rng):
    C = np.zeros((6, 6))
    for di, d in enumerate(DEPTHS):
        for _ in range(n):
            C[di, fn(int(d), snr, rng) - 1] += 1
    return C / n


# --------------------------------------------------- angular, two carrier ratios
for tag, fb in [("a4", 16.0), ("a7", 28.0)]:
    Cs = np.zeros((len(SNRS), 6, 6))
    for si, s in enumerate(SNRS):
        rng = np.random.default_rng(100 + si)
        Cs[si] = confusion(
            lambda d, sn, r, fb=fb: angular_decode(angular_trial(d, fb, sn, W, r), fb, W, r),
            s, NTRI, rng)
    out[f"C_{tag}"] = Cs
    print(tag, "d*=", n_addresses(fb), "acc@0dB",
          np.round(np.diag(Cs[list(SNRS).index(0.0)]), 2))

# --------------------------------------------------------------- containment
Cs = np.zeros((len(SNRS), 6, 6))
for si, s in enumerate(SNRS):
    rng = np.random.default_rng(200 + si)
    Cs[si] = confusion(
        lambda d, sn, r: containment_decode(pac_matrix(containment_trial(d, sn, W, r), W), TAU, r),
        s, NTRI, rng)
out["C_cont"] = Cs
print("cont acc@0dB", np.round(np.diag(Cs[list(SNRS).index(0.0)]), 2))

# ------------------------------- containment at a COMMON carrier (unidentifiable)
rng = np.random.default_rng(300)
C = confusion(
    lambda d, sn, r: containment_decode(
        pac_matrix(containment_trial(d, sn, W, r, common_freq=True), W, common_freq=True), TAU, r),
    5.0, 100, rng)
out["C_cont_common"] = C
print("cont common-freq acc @ +5 dB:", np.round(np.diag(C), 2), "mean", round(np.diag(C).mean(), 3))

# ------------------------------------------------- carrier-ratio sweep (design rule)
RATIOS = np.arange(2, 10)
racc = np.zeros(len(RATIOS))
for ri, r in enumerate(RATIOS):
    rng = np.random.default_rng(400 + ri)
    C = confusion(
        lambda d, sn, rr, fb=float(r) * F_DTH: angular_decode(
            angular_trial(d, fb, sn, W, rr), fb, W, rr), 5.0, 100, rng)
    racc[ri] = np.diag(C).mean()
out["ratios"], out["ratio_acc"] = RATIOS, racc
print("ratio sweep:", dict(zip(RATIOS.tolist(), np.round(racc, 2).tolist())))

# -------------------------------------------------------------- demo waveforms
rng = np.random.default_rng(7)
X, dth, beta, envs, dphi = angular_trial(1, 16.0, 5.0, W, rng, return_parts=True)
out["demo_dth"], out["demo_beta"] = dth, beta
for k in range(7):
    out[f"demo_env{k}"] = envs[k]
out["demo_dphi"] = dphi
out["SNRS"], out["NTRI"], out["TAU"] = SNRS, NTRI, TAU
np.savez("results_phase.npz", **out)


# ------------------------------------------------------------ aliasing index
def alias_index(C, dstar):
    i, j = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")
    lag = np.abs(i - j)
    L = C[lag == dstar].mean()
    O = C[(lag != dstar) & (lag != 0)].mean()
    return L - O


for tag, ds in [("a4", 4), ("a7", 4), ("cont", 4)]:
    A = [alias_index(out[f"C_{tag}"][si], ds) for si in range(len(SNRS))]
    print(f"alias index {tag}:", np.round(A, 3))
print("detection thresholds  N=20/50/100/200:",
      np.round([0.5264 / np.sqrt(n) for n in [20, 50, 100, 200]], 3))
