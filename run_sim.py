"""Run the NAP composition-law simulation.  Writes results.npz."""
import numpy as np
from scipy.optimize import brentq
from gate_core import (make_lexicon, gate, gate_norm, renyi_S, Laws, NPT, TAX,
                       compose_tree, rotation_pair, order_pair, balanced_pair)

RNG = np.random.default_rng(20260715)
BETA0, ALPHA0 = 1.0, 2.0
DEPTHS = np.arange(1, 6)
SNRS = np.array([-15., -10., -5., 0., 5., 10., 15.])
LAWS = ["gate", "gate_norm", "additive", "tensor", "rnn", "mean_pool", "sat_mean", "sat_tied", "meld"]
N_ITEM = 60
N_TRIAL = 240           # per condition per item
out = {}

# ============================================================ E1  gate demo
rng = np.random.default_rng(3)
L = make_lexicon(2, rng)
out["demo_f"], out["demo_g"] = L[0], L[1]
for i, b in enumerate([0.3, 1.0, 4.0]):
    v, lam = gate(L[0:1], L[1:2], b, return_lambda=True)
    out[f"demo_v{i}"], out[f"demo_lam{i}"] = v[0], lam[0]

# ============================================================ E2  associator
N_TRIP = 400
trips = [make_lexicon(3, RNG) for _ in range(N_TRIP)]
F = np.stack([t[0] for t in trips]); G = np.stack([t[1] for t in trips])
H = np.stack([t[2] for t in trips])
hf = np.linalg.norm(H - F, axis=1)


def assoc(F, G, H, beta, alpha, fn=gate):
    a = fn(fn(F, G, beta, alpha), H, beta, alpha)
    b = fn(F, fn(G, H, beta, alpha), beta, alpha)
    return np.linalg.norm(a - b, axis=1)


BETAS = np.logspace(-3, 2.3, 40)
A_beta = np.array([assoc(F, G, H, b, ALPHA0) for b in BETAS])          # (nb, ntrip)
out["betas"], out["A_beta"] = BETAS, A_beta

S2p = lambda l: -(4 * l - 2) / (2 * l ** 2 - 2 * l + 1)
lam_fp = brentq(lambda l: S2p(l) + np.log(2), 0.5, 0.99)
kappa = 1 - 1.5 * lam_fp
out["kappa"], out["lam_fp"] = kappa, lam_fp
out["plateau"] = abs(kappa) * hf                                        # per triple
# does the associator track ||h-f|| and ignore g?  (low-beta regime)
out["corr_hf"] = np.corrcoef(A_beta[2], hf)[0, 1]
out["corr_gf"] = np.corrcoef(A_beta[2], np.linalg.norm(G - F, axis=1))[0, 1]

ALPHAS = np.concatenate([1 + np.logspace(-4, np.log10(2.0), 34)])
A_alpha = np.array([assoc(F[:120], G[:120], H[:120], BETA0, a) for a in ALPHAS])
out["alphas"], out["A_alpha"] = ALPHAS, A_alpha

# ============================================================ E3  depth ladder
laws_obj = Laws(RNG, beta=BETA0, alpha=ALPHA0)


def build(kind, depth, n_item, rng):
    """Return (n_item, NPT) condition means for the two matched conditions."""
    pair = rotation_pair if kind == "bracket" else order_pair
    t1, t2, nleaf = pair(depth)
    M = {}
    lex = [make_lexicon(nleaf, rng) for _ in range(n_item)]
    for name in LAWS:
        Gf = laws_obj.get(name)
        m1 = np.stack([compose_tree(t1, lx, Gf) for lx in lex])
        m2 = np.stack([compose_tree(t2, lx, Gf) for lx in lex])
        M[name] = (m1, m2)
    return M


def decode(m1, m2, sigma, n_trial, rng):
    """Nearest-centroid decoder, train/test split, isotropic additive noise."""
    h = n_trial // 2
    acc = np.empty(m1.shape[0])
    for i in range(m1.shape[0]):
        X1 = m1[i] + sigma * rng.standard_normal((n_trial, NPT))
        X2 = m2[i] + sigma * rng.standard_normal((n_trial, NPT))
        w = X1[:h].mean(0) - X2[:h].mean(0)
        c = 0.5 * (X1[:h].mean(0) + X2[:h].mean(0))
        s1 = (X1[h:] - c) @ w
        s2 = (X2[h:] - c) @ w
        acc[i] = 0.5 * ((s1 > 0).mean() + (s2 < 0).mean())
    return acc


for kind in ["bracket", "order"]:
    ACC = np.zeros((len(LAWS), len(DEPTHS), len(SNRS), N_ITEM))
    DPR = np.zeros((len(LAWS), len(DEPTHS), len(SNRS), N_ITEM))   # d' = ||m1-m2|| / sigma
    DIV = np.zeros((len(DEPTHS), N_ITEM, 2))     # gate-vs-additive r, per bracketing
    DIVM = np.zeros((len(DEPTHS), N_ITEM, 2))    # meld-vs-additive r, per bracketing
    for di, d in enumerate(DEPTHS):
        rng = np.random.default_rng(1000 + d)
        M = build(kind, int(d), N_ITEM, rng)
        for li, name in enumerate(LAWS):
            m1, m2 = M[name]
            # signal power: variance of the evoked response over items x conds x domain,
            # grand mean removed (a constant offset is not an observable)
            allm = np.concatenate([m1, m2], 0)
            allm = allm - allm.mean()
            psig = allm.var()
            for si, snr in enumerate(SNRS):
                sigma = np.sqrt(psig / 10 ** (snr / 10.0))
                ACC[li, di, si] = decode(m1, m2, sigma,
                                         N_TRIAL, np.random.default_rng(7 * di + si + 31 * li))
                DPR[li, di, si] = np.linalg.norm(m1 - m2, axis=1) / sigma
        if kind == "bracket":
            g1, g2 = M["gate"]; a1, a2 = M["additive"]; e1, e2 = M["meld"]
            for i in range(N_ITEM):
                DIV[di, i, 0] = np.corrcoef(g1[i], a1[i])[0, 1]
                DIV[di, i, 1] = np.corrcoef(g2[i], a2[i])[0, 1]
                DIVM[di, i, 0] = np.corrcoef(e1[i], a1[i])[0, 1]
                DIVM[di, i, 1] = np.corrcoef(e2[i], a2[i])[0, 1]
    out[f"ACC_{kind}"] = ACC
    out[f"DPRIME_{kind}"] = DPR
    if kind == "bracket":
        out["DIV"] = DIV
        out["DIV_meld"] = DIVM

# ============================================================ E4  balanced four-leaf pair
# ((A B)(C D)) vs ((A C)(B D)): same terminals, same leaf-depth multiset, same
# content-to-depth map.  A content-independent mixture cannot separate them.
t4a, t4b, n4 = balanced_pair()
rng4 = np.random.default_rng(4000)
lex4 = [make_lexicon(n4, rng4) for _ in range(N_ITEM)]


def four_leaf(Gf):
    m1 = np.stack([compose_tree(t4a, lx, Gf) for lx in lex4])
    m2 = np.stack([compose_tree(t4b, lx, Gf) for lx in lex4])
    return m1, m2


def acc_over_snr(m1, m2, seed0):
    allm = np.concatenate([m1, m2], 0); allm = allm - allm.mean()
    psig = allm.var()
    A = np.zeros((len(SNRS), N_ITEM)); P = np.zeros((len(SNRS), N_ITEM))
    for si, snr in enumerate(SNRS):
        sigma = np.sqrt(psig / 10 ** (snr / 10.0))
        A[si] = decode(m1, m2, sigma, N_TRIAL, np.random.default_rng(seed0 + si))
        P[si] = np.linalg.norm(m1 - m2, axis=1) / sigma
    return A, P


ACC4 = np.zeros((len(LAWS), len(SNRS), N_ITEM)); DPR4 = np.zeros_like(ACC4)
SEP4 = np.zeros((len(LAWS), N_ITEM))          # noise-free relative separation
for li, name in enumerate(LAWS):
    m1, m2 = four_leaf(laws_obj.get(name))
    ACC4[li], DPR4[li] = acc_over_snr(m1, m2, 5000 + 31 * li)
    SEP4[li] = np.linalg.norm(m1 - m2, axis=1) / np.linalg.norm(m1, axis=1)
out["ACC_fourleaf"], out["DPRIME_fourleaf"], out["SEP_fourleaf"] = ACC4, DPR4, SEP4

# the corrected gate on the four-leaf pair as a function of temperature:
# lam* -> 1/2 as beta -> 0 (mean-pooling, pair collapses); tropical limit is
# associative (pair collapses); separation lives at intermediate beta only.
BETAS_SWEEP = np.array([0.03, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0])
ACC4_beta = np.zeros((len(BETAS_SWEEP), len(SNRS), N_ITEM))
SEP4_beta = np.zeros((len(BETAS_SWEEP), N_ITEM))
for bi, b in enumerate(BETAS_SWEEP):
    m1, m2 = four_leaf(lambda u, v, b=b: gate_norm(u, v, b, ALPHA0))
    ACC4_beta[bi], _ = acc_over_snr(m1, m2, 6000 + 7 * bi)
    SEP4_beta[bi] = np.linalg.norm(m1 - m2, axis=1) / np.linalg.norm(m1, axis=1)
out["BETAS_SWEEP"], out["ACC_fourleaf_beta"], out["SEP_fourleaf_beta"] = BETAS_SWEEP, ACC4_beta, SEP4_beta

# ============================================================ E5  beta trade-off, corrected gate
# Bracketing decodability of the d = 1 rotation as a function of beta, for the
# offset-corrected gate (the gate the manuscript recommends) and, for reference,
# the unnormalised gate, whose sweep is confounded by the divergent offset.
rng5 = np.random.default_rng(1001)             # same seed as the d = 1 ladder cell
t1, t2, n1 = rotation_pair(1)
lex5 = [make_lexicon(n1, rng5) for _ in range(N_ITEM)]
ACCb = {"gate_norm": np.zeros((len(BETAS_SWEEP), len(SNRS), N_ITEM)),
        "gate": np.zeros((len(BETAS_SWEEP), len(SNRS), N_ITEM)),
        "meld": np.zeros((len(BETAS_SWEEP), len(SNRS), N_ITEM))}
ACC4m = np.zeros((len(BETAS_SWEEP), len(SNRS), N_ITEM))     # Meld, four-leaf pair vs beta
def meld_at(b):
    return lambda u, v: np.tanh(laws_obj.meld_gain * gate_norm(u, v, b, ALPHA0) @ laws_obj.Wt.T)
for bi, b in enumerate(BETAS_SWEEP):
    m1, m2 = four_leaf(meld_at(b))
    ACC4m[bi], _ = acc_over_snr(m1, m2, 31 * LAWS.index("meld"))
    for name, fn in (("gate_norm", gate_norm), ("gate", gate), ("meld", None)):
        Gf = (lambda u, v, b=b, fn=fn: fn(u, v, b, ALPHA0)) if fn is not None else meld_at(b)
        m1 = np.stack([compose_tree(t1, lx, Gf) for lx in lex5])
        m2 = np.stack([compose_tree(t2, lx, Gf) for lx in lex5])
        # decode seeds identical to the d = 1 ladder cell, so the beta = 1 row
        # reproduces the ladder's d = 1 value exactly
        ACCb[name][bi], _ = acc_over_snr(m1, m2, 31 * LAWS.index(name))
out["ACC_beta_gate_norm"], out["ACC_beta_gate"] = ACCb["gate_norm"], ACCb["gate"]
out["ACC_beta_meld"], out["ACC_fourleaf_beta_meld"] = ACCb["meld"], ACC4m

out["LAWS"] = np.array(LAWS); out["DEPTHS"] = DEPTHS; out["SNRS"] = SNRS
out["N_ITEM"] = N_ITEM
np.savez("results.npz", **out)

# ------------------------------------------------------------------ report
A = out["ACC_bracket"]; O = out["ACC_order"]
s0 = list(SNRS).index(0.0)
print("kappa =", round(kappa, 6), " lam* =", round(lam_fp, 6))
print("assoc monotone in beta? increases =",
      int((np.diff(A_beta.mean(1)) > 1e-9).sum()), "/", len(BETAS) - 1)
print("low-beta assoc corr with ||h-f|| =", round(out['corr_hf'], 3),
      " with ||g-f|| =", round(out['corr_gf'], 3))
sl = np.polyfit(np.log(ALPHAS[:12] - 1), np.log(A_alpha[:12].mean(1)), 1)[0]
print("alpha-1 log-log slope =", round(sl, 3))
print("\nBRACKETING accuracy @ 0 dB (rows=laws, cols=depth 1..5)")
for li, n in enumerate(LAWS):
    print(f"  {n:10s}", " ".join(f"{A[li,di,s0].mean():.3f}" for di in range(len(DEPTHS))))
print("ORDER accuracy @ 0 dB")
for li, n in enumerate(LAWS):
    print(f"  {n:10s}", " ".join(f"{O[li,di,s0].mean():.3f}" for di in range(len(DEPTHS))))
print("d' @ 0 dB, bracketing (||m1-m2||/sigma)")
for li, n in enumerate(LAWS):
    print(f"  {n:10s}", " ".join(f"{out['DPRIME_bracket'][li,di,s0].mean():6.2f}" for di in range(len(DEPTHS))))
print("\nFOUR-LEAF ((AB)(CD)) vs ((AC)(BD)) @ 0 dB:  acc / d' / noise-free rel. separation")
for li, n in enumerate(LAWS):
    print(f"  {n:10s} {ACC4[li,s0].mean():.3f} / {DPR4[li,s0].mean():6.2f} / {SEP4[li].mean():.4f}")
print("four-leaf, corrected gate vs beta @ 0 dB:")
for bi, b in enumerate(BETAS_SWEEP):
    print(f"  beta={b:5}: acc {ACC4_beta[bi,s0].mean():.3f}   rel. sep {SEP4_beta[bi].mean():.4f}")
print("\nBETA SWEEP, d=1 rotation @ 0 dB (corrected gate | unnormalised gate | Meld) ; Meld four-leaf")
for bi, b in enumerate(BETAS_SWEEP):
    print(f"  beta={b:5}: {ACCb['gate_norm'][bi,s0].mean():.3f} | {ACCb['gate'][bi,s0].mean():.3f} | {ACCb['meld'][bi,s0].mean():.3f} ; {ACC4m[bi,s0].mean():.3f}")
THR = 0.5 + 1.645 * np.sqrt(0.25 / 240)
print("  significance threshold", round(THR, 3))
print("\ngate-additive correlation vs depth:",
      " ".join(f"{out['DIV'][di,:,0].mean():.3f}" for di in range(len(DEPTHS))))
print("  same for the OTHER bracketing:",
      " ".join(f"{out['DIV'][di,:,1].mean():.3f}" for di in range(len(DEPTHS))))
