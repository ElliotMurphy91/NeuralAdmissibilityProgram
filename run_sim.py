"""Run the NAP composition-law simulation.  Writes results.npz."""
import numpy as np
from scipy.optimize import brentq
from gate_core import (make_lexicon, gate, gate_norm, renyi_S, Laws, NPT, TAX,
                       compose_tree, rotation_pair, order_pair)

RNG = np.random.default_rng(20260715)
BETA0, ALPHA0 = 1.0, 2.0
DEPTHS = np.arange(1, 6)
SNRS = np.array([-15., -10., -5., 0., 5., 10., 15.])
LAWS = ["gate", "gate_norm", "additive", "tensor", "rnn"]
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
    DIV = np.zeros((len(DEPTHS), N_ITEM, 2))     # gate-vs-additive r, per bracketing
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
        if kind == "bracket":
            g1, g2 = M["gate"]; a1, a2 = M["additive"]
            for i in range(N_ITEM):
                DIV[di, i, 0] = np.corrcoef(g1[i], a1[i])[0, 1]
                DIV[di, i, 1] = np.corrcoef(g2[i], a2[i])[0, 1]
    out[f"ACC_{kind}"] = ACC
    if kind == "bracket":
        out["DIV"] = DIV

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
print("\ngate-additive correlation vs depth:",
      " ".join(f"{out['DIV'][di,:,0].mean():.3f}" for di in range(len(DEPTHS))))
print("  same for the OTHER bracketing:",
      " ".join(f"{out['DIV'][di,:,1].mean():.3f}" for di in range(len(DEPTHS))))
