"""Meld ablations.

Run from inside the NeuralAdmissibilityProgram repository:  python ablate_lin_tied.py

Reproduces, with the repository's own lexicon, weight and decoder seeds (run_sim.py
conventions; new laws take LAWS indices 9 and 10 for their decoder seeds):

  lin_tied : W [ lam* u + (1 - lam*) v ]           Meld minus the saturating transfer, kappa = 1
             (now also law 10 of gate_core.py / run_sim.py; this script predates that)
  meld_noW : tanh( kappa [ lam* u + (1 - lam*) v ] ) Meld with W = I

together with meld, sat_tied, gate_norm and tensor for reference, at 0 dB on the
d = 1..5 rotation ladder (bracketing and daughter order), the balanced four-leaf pair,
and the order (in leaf amplitude) at which each law separates the four-leaf pair.
Nothing in the repository is modified.
"""
import numpy as np
from gate_core import (make_lexicon, gate_norm, Laws, NPT, compose_tree,
                       rotation_pair, order_pair, balanced_pair)

RNG = np.random.default_rng(20260715)
for _ in range(400):                      # run_sim.py draws its 400 associator triples from RNG
    make_lexicon(3, RNG)                  # before constructing Laws; replay that so W is identical
laws = Laws(RNG, beta=1.0, alpha=2.0)
K, W = laws.meld_gain, laws.Wt
LAWS = ["gate", "gate_norm", "additive", "tensor", "rnn", "mean_pool", "sat_mean",
        "sat_tied", "meld", "lin_tied", "meld_noW"]
FNS = {
    "gate_norm": laws.gate_norm,
    "tensor":    laws.tensor,
    "sat_tied":  laws.sat_tied,
    "meld":      laws.meld,
    "lin_tied":  lambda u, v: gate_norm(u, v, 1.0, 2.0) @ W.T,
    "meld_noW":  lambda u, v: np.tanh(K * gate_norm(u, v, 1.0, 2.0)),
}
N_ITEM, N_TRIAL = 60, 240
SNRS = [-15., -10., -5., 0., 5., 10., 15.]
S0 = SNRS.index(0.0)


def decode(m1, m2, sigma, rng):
    h = N_TRIAL // 2
    acc = np.empty(m1.shape[0])
    for i in range(m1.shape[0]):
        X1 = m1[i] + sigma * rng.standard_normal((N_TRIAL, NPT))
        X2 = m2[i] + sigma * rng.standard_normal((N_TRIAL, NPT))
        w = X1[:h].mean(0) - X2[:h].mean(0)
        c = 0.5 * (X1[:h].mean(0) + X2[:h].mean(0))
        acc[i] = 0.5 * (((X1[h:] - c) @ w > 0).mean() + ((X2[h:] - c) @ w < 0).mean())
    return acc


def sigma0(m1, m2):
    allm = np.concatenate([m1, m2], 0)
    allm = allm - allm.mean()
    return np.sqrt(allm.var())          # 0 dB against the response variance


print(f"{'law':10s} {'bracketing d=1..5 (0 dB)':30s} {'order d=1..5':30s} {'4-leaf':7s} {'d4':5s} {'d1':5s}")
for name in ["tensor", "gate_norm", "sat_tied", "meld", "lin_tied", "meld_noW"]:
    li, Gf = LAWS.index(name), FNS[name]
    rows, dp1 = {}, None
    for kind, pair in (("bracket", rotation_pair), ("order", order_pair)):
        out = []
        for di, d in enumerate(range(1, 6)):
            t1, t2, n = pair(d)
            rng = np.random.default_rng(1000 + d)          # one stream per depth, as in run_sim.build
            lex = [make_lexicon(n, rng) for _ in range(N_ITEM)]
            m1 = np.stack([compose_tree(t1, lx, Gf) for lx in lex])
            m2 = np.stack([compose_tree(t2, lx, Gf) for lx in lex])
            s = sigma0(m1, m2)
            out.append(decode(m1, m2, s, np.random.default_rng(7 * di + S0 + 31 * li)).mean())
            if d == 1 and kind == "bracket":
                dp1 = (np.linalg.norm(m1 - m2, axis=1) / s).mean()
        rows[kind] = out
    rng4 = np.random.default_rng(4000)
    lex4 = [make_lexicon(4, rng4) for _ in range(N_ITEM)]
    t4a, t4b, _ = balanced_pair()
    m1 = np.stack([compose_tree(t4a, lx, Gf) for lx in lex4])
    m2 = np.stack([compose_tree(t4b, lx, Gf) for lx in lex4])
    s = sigma0(m1, m2)
    a4 = decode(m1, m2, s, np.random.default_rng(5000 + 31 * li + S0)).mean()
    dp4 = (np.linalg.norm(m1 - m2, axis=1) / s).mean()
    print(f"{name:10s} {' '.join(f'{x:5.3f}' for x in rows['bracket']):30s} "
          f"{' '.join(f'{x:5.3f}' for x in rows['order']):30s} {a4:5.3f}   {dp4:4.1f}  {dp1:4.1f}")

# ---- order of the four-leaf separation in leaf amplitude, and the second-order term of lin_tied
print("\nfour-leaf separation vs leaf amplitude (local log-log slope; 2 = second order, 4 = fourth)")
A, B, C, D = make_lexicon(4, np.random.default_rng(4000))
t1, t2, _ = balanced_pair()
eps = np.array([0.4, 0.2, 0.1])
for name in ["gate_norm", "lin_tied"]:
    Gf = FNS[name]
    sep = np.array([np.linalg.norm(compose_tree(t1, e * np.stack([A, B, C, D]), Gf)
                                   - compose_tree(t2, e * np.stack([A, B, C, D]), Gf)) for e in eps])
    print(f"  {name:10s} slopes {np.round(np.diff(np.log(sep)) / np.diff(np.log(eps)), 2)}")
c = 1.0 / 8                                                   # beta / 8 at beta = 1
pred = c * (W @ (W @ ((A - D) * (B - C)) - (W @ (A - D)) * (W @ (B - C))))
e = 0.05
diff = (compose_tree(t1, e * np.stack([A, B, C, D]), FNS["lin_tied"])
        - compose_tree(t2, e * np.stack([A, B, C, D]), FNS["lin_tied"])) / e ** 2
print(f"  lin_tied second-order term vs c W{{W[(A-D)o(B-C)] - (W(A-D))o(W(B-C))}}: "
      f"r = {np.corrcoef(diff, pred)[0, 1]:.3f}, norm ratio = {np.linalg.norm(diff) / np.linalg.norm(pred):.3f}")
