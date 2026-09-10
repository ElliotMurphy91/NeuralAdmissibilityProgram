"""
test_sims.py -- self-checks on both simulations.

    python test_sims.py          # plain, no pytest needed
    pytest test_sims.py          # also works
"""
import numpy as np
from scipy.optimize import brentq

from gate_core import (Laws, NPT, balanced_pair, compose_tree, gate, gate_norm,
                       make_lexicon, order_pair, renyi_S, rotation_pair)
import phase_core as pc

RNG = np.random.default_rng(0)
LEX = make_lexicon(3, RNG)
F, G, H = LEX[0:1], LEX[1:2], LEX[2:3]


def _assoc(fn, f, g, h, beta, alpha=2.0):
    return np.linalg.norm(fn(fn(f, g, beta, alpha), h, beta, alpha)
                          - fn(f, fn(g, h, beta, alpha), beta, alpha))


# ------------------------------------------------------------ the gate
def test_gate_is_commutative():
    for b in (0.1, 1.0, 10.0):
        assert np.allclose(gate(F, G, b), gate(G, F, b))


def test_gate_is_not_associative():
    """Nonzero associator at finite beta -- the formal residue of constituency."""
    assert _assoc(gate, F, G, H, 1.0) > 1e-3


def test_tropical_limit_is_associative_and_is_the_min():
    assert _assoc(gate, F, G, H, 1e3) < 1e-3
    assert np.abs(gate(F, G, 1e3) - np.minimum(F, G)).max() < 1e-2


def test_shannon_point_is_associative():
    """At alpha -> 1 the gate is a log-sum-exp, hence associative."""
    assert _assoc(gate, F, G, H, 1.0, alpha=1.0) < 1e-4


def test_lambda_star_and_kappa():
    """beta -> 0 associator plateau: |kappa| ||h - f||, kappa = 1 - (3/2) lam*."""
    s2p = lambda l: -(4 * l - 2) / (2 * l ** 2 - 2 * l + 1)
    lam = brentq(lambda l: s2p(l) + np.log(2), 0.5, 0.99)
    kappa = 1 - 1.5 * lam
    assert abs(lam - 0.589414) < 1e-5
    assert abs(kappa - 0.115879) < 1e-5
    lex = make_lexicon(300, np.random.default_rng(5))
    idx = np.stack([np.random.default_rng(6 + i).choice(300, 3, replace=False)
                    for i in range(150)])
    f, g, h = lex[idx[:, 0]], lex[idx[:, 1]], lex[idx[:, 2]]
    got = np.linalg.norm(gate(gate(f, g, 1e-3), h, 1e-3)
                         - gate(f, gate(g, h, 1e-3), 1e-3), axis=1).mean()
    want = (abs(kappa) * np.linalg.norm(h - f, axis=1)).mean()
    assert abs(got - want) / want < 0.01, (got, want)


def test_renyi_entropy_limits():
    """S_alpha(1/2) = log 2 at every order; endpoints go to 0 (up to clipping)."""
    for a in (0.5, 1.0, 2.0, 4.0):
        assert abs(renyi_S(0.5, a) - np.log(2)) < 1e-9
        assert abs(renyi_S(0.0, a)) < 1e-5


# ------------------------------------------------- the offset-corrected gate
def test_gate_norm_is_commutative_and_non_associative():
    for b in (0.1, 1.0, 10.0):
        assert np.allclose(gate_norm(F, G, b), gate_norm(G, F, b))
    assert _assoc(gate_norm, F, G, H, 1.0) > 1e-3


def test_gate_norm_is_not_associative_at_the_shannon_point():
    """The manuscript's key caveat: the two alpha claims hold of DIFFERENT gates."""
    assert _assoc(gate, F, G, H, 1.0, alpha=1.0) < 1e-4        # unnormalised: yes
    assert _assoc(gate_norm, F, G, H, 1.0, alpha=1.0) > 1.0    # corrected: no


def test_gate_norm_beta_zero_associator_is_hf_over_4():
    """With the offset dropped, lam* -> 1/2 is restored and the limit is ||h-f||/4."""
    got = _assoc(gate_norm, F, G, H, 1e-4)
    want = np.linalg.norm(H - F) / 4.0
    assert abs(got - want) / want < 0.02, (got, want)


def test_gate_norm_carries_no_divergent_offset():
    """
    Every application of the unnormalised gate subtracts -S_2(lam*)/beta, so a
    nested tree carries an accumulated constant that DIVERGES as beta -> 0.
    That is the property which disqualifies it as a neural code, and the
    offset-corrected gate is the repair. Lowering beta from 1 to 1/4 should
    therefore move the unnormalised gate's mean by several units and leave the
    corrected gate's essentially where it was.
    """
    lex = make_lexicon(8, np.random.default_rng(4))
    t1, _, n = rotation_pair(5)
    raw = [compose_tree(t1, lex[:n], lambda u, v, b=b: gate(u, v, b)).mean()
           for b in (1.0, 0.25)]
    nrm = [compose_tree(t1, lex[:n], lambda u, v, b=b: gate_norm(u, v, b)).mean()
           for b in (1.0, 0.25)]
    drift_raw, drift_nrm = abs(raw[1] - raw[0]), abs(nrm[1] - nrm[0])
    assert drift_raw > 3.0, drift_raw
    assert drift_nrm < 1.0, drift_nrm
    assert drift_raw / drift_nrm > 5.0, (drift_raw, drift_nrm)


def test_gate_norm_is_idempotent_and_the_unnormalised_gate_is_not():
    """G(u,u) = u for the corrected gate, so Psi(M(A,A)) = Psi(A): self-merge
    collapses.  The unnormalised gate is NOT idempotent, precisely because of
    the offset:  x (+) x = x - log(2)/beta."""
    for b in (0.1, 1.0, 10.0):
        assert np.allclose(gate_norm(F, F, b), F)
        assert np.allclose(gate(F, F, b), F - np.log(2) / b, atol=1e-6)


# ------------------------------------------------------ the four-leaf contrast
def _fourleaf(fn):
    lex = make_lexicon(4, np.random.default_rng(9))
    ta, tb, n = balanced_pair()
    a, b = compose_tree(ta, lex[:n], fn), compose_tree(tb, lex[:n], fn)
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_balanced_pair_matches_content_to_depth_map():
    """Distinct magma elements, identical terminals, identical depth for every leaf."""
    ta, tb, n = balanced_pair()
    assert ta != tb and n == 4

    def leaf_depths(t, d=0):
        return {t: d} if isinstance(t, int) else {**leaf_depths(t[0], d + 1), **leaf_depths(t[1], d + 1)}
    assert leaf_depths(ta) == leaf_depths(tb) == {0: 2, 1: 2, 2: 2, 3: 2}


def test_fixed_pooling_collapses_the_four_leaf_pair():
    """A content-independent mixture cannot separate ((AB)(CD)) from ((AC)(BD))."""
    laws = Laws(np.random.default_rng(2))
    assert _fourleaf(laws.mean_pool) < 1e-12
    assert _fourleaf(laws.additive) < 1e-12


def test_corrected_gate_separates_the_four_leaf_pair_only_at_intermediate_beta():
    """Faithfulness of the corrected gate fails at beta -> 0 (it is mean-pooling
    there) and in the tropical limit (associative); the pair is separated only
    where lambda* is content-dependent."""
    mid = _fourleaf(lambda u, v: gate_norm(u, v, 1.0))
    low = _fourleaf(lambda u, v: gate_norm(u, v, 1e-3))
    high = _fourleaf(lambda u, v: gate_norm(u, v, 1e3))
    assert mid > 0.03, mid
    assert low < 0.01 and high < 0.01, (low, high)
    assert mid > 10 * max(low, high)


def test_mean_pool_shares_the_two_dimensional_signature():
    """mean_pool is bracketing-decodable on the rotation and order-blind -- the
    corner is not unique to the gate.  Only the four-leaf pair separates them."""
    laws = Laws(np.random.default_rng(2))
    lex = make_lexicon(5, np.random.default_rng(3))
    t1, t2, n = rotation_pair(1)
    assert not np.allclose(compose_tree(t1, lex[:n], laws.mean_pool),
                           compose_tree(t2, lex[:n], laws.mean_pool))
    a, b, n = order_pair(2)
    assert np.allclose(compose_tree(a, lex[:n], laws.mean_pool),
                       compose_tree(b, lex[:n], laws.mean_pool))


# ------------------------------------------------------------ rival laws
def test_additive_law_is_bracketing_blind():
    lex = make_lexicon(7, np.random.default_rng(1))
    t1, t2, n = rotation_pair(3)
    add = lambda u, v: u + v
    assert np.allclose(compose_tree(t1, lex[:n], add), compose_tree(t2, lex[:n], add))


def test_gate_is_order_blind_and_rivals_are_not():
    """Commutativity is the axis that separates the gate from tensor and rnn."""
    laws = Laws(np.random.default_rng(2))
    lex = make_lexicon(5, np.random.default_rng(3))
    a, b, n = order_pair(2)
    for name in ("gate", "gate_norm", "additive", "mean_pool"):
        f = laws.get(name)
        assert np.allclose(compose_tree(a, lex[:n], f),
                           compose_tree(b, lex[:n], f), atol=1e-9), name
    for name in ("tensor", "rnn"):
        f = laws.get(name)
        assert not np.allclose(compose_tree(a, lex[:n], f),
                               compose_tree(b, lex[:n], f), atol=1e-6), name


def test_role_operators_are_norm_preserving():
    """R_L, R_R orthonormal scaled 1/sqrt(2): the tensor law preserves norm."""
    laws = Laws(np.random.default_rng(2))
    for r in (laws.RL, laws.RR):
        assert np.allclose(2.0 * r @ r.T, np.eye(NPT), atol=1e-9)


# --------------------------------------------------------- tree shape matching
def test_rotation_pair_is_shape_matched():
    """The two bracketings match on height and leaf-depth multiset at every depth."""
    def height(t):
        return 0 if isinstance(t, int) else 1 + max(height(t[0]), height(t[1]))

    def depths(t, d=0):
        return [d] if isinstance(t, int) else sorted(depths(t[0], d + 1) + depths(t[1], d + 1))

    for d in range(1, 6):
        t1, t2, n = rotation_pair(d)
        assert height(t1) == height(t2)
        assert depths(t1) == depths(t2)
        assert n == d + 2
    for d in range(1, 6):
        a, b, n = order_pair(d)
        assert height(a) == height(b)
        assert depths(a) == depths(b)


# =============================================================== phase codes
def test_carrier_ratio_fixes_the_number_of_addresses():
    """d* = f_beta / f_dtheta is an identity, not a bound."""
    for ratio in range(2, 10):
        fb = ratio * pc.F_DTH
        assert pc.n_addresses(fb) == ratio
        assert np.isclose(2 * np.pi / pc.n_addresses(fb), 2 * np.pi * pc.F_DTH / fb)


def test_angular_code_aliases_at_lag_dstar():
    """At ratio 4, depth d and depth d+4 receive the SAME phase address."""
    dphi = 2 * np.pi / 4
    for d in (1, 2):
        assert np.isclose((d * dphi) % (2 * np.pi), ((d + 4) * dphi) % (2 * np.pi))


def test_same_frequency_ancestors_collapse_exactly():
    """A sum of same-frequency sinusoids IS one sinusoid: the ancestor set is
    not merely hard to recover, it is not encoded."""
    t = np.linspace(0, 4, 2000, endpoint=False)
    m, psi = [0.5, 0.4, 0.2], [0.3, 2.1, 4.0]
    a = 1 + sum(mi * np.cos(2 * np.pi * 4 * t + p) for mi, p in zip(m, psi))
    z = sum(mi * np.exp(1j * p) for mi, p in zip(m, psi))
    b = 1 + np.abs(z) * np.cos(2 * np.pi * 4 * t + np.angle(z))
    assert np.abs(a - b).max() < 1e-12


def test_modulation_index_is_phase_offset_invariant():
    """MI(phi + c ; a) = MI(phi ; a) -- shifting phase only relabels bins."""
    t = np.linspace(0, 4, 4000, endpoint=False)
    am = 1 + 0.8 * np.cos(2 * np.pi * 4 * t + 0.7)

    def mi(ph, nb=18):
        e = np.linspace(-np.pi, np.pi, nb + 1)
        b = np.clip(np.digitize(ph, e) - 1, 0, nb - 1)
        p = np.array([am[b == k].mean() for k in range(nb)])
        p = p / p.sum()
        return (np.log(nb) + (p * np.log(p)).sum()) / np.log(nb)

    base = np.angle(np.exp(1j * 2 * np.pi * 4 * t))
    shifted = np.angle(np.exp(1j * (2 * np.pi * 4 * t + 2.0)))
    assert abs(mi(base) - mi(shifted)) < 1e-3


def test_band_calibrated_noise_hits_the_requested_snr():
    """Broadband SNR is meaningless here; add_noise calibrates band by band."""
    rng = np.random.default_rng(0)
    W = pc.patterns(np.random.default_rng(11))
    clean = pc.angular_trial(2, 16.0, 60.0, W, np.random.default_rng(1))
    for snr in (-10.0, 0.0):
        noisy = pc.add_noise(clean, snr, rng)
        f = np.fft.rfftfreq(clean.shape[1], d=1.0 / pc.FS)
        m = (f >= pc.GAM_BAND[0]) & (f <= pc.GAM_BAND[1])
        ps = np.mean(np.abs(np.fft.rfft(clean, axis=1)[:, m]) ** 2)
        pn = np.mean(np.abs(np.fft.rfft(noisy - clean, axis=1)[:, m]) ** 2)
        got = 10 * np.log10(ps / pn)
        assert abs(got - snr) < 1.5, (snr, got)


def test_patterns_are_orthonormal():
    W = pc.patterns(np.random.default_rng(11))
    assert np.allclose(W @ W.T, np.eye(pc.NNODE), atol=1e-9)


def test_containment_decoder_is_unbiased_when_graph_is_empty():
    """An empty PAC graph must give a uniform guess, not a default of depth 1."""
    M = np.zeros((pc.NNODE, pc.NNODE))
    picks = [pc.containment_decode(M, 0.01, np.random.default_rng(i)) for i in range(400)]
    counts = np.bincount(picks, minlength=8)[1:7]
    assert counts.min() > 30, counts        # nothing like a constant default


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
