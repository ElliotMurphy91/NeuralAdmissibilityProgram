"""
Core objects for the NAP composition-law simulation.

Implements, per the accompanying manuscript:
  - Gabor-wavelet lexical atoms on a compact domain          (Sec 4.10)
  - the entropy-optimized Renyi gate  f (+)_{S_a,beta} g      (Sec 4.9)
  - five rival composition laws, incl. fixed mean-pooling     (Sec 7.2)
  - matched-terminal, matched-height bracketing contrasts     (Sec 5.1.1)
  - the balanced four-leaf faithfulness contrast              (Sec 4.9.1)
"""
import numpy as np

# ----------------------------------------------------------------- domain
NPT = 128
TAX = np.linspace(0.0, 1.0, NPT)


# ----------------------------------------------------------------- atoms
def make_lexicon(n, rng):
    """n Gabor-like wavelets  l_i(t) = a_i(t) cos(w_i t + phi_i)  on TAX.

    Returns (n, NPT) array, each row unit-variance, zero-mean.
    """
    out = np.empty((n, NPT))
    for i in range(n):
        centre = rng.uniform(0.2, 0.8)
        width = rng.uniform(0.10, 0.28)
        freq = rng.uniform(6.0, 26.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        env = np.exp(-0.5 * ((TAX - centre) / width) ** 2)
        w = env * np.cos(freq * 2 * np.pi * TAX + phase)
        w = w - w.mean()
        out[i] = w / (w.std() + 1e-12)
    return out


# ----------------------------------------------------------------- entropy
def gate_norm(f, g, beta, alpha=2.0, chunk=64):
    """Offset-corrected gate:  lam* f + (1 - lam*) g,  lam* the argmin of the full
    free-energy objective.  Identical argmin, but the -S(lam*)/beta constant is not
    carried forward, so the law is scale-preserving under nesting.
    """
    v, lam = gate(f, g, beta, alpha, return_lambda=True, chunk=chunk)
    f = np.atleast_2d(f); g = np.atleast_2d(g)
    return lam * f + (1.0 - lam) * g


def renyi_S(lam, alpha):
    """Renyi entropy of order alpha for the binary mixture (lam, 1-lam).

    alpha -> 1 is handled by the Shannon limit.
    """
    lam = np.clip(lam, 1e-12, 1 - 1e-12)
    if abs(alpha - 1.0) < 1e-9:
        return -(lam * np.log(lam) + (1 - lam) * np.log(1 - lam))
    return np.log(lam ** alpha + (1 - lam) ** alpha) / (1.0 - alpha)


# ----------------------------------------------------------------- the gate
_NLAM = 1201
_LAM = np.linspace(0.0, 1.0, _NLAM)


def gate(f, g, beta, alpha=2.0, return_lambda=False, chunk=64):
    """f (+)_{S_alpha,beta} g  =  min_{lam in [0,1]} { lam f + (1-lam) g - S_alpha(lam)/beta }

    Pointwise over the compact domain.  f, g : (n, NPT).

    Grid minimisation over 1201 knots (lam to ~8e-4).  At alpha = 2 the binary
    objective is convex in lam -- J''(lam) = 8 lam (1-lam) / (beta u(lam)^2) >= 0
    with u(lam) = lam^2 + (1-lam)^2 -- so the minimiser is unique and the grid
    agrees with a stationarity solve to grid resolution.  At alpha != 2 convexity
    on the interior is not asserted; the grid does not need it, since a scan over
    the knots returns the global minimiser by exhaustion whatever the shape of J.
    """
    f = np.atleast_2d(f)
    g = np.atleast_2d(g)
    S = renyi_S(_LAM, alpha)                       # (_NLAM,)
    pen = -S / beta                                # (_NLAM,)
    out = np.empty_like(f)
    lam_star = np.empty_like(f)
    for a in range(0, f.shape[0], chunk):
        b = min(a + chunk, f.shape[0])
        # (m, NPT, _NLAM)
        obj = (f[a:b, :, None] * _LAM[None, None, :]
               + g[a:b, :, None] * (1.0 - _LAM)[None, None, :]
               + pen[None, None, :])
        k = np.argmin(obj, axis=2)
        out[a:b] = np.take_along_axis(obj, k[:, :, None], axis=2)[:, :, 0]
        lam_star[a:b] = _LAM[k]
    if return_lambda:
        return out, lam_star
    return out


# ----------------------------------------------------------------- laws
class Laws:
    """Composition laws G(u, v) on (n, NPT) codes.

    mean_pool is the offset-corrected gate with lambda frozen at 1/2: it is
    commutative, non-associative and order-blind, so it shares the gate's
    two-dimensional signature.  It is the control for what the entropy term
    actually adds -- a content-dependent lambda*.

    Role matrices for the tensor/role-filler law and the recurrent weights are
    drawn once and held fixed, so every tree in a run uses the same operator.
    """

    def __init__(self, rng, beta=1.0, alpha=2.0):
        self.beta = beta
        self.alpha = alpha
        # orthonormal role operators, scaled 1/sqrt(2) so the law is norm-preserving
        self.RL = np.linalg.qr(rng.normal(size=(NPT, NPT)))[0] / np.sqrt(2.0)
        self.RR = np.linalg.qr(rng.normal(size=(NPT, NPT)))[0] / np.sqrt(2.0)
        # recurrent law
        s = 1.0 / np.sqrt(NPT)
        self.Wu = rng.normal(scale=s, size=(NPT, NPT))
        self.Wv = rng.normal(scale=s, size=(NPT, NPT))

    def additive(self, u, v):
        return u + v

    def mean_pool(self, u, v):
        return 0.5 * (u + v)

    def tensor(self, u, v):
        return u @ self.RL.T + v @ self.RR.T

    def rnn(self, u, v):
        return np.tanh(u @ self.Wu.T + v @ self.Wv.T)

    def gate(self, u, v):
        return gate(u, v, self.beta, self.alpha)

    def gate_norm(self, u, v):
        return gate_norm(u, v, self.beta, self.alpha)

    def get(self, name):
        return {"gate": self.gate, "gate_norm": self.gate_norm,
                "additive": self.additive, "tensor": self.tensor,
                "rnn": self.rnn, "mean_pool": self.mean_pool}[name]


NAMES = ["gate", "gate_norm", "additive", "tensor", "rnn", "mean_pool"]


# ----------------------------------------------------------------- trees
def compose_tree(tree, atoms, G):
    """tree: nested tuples of leaf indices.  atoms: (n_leaf, NPT).  G: binary law."""
    if isinstance(tree, int):
        return atoms[tree]
    l = compose_tree(tree[0], atoms, G)
    r = compose_tree(tree[1], atoms, G)
    return G(np.atleast_2d(l), np.atleast_2d(r))[0]


def rotation_pair(depth):
    """A three-terminal rotation embedded at `depth` levels below the root.

    depth = 1 -> the rotation IS the root.
    Both members share height, node count, terminal order and leaf-depth
    multiset; they differ only in the bracketing of leaves 0,1,2.
    """
    left = ((0, 1), 2)
    right = (0, (1, 2))
    nxt = 3
    for _ in range(depth - 1):
        left = (left, nxt)
        right = (right, nxt)
        nxt += 1
    return left, right, nxt        # nxt = number of terminals


def balanced_pair():
    """The balanced four-leaf faithfulness contrast, ((A B)(C D)) vs ((A C)(B D)).

    The two members are distinct elements of the free commutative
    non-associative magma with the same terminals, the same leaf-depth
    multiset AND the same content-to-depth map (every leaf at depth 2).  Any
    content-independent mixture maps both to the same function -- under
    mean-pooling both are (A+B+C+D)/4 -- so this pair separates the entropy
    gate from fixed pooling, and it is the pair on which the corrected gate's
    faithfulness fails as beta -> 0.  It is not a matched-string contrast:
    ((A C)(B D)) has no linearisation as the string A B C D.
    """
    return ((0, 1), (2, 3)), ((0, 2), (1, 3)), 4


def order_pair(depth):
    """Same spine, but the contrast is daughter ORDER of the deepest pair."""
    a = ((0, 1), 2)
    b = ((1, 0), 2)
    nxt = 3
    for _ in range(depth - 1):
        a = (a, nxt)
        b = (b, nxt)
        nxt += 1
    return a, b, nxt
