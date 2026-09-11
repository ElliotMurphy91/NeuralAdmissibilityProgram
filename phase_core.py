"""
Angular phase-address code vs. containment (PAC-graph) code.

Two rival generative models for how embedding depth is written into cortical
dynamics, each with its
matched read-out.

  (i)  ANGULAR      depth d -> delta-theta phase address Phi(d) = d*dphi,
                    physically realised as the beta trough at that phase.
                    Beta troughs occur f_beta/f_dtheta times per delta-theta
                    cycle, so the number of DISTINCT addresses is exactly that
                    ratio: dphi is not a free parameter, and d* = f_beta/f_dth.

  (ii) CONTAINMENT  depth d -> position in the partial order induced by the
                    directed PAC graph ("phase of X modulates amplitude of Y").
                    Each constituent carries an intrinsic low-frequency carrier;
                    a node's gamma amplitude is modulated by the carriers of ALL
                    its ancestors with weight decaying in distance, so the PAC
                    graph realizes dominance and its transitive reduction
                    realizes immediate dominance.  Depth is read from topology.

                    NOTE: the constituents' carriers must be at DISTINCT
                    frequencies.  At a common frequency the code is not merely
                    noisy but algebraically unidentifiable (see run_phase.py).

Terminals are matched: every trial contains the same seven contents on the same
seven orthonormal spatial patterns; only the assignment of content to depth
varies.  The probe content is fixed (index 0) and its depth is decoded.
"""
import numpy as np

FS = 500.0
T = 4.0
NT = int(FS * T)
TAX = np.arange(NT) / FS
NCH = 16
NNODE = 7                       # chain positions 0 (root) .. 6
F_DTH = 4.0
F_GAM = 80.0
GBW = 15.0
NBIN = 18
# intrinsic low-frequency carrier of each CONTENT (a fixed property of its site)
F_INTR = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
DEPTHS = np.arange(1, 7)


# --------------------------------------------------------------- utilities
def pink(n_ch, n_t, rng):
    f = np.fft.rfftfreq(n_t, d=1.0 / FS)
    amp = np.zeros_like(f)
    amp[1:] = 1.0 / f[1:]
    ph = rng.uniform(0, 2 * np.pi, (n_ch, f.size))
    X = np.fft.irfft(amp[None, :] * np.exp(1j * ph), n=n_t, axis=1)
    return X / X.std(axis=1, keepdims=True)


def patterns(rng):
    return np.linalg.qr(rng.normal(size=(NCH, NCH)))[0][:, :NNODE].T


def bp(x, lo, hi):
    f = np.fft.rfftfreq(x.shape[-1], d=1.0 / FS)
    X = np.fft.rfft(x, axis=-1)
    X[..., (f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[-1], axis=-1)


def _hstep(n):
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1; h[1:n // 2] = 2
    else:
        h[0] = 1; h[1:(n + 1) // 2] = 2
    return h


def analytic(x):
    return np.fft.ifft(np.fft.fft(x, axis=-1) * _hstep(x.shape[-1]), axis=-1)


LF_BAND = (1.0, 10.0)
GAM_BAND = (F_GAM - GBW - 5, F_GAM + GBW + 5)


def add_noise(S, snr_db, rng, bands=(LF_BAND, GAM_BAND)):
    """1/f background, calibrated band by band.

    A broadband SNR is meaningless here: 1/f noise carries almost no power at
    80 Hz, so a nominal -20 dB broadband SNR still leaves the gamma read-out
    essentially clean.  We therefore scale the 1/f background within each
    analysis band so that the band-limited SNR equals `snr_db` -- which is also
    how SNR is reported for real recordings.  The noise keeps its 1/f shape
    within each band.
    """
    N = pink(S.shape[0], S.shape[1], rng)
    f = np.fft.rfftfreq(S.shape[1], d=1.0 / FS)
    Nf = np.fft.rfft(N, axis=1)
    Sf = np.fft.rfft(S, axis=1)
    scaled = np.zeros(f.shape, dtype=bool)
    g_last = 1.0
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        psig = np.mean(np.abs(Sf[:, m]) ** 2)
        pnoi = np.mean(np.abs(Nf[:, m]) ** 2)
        g_last = np.sqrt(psig / (10 ** (snr_db / 10.0) * max(pnoi, 1e-30)))
        Nf[:, m] *= g_last
        scaled |= m
    Nf[:, ~scaled] *= g_last
    return S + np.fft.irfft(Nf, n=S.shape[1], axis=1)


def assign_contents(depth, rng):
    """content -> chain position; probe (content 0) goes to `depth`."""
    others = [k for k in range(NNODE) if k != depth]
    rng.shuffle(others)
    pos = {0: depth}
    for c, k in zip(range(1, NNODE), others):
        pos[c] = k
    return pos


# =============================================================== ANGULAR
def n_addresses(f_beta):
    return int(round(f_beta / F_DTH))


def angular_trial(depth, f_beta, snr_db, W, rng, return_parts=False):
    naddr = n_addresses(f_beta)
    dphi = 2 * np.pi / naddr
    pos = assign_contents(depth, rng)
    S = np.zeros((NCH, NT))
    dth = np.cos(2 * np.pi * F_DTH * TAX)
    beta = np.cos(2 * np.pi * f_beta * TAX)
    S += 1.0 * dth[None, :] + 0.6 * beta[None, :]
    sigma_t = 0.40 / f_beta
    envs = {}
    for c, k in pos.items():
        phi = (k * dphi) % (2 * np.pi)
        centres = phi / (2 * np.pi * F_DTH) + np.arange(0, T * F_DTH) / F_DTH
        env = np.zeros(NT)
        for c0 in centres:
            env += np.exp(-0.5 * ((TAX - c0) / sigma_t) ** 2)
        envs[k] = env
        g = env * np.cos(2 * np.pi * F_GAM * TAX + rng.uniform(0, 2 * np.pi))
        S += np.outer(W[c], g)
    X = add_noise(S, snr_db, rng)
    if return_parts:
        return X, dth, beta, envs, dphi
    return X


def angular_decode(X, f_beta, W, rng):
    dphi = 2 * np.pi / n_addresses(f_beta)
    ph = np.angle(analytic(bp(X.mean(0), 2.0, 6.0)))
    env = np.abs(analytic(bp(W[0] @ X, F_GAM - GBW, F_GAM + GBW)))
    phi_hat = np.angle(np.sum(env * np.exp(1j * ph)))
    targets = (DEPTHS * dphi) % (2 * np.pi)
    dist = np.abs(np.angle(np.exp(1j * (phi_hat - targets))))
    cand = np.flatnonzero(dist <= dist.min() + 1e-9)
    return int(DEPTHS[rng.choice(cand)])


# =========================================================== CONTAINMENT
def containment_trial(depth, snr_db, W, rng, m0=0.9, rho=0.5, common_freq=False):
    """If common_freq, every constituent carries the SAME low-frequency carrier
    (the code as literally stated in Sec 6.5) -- which is unidentifiable."""
    pos = assign_contents(depth, rng)
    by_pos = {k: c for c, k in pos.items()}
    psi = rng.uniform(0, 2 * np.pi, NNODE)
    freq = (np.full(NNODE, F_DTH) if common_freq else F_INTR)
    S = np.zeros((NCH, NT))
    for k in range(NNODE):
        c = by_pos[k]
        lf = np.cos(2 * np.pi * freq[c] * TAX + psi[k])
        a = np.ones(NT)
        for j in range(k):                       # every ancestor modulates
            cj = by_pos[j]
            a += m0 * (rho ** (k - j - 1)) * np.cos(
                2 * np.pi * freq[cj] * TAX + psi[j])
        a = np.clip(a, 0.05, None)
        g = a * np.cos(2 * np.pi * F_GAM * TAX + rng.uniform(0, 2 * np.pi))
        S += np.outer(W[c], 0.9 * lf + 0.55 * g)
    return add_noise(S, snr_db, rng)


def pac_matrix(X, W, common_freq=False):
    """M[i, j] = Tort MI( phase of content i's intrinsic carrier -> gamma
    amplitude of content j )."""
    P = W @ X
    am = np.abs(analytic(bp(P, F_GAM - GBW, F_GAM + GBW)))          # (NNODE, NT)
    M = np.zeros((NNODE, NNODE))
    edges = np.linspace(-np.pi, np.pi, NBIN + 1)
    for i in range(NNODE):
        f0 = F_DTH if common_freq else F_INTR[i]
        ph = np.angle(analytic(bp(P[i], f0 - 0.6, f0 + 0.6)))
        b = np.clip(np.digitize(ph, edges) - 1, 0, NBIN - 1)
        O = np.zeros((NT, NBIN)); O[np.arange(NT), b] = 1.0
        cnt = np.maximum(O.sum(0), 1)
        Bm = (O.T @ am.T) / cnt[:, None]                            # (NBIN, NNODE)
        p = np.clip(Bm / Bm.sum(0, keepdims=True), 1e-12, None)
        H = -(p * np.log(p)).sum(0)
        M[i] = (np.log(NBIN) - H) / np.log(NBIN)
    np.fill_diagonal(M, 0.0)
    return M


def containment_decode(M, tau, rng=None):
    """Threshold -> orient -> transitive closure -> transitive reduction;
    depth of the probe (content 0) = longest path to it in the reduction."""
    A = (M > tau) & (M > M.T)
    np.fill_diagonal(A, False)
    R = A.copy()
    for _ in range(NNODE):
        R2 = R | ((R.astype(int) @ R.astype(int)) > 0)
        if np.array_equal(R2, R):
            break
        R = R2
    red = A & ~((R.astype(int) @ R.astype(int)) > 0)
    dist = np.zeros(NNODE, dtype=int)
    for _ in range(NNODE):
        for j in range(NNODE):
            for i in range(NNODE):
                if red[i, j]:
                    dist[j] = max(dist[j], dist[i] + 1)
    if dist[0] == 0:          # probe unreachable / graph empty -> unbiased guess
        return int(rng.choice(DEPTHS)) if rng is not None else int(np.random.choice(DEPTHS))
    return int(np.clip(dist[0], 1, 6))
