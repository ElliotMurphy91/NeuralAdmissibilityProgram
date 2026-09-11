import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from gate_core import TAX

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.6,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 8, "axes.titlesize": 8.4, "axes.titleweight": "regular",
    "axes.titlepad": 5, "axes.labelpad": 2.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.labelspacing": 0.32, "legend.borderpad": 0.1,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.color": "#3d4a56", "ytick.color": "#3d4a56",
    "axes.edgecolor": "#3d4a56", "axes.labelcolor": "#0A1F36",
    "text.color": "#0A1F36", "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

INK = "#0A1F36"
C = {"gate": "#12807F", "gate_norm": "#7FC6BE", "additive": "#9AA5B1",
     "tensor": "#E27D29", "rnn": "#3C53A4", "mean_pool": "#B5893F", "sat_tied": "#8E3B6B", "meld": "#C0392B"}
PRETTY = {"gate": "Rényi gate  $\\oplus_{S_2,\\beta}$",
          "gate_norm": "gate, offset-corrected",
          "additive": "additive  $u+v$", "tensor": "tensor / role–filler",
          "rnn": "recurrent", "mean_pool": "mean-pooling  $\\frac{1}{2}(u+v)$",
          "sat_tied": "tied-weight  $\\tanh(W(u+v)/2)$",
          "meld": "Meld  $\\tanh(\\kappa W\\,[\\lambda^{\\star}u+(1-\\lambda^{\\star})v])$"}
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "tl", ["#f7f9fa", "#cfe4e2", "#8dc4c0", "#3f9d99", "#12807F", "#0a4d4c"])

R = np.load("results.npz", allow_pickle=True)
LAWS = [str(x) for x in R["LAWS"]]
D, S = R["DEPTHS"], R["SNRS"]


def tag(ax, s, dx=-0.235, dy=1.13):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left", color=INK)


def ci(a, axis=0):
    m = a.mean(axis)
    e = 1.96 * a.std(axis, ddof=1) / np.sqrt(a.shape[axis])
    return m, e


# ================================================================= FIGURE 1
fig = plt.figure(figsize=(7.2, 4.15))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.62], hspace=0.52, wspace=0.36,
                      left=0.07, right=0.985, top=0.9, bottom=0.11)

# ---- a: gate waveforms + lambda*
axA, axL = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
axA.plot(TAX, R["demo_f"], color="#c3ccd4", lw=0.9)
axA.plot(TAX, R["demo_g"], color="#c3ccd4", lw=0.9, ls=(0, (2.5, 1.5)))
axA.text(0.33, 0.99, "$f$", transform=axA.transAxes, color="#8b96a2", fontsize=7.5, va="top")
axA.text(0.42, 0.99, "$g$", transform=axA.transAxes, color="#8b96a2", fontsize=7.5, va="top")
shades = ["#9ad6d1", "#2f9c98", "#0a4d4c"]
for i, (b, sh) in enumerate(zip([0.3, 1.0, 4.0], shades)):
    axA.plot(TAX, R[f"demo_v{i}"] - R[f"demo_v{i}"].mean(), color=sh, lw=1.25,
             label=f"$\\beta={b}$")
    axL.plot(TAX, R[f"demo_lam{i}"], color=sh, lw=1.1)
axA.legend(loc="lower right", ncol=3, columnspacing=0.9, handletextpad=0.45,
           borderaxespad=0.1)
axA.set_ylabel("amplitude"); axA.set_xticklabels([])
axA.set_title("Entropy-optimized gate", loc="left")
axL.axhline(0.5, color="#c3ccd4", lw=0.6, zorder=0)
axL.set_ylabel("$\\lambda^{\\star}(t)$"); axL.set_xlabel("domain $t$")
axL.set_ylim(-0.06, 1.06); axL.set_yticks([0, 0.5, 1])
tag(axA, "a")

# ---- b: associator vs beta
axB, axB2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
be, Ab = R["betas"], R["A_beta"]
m, e = ci(Ab, 1)
axB.fill_between(be, m - e, m + e, color=C["gate"], alpha=0.18, lw=0)
axB.plot(be, m, color=C["gate"], lw=1.5)
pl = R["plateau"].mean()
axB.axhline(pl, color="#E27D29", lw=0.9, ls=(0, (3, 2)))
axB.text(1.3e-3, pl * 1.09, "$|\\kappa|\\,\\|h-f\\|$", color="#E27D29", fontsize=7)
axB.axhline(0, color="#c3ccd4", lw=0.6, zorder=0)
axB.set_xscale("log"); axB.set_xticklabels([])
axB.set_ylabel("associator norm", fontsize=7.6)
axB.set_title("Non-associativity vs. temperature", loc="left")
axB.set_ylim(-0.15, pl * 1.28)
d = np.diff(m)
axB2.axhline(0, color="#c3ccd4", lw=0.6, zorder=0)
axB2.plot(be[1:], d, color=C["gate"], lw=1.0, marker="o", ms=1.7, mfc=C["gate"], mec="none")
axB2.set_xscale("log"); axB2.set_xlabel("inverse temperature  $\\beta$")
axB2.set_ylabel("$\\Delta$", labelpad=6)
axB2.text(0.97, 0.12, f"0 / {len(be) - 1} increases", transform=axB2.transAxes, ha="right",
          fontsize=6.6, color="#5a6674")
tag(axB, "b")

# ---- c: associator vs alpha
axC, axC2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
al, Aa = R["alphas"], R["A_alpha"]
m2, e2 = ci(Aa, 1)
axC.fill_between(al - 1, m2 - e2, m2 + e2, color=C["gate"], alpha=0.18, lw=0)
axC.plot(al - 1, m2, color=C["gate"], lw=1.5, label="gate")
axC.axhline(0, color="#c3ccd4", lw=0.6, zorder=0)
axC.set_xlim(-0.02, 1.0)
axC.set_ylabel("associator norm", fontsize=7.6)
axC.set_xlabel("$\\alpha - 1$")
axC.set_title("Vanishes at the Shannon point", loc="left")
sl, ic = np.polyfit(np.log(al[:12] - 1), np.log(m2[:12]), 1)
axC2.plot(al[:14] - 1, m2[:14], color=C["gate"], lw=0, marker="o", ms=2.4,
          mfc="none", mew=0.8)
xx = np.array([al[0] - 1, al[13] - 1])
axC2.plot(xx, np.exp(ic) * xx ** sl, color="#E27D29", lw=0.9, ls=(0, (3, 2)))
axC2.set_xscale("log"); axC2.set_yscale("log")
axC2.set_xlabel("$\\alpha - 1$"); axC2.set_ylabel("norm", labelpad=1)
axC2.text(0.05, 0.86, f"slope {sl:.3f}", transform=axC2.transAxes, fontsize=6.8,
          color="#E27D29")
tag(axC, "c")
fig.savefig("Figure_S1.png"); fig.savefig("Figure_S1.pdf")
plt.close(fig)

# ================================================================= FIGURE 2
fig = plt.figure(figsize=(7.2, 4.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.58, wspace=0.42,
                      left=0.065, right=0.985, top=0.9, bottom=0.1)

A = R["ACC_bracket"]; O = R["ACC_order"]
s0 = list(S).index(0.0)
THR = 0.5 + 1.645 * np.sqrt(0.25 / 240)
ORDER = ["tensor", "rnn", "meld", "sat_tied", "mean_pool", "gate_norm", "gate", "additive"]

# ---- a: bracketing accuracy vs depth
axA = fig.add_subplot(gs[0, 0:2])
axA.axhspan(0.5 - (THR - 0.5), THR, color="#eceff1", lw=0, zorder=0)
axA.axhline(0.5, color="#b9c2ca", lw=0.6, zorder=1)
for nm in ORDER:
    li = LAWS.index(nm)
    m, e = ci(A[li, :, s0].T, 0)
    axA.fill_between(D, m - e, m + e, color=C[nm], alpha=0.16, lw=0, zorder=2)
    axA.plot(D, m, color=C[nm], lw=2.0 if nm == "meld" else 1.5, marker="o", ms=3.2, mec="white", mew=0.6,
             label=PRETTY[nm], zorder=4 if nm == "meld" else 3,
             ls=(0, (3, 1.5)) if nm == "mean_pool" else (0, (1, 1.2)) if nm == "sat_tied" else "-")
axA.set_xticks(D); axA.set_ylim(0.44, 1.04)
axA.set_xlabel("embedding depth of the rotation,  $d$")
axA.set_ylabel("bracketing decoding accuracy")
axA.set_title("Bracketing decodability, matched terminals   (0 dB)", loc="left")
axA.set_xlim(0.7, 8.8)
axA.legend(loc="center right", bbox_to_anchor=(1.005, 0.5), fontsize=6.0, handlelength=1.3)
tag(axA, "a", dx=-0.115)

# ---- b: powered region, Renyi gate | Meld
subB = gs[0, 2].subgridspec(1, 2, wspace=0.12)
axB = fig.add_subplot(subB[0]); axB2 = fig.add_subplot(subB[1])
for ax, nm, ttl in ((axB, "gate", "Rényi gate"), (axB2, "meld", "Meld")):
    li = LAWS.index(nm)
    H = A[li, :, :].mean(2).T                                  # (SNR, depth)
    im = ax.imshow(H, origin="lower", aspect="auto", cmap=TEAL_CMAP, vmin=0.5, vmax=1.0,
                   extent=[0.5, 5.5, -0.5, len(S) - 0.5])
    if (H > THR).any() and (H < THR).any():
        ax.contour(np.linspace(1, 5, 5), np.arange(len(S)), H, levels=[THR],
                   colors="#E27D29", linewidths=1.6, linestyles="--")
    ax.set_xticks(D); ax.set_xlabel("depth  $d$")
    ax.set_title(ttl, loc="left", fontsize=7.6)
axB.set_yticks(range(len(S))); axB.set_yticklabels([f"{int(s):+d}" for s in S])
axB.set_ylabel("SNR (dB)")
axB2.set_yticks(range(len(S))); axB2.set_yticklabels([])
cb = fig.colorbar(im, ax=[axB, axB2], pad=0.03, fraction=0.06)
cb.set_ticks([0.5, 0.75, 1.0]); cb.ax.tick_params(labelsize=6.4, width=0.6, length=2)
cb.outline.set_linewidth(0.6)
tag(axB, "b", dx=-0.62, dy=1.26)

# ---- c: law - additive gap, gate (teal) and Meld (red)
axC = fig.add_subplot(gs[1, 0])
axC.axhline(0, color="#b9c2ca", lw=0.6, zorder=0)
gi = LAWS.index("gate"); mi_ = LAWS.index("meld")
for si, sh, shm in zip([list(S).index(v) for v in [10., 5., 0.]],
                       ["#0a4d4c", "#2f9c98", "#9ad6d1"], ["#7b1f16", "#C0392B", "#e8a39a"]):
    g = (A[gi, :, si] - A[LAWS.index("additive"), :, si]); m, e = ci(g.T, 0)
    axC.fill_between(D, m - e, m + e, color=sh, alpha=0.18, lw=0)
    axC.plot(D, m, color=sh, lw=1.5, marker="o", ms=3.0, mec="white", mew=0.6, label=f"{int(S[si]):+d} dB")
    gm = (A[mi_, :, si] - A[LAWS.index("additive"), :, si]); m2_, e2_ = ci(gm.T, 0)
    axC.fill_between(D, m2_ - e2_, m2_ + e2_, color=shm, alpha=0.18, lw=0)
    axC.plot(D, m2_, color=shm, lw=1.5, marker="o", ms=3.0, mec="white", mew=0.6, ls=(0, (3, 1.5)))
axC.set_xticks(D); axC.set_xlabel("depth  $d$")
axC.set_ylabel("law $-$ additive  (acc.)")
axC.set_title("Gate gap closes; Meld gap holds", loc="left")
axC.legend(loc="center right", bbox_to_anchor=(1.02, 0.71), title="gate solid, Meld dashed", title_fontsize=6.0, fontsize=6.0)
tag(axC, "c")

# ---- d: order decoding
axD = fig.add_subplot(gs[1, 1])
axD.axhspan(0.5 - (THR - 0.5), THR, color="#eceff1", lw=0, zorder=0)
axD.axhline(0.5, color="#b9c2ca", lw=0.6, zorder=1)
for nm in ORDER:
    li = LAWS.index(nm)
    m, e = ci(O[li, :, s0].T, 0)
    axD.fill_between(D, m - e, m + e, color=C[nm], alpha=0.16, lw=0, zorder=2)
    axD.plot(D, m, color=C[nm], lw=1.5, marker="o", ms=3.2, mec="white", mew=0.6, zorder=3,
             ls=(0, (3, 1.5)) if nm == "mean_pool" else (0, (1, 1.2)) if nm == "sat_tied" else "-")
axD.set_xticks(D); axD.set_ylim(0.44, 1.04)
axD.set_xlabel("depth  $d$"); axD.set_ylabel("daughter-order accuracy")
axD.set_title("Commutativity separates the laws", loc="left")
tag(axD, "d")

# ---- e: representational divergence from the additive law, gate and Meld
axE = fig.add_subplot(gs[1, 2])
for key, col, lab in (("DIV", C["gate"], "gate"), ("DIV_meld", C["meld"], "Meld")):
    DV = R[key]
    for j, (mk, brk) in enumerate([("o", "$((AB)C)$"), ("s", "$(A(BC))$")]):
        m, e = ci(DV[:, :, j].T, 0)
        axE.errorbar(D + (j - 0.5) * 0.08, m, yerr=e, color=col, lw=1.4,
                     marker=mk, ms=3.4, mfc="white" if j else col, mew=0.9,
                     capsize=1.6, elinewidth=0.8, label=f"{lab} {brk}")
axE.set_xticks(D); axE.set_xlabel("depth  $d$")
axE.set_ylabel("$r$ (law, additive)")
axE.set_title("Divergence is bracketing-blind", loc="left")
axE.legend(loc="center left", ncol=2, columnspacing=0.6, handletextpad=0.4, fontsize=6.0, bbox_to_anchor=(0.0, 0.42))
tag(axE, "e")

fig.savefig("Figure_S2.png"); fig.savefig("Figure_S2.pdf")
plt.close(fig)
print("ok")
print("threshold", round(THR, 4), "| alpha slope", round(sl, 3))
