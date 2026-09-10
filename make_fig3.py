import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from phase_core import TAX, F_DTH

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
C_A4, C_A7, C_CT = "#E27D29", "#3C53A4", "#12807F"
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "tl", ["#f7f9fa", "#cfe4e2", "#8dc4c0", "#3f9d99", "#12807F", "#0a4d4c"])
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "dp", ["#12807F", "#4b8fbe", "#8a7fb5", "#c96f8a", "#E27D29"])
DC = [DEPTH_CMAP(x) for x in np.linspace(0, 1, 6)]

R = np.load("results_phase.npz", allow_pickle=True)
S = R["SNRS"]; N = int(R["NTRI"])
s0 = list(S).index(0.0)
D = np.arange(1, 7)


def tag(ax, s, dx=-0.26, dy=1.22):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="left", color=INK)


def alias_index(C, dstar=4):
    i, j = np.meshgrid(D, D, indexing="ij")
    lag = np.abs(i - j)
    return C[lag == dstar].mean() - C[(lag != dstar) & (lag != 0)].mean()


fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.72, wspace=0.44,
                      left=0.065, right=0.98, top=0.9, bottom=0.09)

# ------------------------------------------------------------------ a: signal
sub = gs[0, 0].subgridspec(2, 1, height_ratios=[0.62, 1], hspace=0.18)
axC, axE = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
w = TAX <= 1.0 / F_DTH                      # one delta-theta cycle
t = TAX[w] * 1000
axC.plot(t, R["demo_dth"][w], color="#8b96a2", lw=1.4)
axC.plot(t, 0.55 * R["demo_beta"][w], color="#c3ccd4", lw=0.9)
axC.set_ylim(-1.4, 1.5); axC.set_xticks([]); axC.set_yticks([])
for sp in ["left", "bottom"]:
    axC.spines[sp].set_visible(False)
axC.set_title("Four β troughs per δ–θ cycle", loc="left")
for d in range(1, 7):
    e = R[f"demo_env{d}"][w]
    axE.plot(t, e / e.max() * 0.85 + (6 - d), color=DC[d - 1],
             lw=1.5 if d <= 4 else 1.2, ls="-" if d <= 4 else (0, (2.2, 1.4)))
axE.set_yticks(np.arange(6)); axE.set_yticklabels([6, 5, 4, 3, 2, 1])
axE.set_ylabel("depth  $d$"); axE.set_xlabel("time within cycle (ms)")
axE.set_ylim(-0.35, 6.05)
axE.set_title("γ bursts: $d$=5 lands on $d$=1", loc="left")
tag(axC, "a", dx=-0.26, dy=1.42)

# -------------------------------------------------------- b, c: confusions
for cell, key, ttl, lab in [
        (gs[0, 1], "C_a4", "Angular,  $f_β/f_{δθ}$ = 4", "b"),
        (gs[0, 2], "C_cont", "Containment (PAC graph)", "c")]:
    ax = fig.add_subplot(cell)
    im = ax.imshow(R[key][s0], cmap=TEAL_CMAP, vmin=0, vmax=1,
                   extent=[0.5, 6.5, 6.5, 0.5])
    ax.set_xticks(D); ax.set_yticks(D)
    ax.set_xlabel("decoded  $\\hat{d}$"); ax.set_ylabel("true  $d$")
    ax.set_title(ttl, loc="left")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.055)
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=6.2, width=0.6, length=2)
    cb.outline.set_linewidth(0.6)
    tag(ax, lab, dx=-0.30)
    if key == "C_a4":                      # ring the d <-> d+4 fingerprint
        for (i, j) in [(1, 5), (2, 6), (5, 1), (6, 2)]:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec="#E27D29", lw=1.3))

# ------------------------------------------------------- d: accuracy vs depth
ax = fig.add_subplot(gs[1, 0])
ax.axhline(1 / 6, color="#b9c2ca", lw=0.6, zorder=0)
for key, col, lab in [("C_a4", C_A4, "angular,  $f_β/f_{δθ}$ = 4"),
                      ("C_a7", C_A7, "angular,  $f_β/f_{δθ}$ = 7"),
                      ("C_cont", C_CT, "containment")]:
    acc = np.diag(R[key][s0])
    e = 1.96 * np.sqrt(acc * (1 - acc) / N)
    ax.fill_between(D, acc - e, acc + e, color=col, alpha=0.16, lw=0)
    ax.plot(D, acc, color=col, lw=1.6, marker="o", ms=3.4, mec="white", mew=0.6,
            label=lab)
ax.set_xticks(D); ax.set_ylim(0, 1.08)
ax.set_xlabel("embedding depth  $d$"); ax.set_ylabel("depth decoding accuracy")
ax.set_title("Three distinct depth profiles", loc="left")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
tag(ax, "d")

# ------------------------------------------------------ e: alias index vs SNR
ax = fig.add_subplot(gs[1, 1])
ax.axhline(0, color="#b9c2ca", lw=0.6, zorder=0)
for key, col in [("C_a4", C_A4), ("C_a7", C_A7), ("C_cont", C_CT)]:
    A = np.array([alias_index(R[key][si]) for si in range(len(S))])
    ax.plot(S, A, color=col, lw=1.6, marker="o", ms=3.4, mec="white", mew=0.6)
for n, sh in zip([20, 100], ["#7e8a96", "#b9c2ca"]):
    ax.axhline(0.5264 / np.sqrt(n), color=sh, lw=0.8, ls=(0, (3, 2)))
    ax.text(5.9, 0.5264 / np.sqrt(n) + 0.022, f"N={n}", fontsize=6.2, color=sh,
            va="bottom", ha="right")
ax.set_xlabel("γ-band SNR (dB)"); ax.set_ylabel("aliasing index")
ax.set_title("Detectability of the fingerprint", loc="left")
ax.set_xlim(-21, 6.5); ax.set_ylim(-0.12, 0.62)
tag(ax, "e")

# --------------------------------------------------- f: carrier-ratio rule
ax = fig.add_subplot(gs[1, 2])
ra, rc = R["ratios"], R["ratio_acc"]
ax.axhline(1 / 6, color="#b9c2ca", lw=0.6, zorder=0)
ax.axvline(6, color="#E27D29", lw=0.9, ls=(0, (3, 2)), zorder=0)
ax.plot(ra, rc, color=C_A4, lw=1.6, marker="o", ms=3.6, mec="white", mew=0.6)
ax.plot([4], [rc[list(ra).index(4)]], marker="o", ms=7.5, mfc="none",
        mec=INK, mew=1.1)
ax.annotate("16 / 4 Hz", xy=(4, rc[list(ra).index(4)]), xytext=(2.3, 0.86),
            fontsize=6.8, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax.set_xticks(ra); ax.set_ylim(0, 1.08)
ax.set_xlabel("carrier ratio  $f_β / f_{δθ}$"); ax.set_ylabel("accuracy,  $d$ = 1…6")
ax.set_title("Δφ is not free: addresses = ratio", loc="left")
tag(ax, "f")

fig.savefig("Figure_S3.png"); fig.savefig("Figure_S3.pdf")
print("ok")
for key in ["C_a4", "C_a7", "C_cont"]:
    print(key, "alias idx:", np.round([alias_index(R[key][i]) for i in range(len(S))], 3))
