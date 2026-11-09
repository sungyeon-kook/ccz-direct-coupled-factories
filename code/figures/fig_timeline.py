"""Fig. 3: local operating sequence for one supplied state in L, D1 and D2.

Example parameters (as in the manuscript): d_X = d_p2 = 15, d_m = 7, d_m2 = 13, R = 10,
nine-cycle Y branch (t_Y = floor(15/2) + 2 = 9), no queueing.
L follows Appendix B-A: output released at departure, d_m2-cycle transport, one cycle to clear
the |m> spot; decoding is overlapped (hidden) by the correction patch.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "Carlito",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "cm",
})

TEAL, TEAL_E = "#247d83", "#195f64"
YEL, YEL_E = "#f5d998", "#a98433"
PINK, PINK_E = "#f4d7d2", "#ad5b50"
GREY, NAVY = "#e8ebef", "#304967"
BEIGE, BEIGE_E = "#f2e0c5", "#ab8959"
INK = "#253343"

dX = dp2 = 15
dm, dm2, R = 7, 13, 10
tY = dp2 // 2 + 2          # 9
att = 6 * dm               # 42-cycle level-1 attempt

ROWS = ["Source slot", "Transport", "Spot / buffer", "Core merge", "Decode / readout", "Supplier"]
ROW_Y = {r: len(ROWS) - 1 - i for i, r in enumerate(ROWS)}
H = 0.56


def bar(ax, row, x0, x1, fc, ec=None, text=None, tc=INK, hatch=None, fs=6.3):
    y = ROW_Y[row]
    ax.add_patch(Rectangle((x0, y - H / 2), x1 - x0, H, fc=fc, ec=ec or fc, lw=0.8, hatch=hatch))
    if text:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=fs, color=tc)


def panel(ax, title, free, label_free):
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.7, len(ROWS) - 0.4)
    ax.set_xticks(range(0, 71, 10))
    ax.set_yticks([ROW_Y[r] for r in ROWS])
    ax.set_yticklabels(ROWS)
    ax.tick_params(axis="both", length=2.5, labelsize=6.8)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=11)
    ax.axvline(free, color=TEAL, ls=":", lw=1.0)
    ax.text(free + 1.2, len(ROWS) - 0.42, label_free, color=TEAL, fontsize=6.8, va="bottom")
    ax.set_xlabel("Cycles from input readiness", fontsize=7.5)


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), sharey=True)

# ---- L -------------------------------------------------------------------------------------
ax = axes[0]
panel(ax, r"L: transport to a $|m\rangle$ spot", 0, "slot free: 0")
ax.plot([0, 0], [ROW_Y["Source slot"] - H / 2, ROW_Y["Source slot"] + H / 2], color=TEAL, lw=2.2)
ax.text(1.5, ROW_Y["Source slot"], "released at departure", fontsize=5.8, color="#5b6b7c", va="center")
bar(ax, "Transport", 0, dm2, YEL, YEL_E, f"{dm2}")
bar(ax, "Spot / buffer", dm2, 2 * dm2 + 1, YEL, YEL_E, r"$|m\rangle$ spot")
bar(ax, "Core merge", dm2, 2 * dm2, PINK, PINK_E)
bar(ax, "Decode / readout", 2 * dm2, 2 * dm2 + R, GREY, "#b6bec8")
ax.text(2 * dm2 + R + 1.5, ROW_Y["Decode / readout"], "decode (hidden)", fontsize=6.0, color="#5b6b7c", va="center")
bar(ax, "Supplier", 0, att, BEIGE, BEIGE_E, "next 15-to-1 attempt")

# ---- D1 ------------------------------------------------------------------------------------
ax = axes[1]
free1 = dm2 + R + tY
panel(ax, "D1: read in place", free1, f"slot free: {free1}")
bar(ax, "Source slot", 0, free1, TEAL, TEAL_E)
ax.text(2, ROW_Y["Spot / buffer"], "not used", fontsize=6.3, color="#5b6b7c", va="center")
ax.text(2, ROW_Y["Transport"], "not used", fontsize=6.3, color="#5b6b7c", va="center")
bar(ax, "Core merge", 0, dm2, PINK, PINK_E)
bar(ax, "Decode / readout", dm2, dm2 + R, GREY, "#b6bec8")
bar(ax, "Decode / readout", dm2 + R, free1, NAVY)
bar(ax, "Supplier", free1 - dm, free1 - dm + att, BEIGE, BEIGE_E, "next 15-to-1 attempt")

# ---- D2 ------------------------------------------------------------------------------------
ax = axes[2]
h = max(dX, dp2)
panel(ax, "D2: handoff, then read buffer", h, f"slot free: {h}")
bar(ax, "Source slot", 0, h, TEAL, TEAL_E)
ax.text(2, ROW_Y["Transport"], "not used", fontsize=6.3, color="#5b6b7c", va="center")
bar(ax, "Spot / buffer", h, h + dm2 + R + tY, YEL, YEL_E)
bar(ax, "Core merge", h, h + dm2, PINK, PINK_E)
bar(ax, "Decode / readout", h + dm2, h + dm2 + R, GREY, "#b6bec8")
bar(ax, "Decode / readout", h + dm2 + R, h + dm2 + R + tY, NAVY)
bar(ax, "Supplier", h - dm, h - dm + att, BEIGE, BEIGE_E, "next 15-to-1 attempt")

for a in axes[1:]:
    a.tick_params(axis="y", length=0)

fig.subplots_adjust(left=0.115, right=0.995, bottom=0.17, top=0.86, wspace=0.07)
out = sys.argv[1]
fig.savefig(out)
print("saved", out, "D1 free", free1, "D1 supplier end", free1 - dm + att, "D2 buffer end", h + dm2 + R + tY,
      "D2 supplier end", h - dm + att)
