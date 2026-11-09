"""Fig. 6: boundary arrangement of direct coupling (item 01 of the revision guide).

(a) Litinski's (15-to-1) block during its last steps [Litinski 2019, Fig. 11, steps 5-7]: qubit 1 holds the output;
    its Z boundaries (top/bottom) are the edges through which it enters the Z-type rotations; the d_X x d_X 'out'
    ancilla region below it reaches the outer edge of the block and is used to consume the output.
(b) D1 at the 120-cycle design (d_X=15, d_Z=7, d_X2=25, d_Z2=15): the upper bank keeps Litinski's orientation, the
    lower bank is reflected; one level-2 merge P_v (x) Z_A uses the 'out' region and the merge-ancilla row as one
    ancilla, touching Z boundaries only.
Drawn to scale in data-qubit sites; the faulty-T |+> strips (d_p1 x 1) are not drawn, as in Fig. 2.
usage: python3 make_fig6.py out.pdf
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "Carlito", "pdf.fonttype": 42, "mathtext.fontset": "cm", "font.size": 7.5})
TEAL, TEAL_E = "#247d83", "#195f64"
PINK, PINK_E = "#f4d7d2", "#ad5b50"
BEIGE, BEIGE_E = "#f2e0c5", "#ab8959"
OUTC = "#e9eef0"
NAVY = "#304967"
INK = "#253343"
ZC = "#c2410c"       # Z boundary colour
HATCH = "#c2410c"

dX, dZ, dX2, dZ2 = 15, 7, 25, 15
SW, BH = dX + 4 * dZ, 3 * dX


def rect(ax, x, y, w, h, fc, ec, lw=0.6, z=2, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def patch(ax, x, y, w, h, fc, label=None, tc="white", fs=6.5):
    """logical patch with Z boundaries top/bottom (thick) and X boundaries left/right (dashed)"""
    rect(ax, x, y, w, h, fc, fc, lw=0, z=3)
    ax.plot([x, x + w], [y, y], color=ZC, lw=1.8, zorder=5, solid_capstyle="butt")
    ax.plot([x, x + w], [y + h, y + h], color=ZC, lw=1.8, zorder=5, solid_capstyle="butt")
    ax.plot([x, x], [y, y + h], color=INK, lw=0.8, ls=(0, (2, 1.5)), zorder=5)
    ax.plot([x + w, x + w], [y, y + h], color=INK, lw=0.8, ls=(0, (2, 1.5)), zorder=5)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=6)


def block(ax, x0, y0, flip, tag):
    """Litinski 15-to-1 block at steps 5-7; flip=True reflects it about a horizontal axis (out on top)."""
    rows = [(0, "low"), (dX, "mid"), (2 * dX, "up")]
    Y = (lambda r: y0 + BH - r - dX) if flip else (lambda r: y0 + r)
    rect(ax, x0, y0, SW, BH, "white", BEIGE_E, lw=0.8, z=1)
    # upper / lower ancilla rows of the block (internal)
    rect(ax, x0, Y(2 * dX), SW, dX, BEIGE, BEIGE_E, z=2)
    ax.text(x0 + SW / 2, Y(2 * dX) + dX / 2, "level-1 ancilla", ha="center", va="center", fontsize=5.6, color=BEIGE_E, zorder=6)
    rect(ax, x0 + dX, Y(0), SW - dX, dX, BEIGE, BEIGE_E, z=2)
    ax.text(x0 + dX + (SW - dX) / 2, Y(0) + dX / 2, "level-1\nancilla", ha="center", va="center", fontsize=5.2, color=BEIGE_E, zorder=6)
    # out region
    rect(ax, x0, Y(0), dX, dX, OUTC, "#8a99a6", z=2, ls=(0, (1.5, 1)))
    ax.text(x0 + dX / 2, Y(0) + dX / 2, "out", ha="center", va="center", fontsize=6.0, color=INK, zorder=6)
    # middle row: qubit 1 (output) and qubits 2-5
    patch(ax, x0, Y(dX), dX, dX, TEAL, "1", fs=7)
    for k in range(4):
        patch(ax, x0 + dX + k * dZ, Y(dX), dZ, dX, "#d9c7a6", str(k + 2), tc=INK, fs=5.8)
    return Y(0)


fig = plt.figure(figsize=(7.0, 3.0))
# ---------------- (a)
axA = fig.add_axes([0.02, 0.22, 0.36, 0.66])
axA.set_xlim(-14, SW + 14); axA.set_ylim(-22, BH + 6); axA.set_aspect("equal"); axA.axis("off")
yo = block(axA, 0, 0, False, "")
axA.annotate("", xy=(dX / 2, -1), xytext=(dX / 2, -13), arrowprops=dict(arrowstyle="<->", lw=0.8, color=ZC))
axA.text(dX / 2 + 2.5, -8, "consumer: transport (L)\nor core ancilla (D1)", ha="left", va="center", fontsize=6.0, color=INK)
axA.text(-2, dX + dX / 2, r"$|A\rangle$", ha="right", va="center", fontsize=7, color=TEAL_E)
axA.text(SW / 2, BH + 9, "(a) supplier output slot (Litinski Fig. 11, steps 5-7)", ha="center", fontsize=7.6, color=INK)

# ---------------- (b)
W = 3 * dX2 + dZ2       # 90
H = 2 * BH + 3 * dX2    # 165
axB = fig.add_axes([0.42, 0.02, 0.57, 0.92])
axB.set_xlim(-22, W + 36); axB.set_ylim(-8, H + 20); axB.set_aspect("equal"); axB.axis("off")
yL, yAl, yC, yAu, yU = 0, BH, BH + dX2, BH + 2 * dX2, BH + 3 * dX2
rect(axB, 0, yAl, W, dX2, PINK, PINK_E, z=2)
rect(axB, 0, yAu, W, dX2, PINK, PINK_E, z=2)
axB.text(W + 2, yAu + dX2 / 2, "upper merge-\nancilla row", ha="left", va="center", fontsize=6.0, color=PINK_E)
axB.text(W + 2, yAl + dX2 / 2, "lower merge-\nancilla row", ha="left", va="center", fontsize=6.0, color=PINK_E)
xs = 0
for i, w in enumerate((dX2, dX2, dX2, dZ2)):
    patch(axB, xs, yC, w, dX2, NAVY, str(i + 1), fs=8)
    xs += w
axB.text(W + 2, yC + dX2 / 2, "CCZ core", ha="left", va="center", fontsize=6.0, color=NAVY)
for k in range(2):
    block(axB, k * SW, yU, False, "S1" if k == 0 else "")
    block(axB, k * SW, yL, True, "S3" if k == 0 else "")
    axB.text(k * SW + SW / 2, yU + BH + 1.5, f"S{k + 1}", ha="center", va="bottom", fontsize=6.3, color=INK)
    axB.text(k * SW + SW / 2, yL - 1.5, f"S{k + 3}", ha="center", va="top", fontsize=6.3, color=INK)
rect(axB, 0, 0, W, H, "none", "#9aa5b1", lw=0.7, z=7)
axB.text(W + 2, yU + BH / 2, "upper bank:\nLitinski\norientation", ha="left", va="center", fontsize=6.0, color=INK)
axB.text(W + 2, yL + BH / 2, "lower bank:\nreflected", ha="left", va="center", fontsize=6.0, color=INK)
# merge region of one rotation P_v (x) Z_A with P_v = Z_1 Z_4 (support from S1's out region along the upper row)
axB.add_patch(Rectangle((0, yAu), W, dX2, fc="none", ec=HATCH, hatch="////", lw=0, zorder=4, alpha=0.55))
axB.add_patch(Rectangle((0, yU), dX, dX, fc="none", ec=HATCH, hatch="////", lw=0, zorder=4, alpha=0.55))
axB.text(-2, yAu + dX2 / 2, r"$P_v\otimes Z_A$", ha="right", va="center", fontsize=7, color=ZC)
axB.text(W / 2, H + 15, "(b) D1: the core ancilla meets Z boundaries only", ha="center", fontsize=7.6, color=INK)

handles = [Line2D([0], [0], color=ZC, lw=1.8, label="Z boundary"),
           Line2D([0], [0], color=INK, lw=0.8, ls=(0, (2, 1.5)), label="X boundary"),
           Rectangle((0, 0), 1, 1, fc="none", ec=HATCH, hatch="////", lw=0, label=r"ancilla of one merge $P_v\otimes Z_A$")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=1, frameon=False, fontsize=6.5, handlelength=2.2)
fig.savefig(sys.argv[1])
print("saved", sys.argv[1])
