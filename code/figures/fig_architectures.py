"""Fig. 2: architectures at the 120-cycle operating point of Table 2, drawn to scale in data-qubit sites.

(a) L (Litinski interface): component areas of Eq. (3) arranged after Litinski Fig. 13(a)
(b) D1, (c) D2: enclosing rectangles of Eqs. (5), (6)
(d) footprint by component
All dimensions are computed here and asserted against the published totals.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.rcParams.update({
    "font.family": "Carlito",
    "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
    "font.size": 7.5,
})

TEAL, TEAL_E = "#247d83", "#195f64"
YEL, YEL_E = "#f5d998", "#a98433"
YEL_L = "#fbecc4"
PINK, PINK_E = "#f4d7d2", "#ad5b50"
GREYU, GREYU_E = "#e8ebef", "#b6bec8"
NAVY = "#304967"
BEIGE, BEIGE_E = "#f2e0c5", "#ab8959"
INK = "#253343"

# ------------------------------------------------------------------ designs (Table 2, 120-cycle)
# L : n=4, level 1 (15,5,7); d_p1=5, level 2 (25,13,17); d_p2=15
L = dict(n=4, dX=15, dZ=5, dm=7, dp1=5, dX2=25, dZ2=13, dm2=17, dp2=15)
# D1: n=4, level 1 (15,7,5); d_p1=7, level 2 (25,15,13)
D1 = dict(n=4, dX=15, dZ=7, dm=5, dp1=7, dX2=25, dZ2=15, dm2=13)
# D2: n=4, level 1 (13,7,7); d_p1=7, level 2 (25,15,17); d_p2=15
D2 = dict(n=4, dX=13, dZ=7, dm=7, dp1=7, dX2=25, dZ2=15, dm2=17, dp2=15)

# --- areas
def q_L(p):
    n, dX, dZ, dp1, dX2, dZ2, dp2 = (p[k] for k in ("n", "dX", "dZ", "dp1", "dX2", "dZ2", "dp2"))
    core = 2 * (3 * dX2 + dZ2) * 3 * dX2
    sup = 2 * n * ((dX + 4 * dZ) * 3 * dX + 2 * dp1)
    chan = 2 * n * (dX + 4 * dZ) * dp2 / 2
    inter = 2 * (20 * dp2 ** 2 + 2 * dX2 * dp2)
    return core, sup, chan + inter


core_L, sup_L, int_L = q_L(L)
tot_L = core_L + sup_L + int_L


def dims_D(p, buffered):
    W = max(3 * p["dX2"] + p["dZ2"], (p["n"] // 2) * (p["dX"] + 4 * p["dZ"]))
    H = 6 * p["dX"] + 3 * p["dX2"] + (2 * p["dp2"] if buffered else 0)
    return W, H


W1, H1 = dims_D(D1, False)
W2, H2 = dims_D(D2, True)
tot_D1 = 2 * W1 * H1 + 4 * D1["n"] * D1["dp1"]
tot_D2 = 2 * W2 * H2 + 4 * D2["n"] * D2["dp1"]
assert (round(tot_L), tot_D1, tot_D2) == (38480, 29812, 33052), (tot_L, tot_D1, tot_D2)
assert (W1, H1, W2, H2) == (90, 165, 90, 183)


def comp_D(p, W, H, buffered):
    core = 2 * (3 * p["dX2"] * W)                       # two ancilla rows + core row, full width
    sup = 2 * (p["n"] * (p["dX"] + 4 * p["dZ"]) * 3 * p["dX"] // 1) + 4 * p["n"] * p["dp1"]
    sup = 2 * (p["n"] * (p["dX"] + 4 * p["dZ"]) * 3 * p["dX"]) + 4 * p["n"] * p["dp1"]
    intf = 2 * W * 2 * p["dp2"] if buffered else 0
    return core, sup, intf


core_1, sup_1, int_1 = comp_D(D1, W1, H1, False)
core_2, sup_2, int_2 = comp_D(D2, W2, H2, True)
un_1 = tot_D1 - core_1 - sup_1 - int_1
un_2 = tot_D2 - core_2 - sup_2 - int_2
print("L  :", sup_L, core_L, int_L, tot_L)
print("D1 :", sup_1, core_1, int_1, un_1, tot_D1)
print("D2 :", sup_2, core_2, int_2, un_2, tot_D2)
assert int_L == 12600 and int_2 == 5400


# ------------------------------------------------------------------ drawing helpers
def rect(ax, x, y, w, h, fc, ec=None, lw=0.7, z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec or fc, lw=lw, zorder=z))


def txt(ax, x, y, s, fs=6.0, color=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=6, **kw)


def supplier(ax, x, y, w, h, label, params, z=3):
    rect(ax, x, y, w, h, BEIGE, BEIGE_E, z=z)
    txt(ax, x + w / 2, y + h / 2, label + "\n" + params, fs=5.6)


def core_row(ax, x0, y0, p, W):
    """merge ancilla / core / merge ancilla stack (width W): returns nothing"""
    dX2, dZ2 = p["dX2"], p["dZ2"]
    rect(ax, x0, y0, W, dX2, PINK, PINK_E)
    txt(ax, x0 + W / 2, y0 + dX2 / 2, "merge ancilla", fs=6.0, color=PINK_E)
    rect(ax, x0, y0 + dX2, W, dX2, NAVY, "white", lw=0.6)
    xs = x0
    for i in range(3):
        rect(ax, xs, y0 + dX2, dX2, dX2, NAVY, "white", lw=0.8)
        txt(ax, xs + dX2 / 2, y0 + dX2 + dX2 / 2, str(i + 1), fs=8.5, color="white")
        xs += dX2
    rect(ax, xs, y0 + dX2, dZ2, dX2, NAVY, "white", lw=0.8)
    txt(ax, xs + dZ2 / 2, y0 + dX2 + dX2 / 2, "4", fs=8.5, color="white")
    rect(ax, x0, y0 + 2 * dX2, W, dX2, PINK, PINK_E)
    txt(ax, x0 + W / 2, y0 + 2 * dX2 + dX2 / 2, "merge ancilla", fs=6.0, color=PINK_E)


def dim_arrows(ax, W, H, label_level2, qubits, title):
    ax.annotate("", xy=(W + 5, 0), xytext=(W + 5, H),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text(W + 11, H / 2, f"H = {H}", rotation=90, fontsize=6.5, color=INK, va="center", ha="center")
    ax.annotate("", xy=(0, -6), xytext=(W, -6),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text(W / 2, -14, f"W = {W}", fontsize=6.5, color=INK, ha="center", va="center")
    ax.text(W / 2, -24, label_level2, fontsize=6.3, color=INK, ha="center", va="center")
    ax.text(W / 2, H + 14, qubits, fontsize=7.4, color=INK, ha="center", va="center")
    ax.text(W / 2, H + 31, title, fontsize=8.4, color=INK, ha="center", va="center")


def draw_D(ax, p, W, H, buffered, label_level2, qubits, title):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    dX, dZ, dX2 = p["dX"], p["dZ"], p["dX2"]
    sw, bank = dX + 4 * dZ, 3 * dX
    strip = p["dp2"] if buffered else 0
    rect(ax, 0, 0, W, H, GREYU, GREYU_E, z=1)                       # enclosing rectangle (charged)
    y_core = bank + strip
    core_row(ax, 0, y_core, p, W)
    nper = p["n"] // 2
    p_lab = f"({dX},{dZ},{p['dm']})"
    for k in range(nper):
        xs = k * sw
        # upper bank: S1, S2 ; A at the lower-left corner (faces the upper ancilla row)
        yb = H - bank
        supplier(ax, xs, yb, sw, bank, f"S{k + 1}: 15-to-1", p_lab)
        rect(ax, xs, yb, dX, dX, TEAL, TEAL_E, z=4)
        txt(ax, xs + dX / 2, yb + dX / 2, "A", fs=6.5, color="white")
        # lower bank mirrored: S3, S4 ; A at the upper-left corner
        supplier(ax, xs, 0, sw, bank, f"S{k + 1 + nper}: 15-to-1", p_lab)
        rect(ax, xs, bank - dX, dX, dX, TEAL, TEAL_E, z=4)
        txt(ax, xs + dX / 2, bank - dX / 2, "A", fs=6.5, color="white")
        if buffered:
            dp = p["dp2"]
            rect(ax, xs, H - bank - dp, dp, dp, YEL, YEL_E, z=4)
            txt(ax, xs + dp / 2, H - bank - dp / 2, "B", fs=6.5)
            rect(ax, xs, bank, dp, dp, YEL, YEL_E, z=4)
            txt(ax, xs + dp / 2, bank + dp / 2, "B", fs=6.5)
    dim_arrows(ax, W, H, label_level2, qubits, title)


# Litinski, Quantum 3, 205 (2019), Fig. 17 (step 4), central level-2 region; coordinates in the
# units of that figure (extracted from the vector figure). The patch width d_X2 = 25 sites is 21.8
# units, hence U units per site.
U = 21.8 / 25
LIT = dict(
    patches=[(50.5, 390.6, 72.3, 412.4, "1"), (77.3, 390.6, 99.0, 412.4, "2"),
             (104.1, 390.6, 125.8, 412.4, "3"), (130.9, 390.6, 142.8, 412.4, "4")],
    out=[(50.5, 415.6, 72.3, 443.0), (77.3, 415.6, 99.1, 443.0)],
    ancilla=[(35.3, 365.6, 142.9, 387.3), (104.1, 415.6, 158.1, 437.3)],
    spots=[(20.1, 360.1, 32.1, 387.3, "m"), (20.1, 390.5, 47.3, 402.5, "m"), (20.1, 405.8, 47.3, 417.7, "0"),
           (146.1, 385.1, 173.3, 397.1, "0"), (146.1, 400.4, 173.3, 412.3, "m"), (161.3, 415.6, 173.3, 442.8, "m")],
    chan_left=[(16.8, 402.5), (16.8, 390.6), (16.9, 360.1), (4.9, 360.1), (4.9, 390.5), (1.9, 390.5), (1.9, 402.5), (4.9, 402.5)],
    chan_right=[(176.5, 400.4), (176.5, 412.3), (176.5, 442.8), (188.5, 442.8), (188.5, 412.3), (191.5, 412.3),
                (191.5, 400.4), (188.5, 400.4)],
    box=(1.4, 356.5, 191.9, 446.5),
)
# channel bands that lead from the level-1 blocks to the stubs at the box edges (y ranges of the stubs)
BAND_L = (390.5, 402.5)     # left stub, at x = 1.9
BAND_R = (400.4, 412.3)     # right stub, at x = 191.5


def draw_L(ax, p):
    """L after Litinski Figs. 13(a)/17: level-1 blocks left and right of the level-2 region, one horizontal
    transport channel per side between two blocks, |m> spots (4) and |0> patches (2), grey = unused."""
    dX, dZ = p["dX"], p["dZ"]
    bw, bh = (dX + 4 * dZ) * U, 3 * dX * U               # block size in figure units (35 x 45 sites)
    x_min, x_max = LIT["box"][0] + 1.9 - 1.9 - bw - 1.4 + 1.4, LIT["box"][2] + bw   # = box left - bw, box right + bw
    x_min = LIT["box"][0] - bw
    y_top = min(BAND_L[0] - bh, BAND_R[0] - bh)
    y_bot = max(BAND_L[1] + bh, BAND_R[1] + bh)
    X = lambda x: (x - x_min) / U
    Y = lambda y: (y_bot - y) / U
    Wd, Hd = (x_max - x_min) / U, (y_bot - y_top) / U
    ax.set_xlim(0, Wd)
    ax.set_ylim(0, Hd)
    ax.axis("off")
    rect(ax, 0, 0, Wd, Hd, GREYU, GREYU_E, z=1)                                   # bounding rectangle, not charged

    def R(x0, y0, x1, y1, fc, ec=None, z=3, ls="-", lw=0.7):
        ax.add_patch(Rectangle((X(x0), Y(y1)), (x1 - x0) / U, (y1 - y0) / U, fc=fc, ec=ec or fc, lw=lw, ls=ls, zorder=z))

    def C(x0, y0, x1, y1):
        return X((x0 + x1) / 2), Y((y0 + y1) / 2)
    # merge ancilla regions
    for a in LIT["ancilla"]:
        R(*a, PINK, PINK_E)
        ax.text(*C(*a), "merge ancilla", fontsize=6.0, color=PINK_E, ha="center", va="center", zorder=6)
    # core patches 1-4 and the two output patches
    for (*a, lab) in LIT["patches"]:
        R(*a, NAVY, "white", lw=0.8)
        ax.text(*C(*a), lab, fontsize=8.5, color="white", ha="center", va="center", zorder=6)
    for a in LIT["out"]:
        R(*a, NAVY, "white", lw=0.8)
        ax.text(*C(*a), "out", fontsize=7.0, color="white", ha="center", va="center", zorder=6)
    # transport channels (left/right stubs and the bands to the level-1 blocks)
    for poly in (LIT["chan_left"], LIT["chan_right"]):
        ax.add_patch(Polygon([(X(x), Y(y)) for x, y in poly], closed=True, fc=YEL_L, ec=YEL_E, lw=0.7, zorder=3))
    R(x_min + bw, BAND_L[0], LIT["chan_left"][5][0], BAND_L[1], YEL_L, YEL_E)
    R(LIT["chan_right"][5][0], BAND_R[0], LIT["box"][2] + bw - bw + bw, BAND_R[1], YEL_L, YEL_E)
    # |m> spots and |0> patches (dotted)
    for (*a, kind) in LIT["spots"]:
        R(*a, YEL, YEL_E, z=4, ls=(0, (1.5, 1.2)), lw=0.9)
        ax.text(*C(*a), r"$|m\rangle$" if kind == "m" else r"$|0\rangle$", fontsize=6.0, color=INK, ha="center",
                va="center", zorder=6)
    # level-1 blocks: S1,S2 left (above / below the channel), S3,S4 right
    lab = f"({dX},{dZ},{p['dm']})"
    blocks = [(x_min + 0, BAND_L[0] - bh, 1), (x_min + 0, BAND_L[1], 2),
              (LIT["box"][2], BAND_R[0] - bh, 3), (LIT["box"][2], BAND_R[1], 4)]
    for x, y, k in blocks:
        R(x, y, x + bw, y + bh, BEIGE, BEIGE_E, z=3)
        ax.text(*C(x, y, x + bw, y + bh), f"S{k}: 15-to-1\n{lab}", fontsize=5.6, color=INK, ha="center", va="center", zorder=6)
    # dimension labels
    ax.annotate("", xy=(Wd + 5, 0), xytext=(Wd + 5, Hd),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text(Wd + 11, Hd / 2, f"H \u2248 {Hd:.0f}", rotation=90, fontsize=6.5, color=INK, va="center", ha="center")
    ax.annotate("", xy=(0, -6), xytext=(Wd, -6),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text(Wd / 2, -14, f"W \u2248 {Wd:.0f}   (bounding rectangle, not charged)", fontsize=6.5, color=INK, ha="center", va="center")
    ax.text(Wd / 2, -24, "level 2: (25,13,17)", fontsize=6.3, color=INK, ha="center", va="center")
    ax.text(Wd / 2, Hd + 8, "38,480 qubits (sum of component areas, Eq. (3))", fontsize=7.4, color=INK, ha="center")
    ax.text(Wd / 2, Hd + 25, "(a) L: Litinski interface", fontsize=8.4, color=INK, ha="center")
    return Wd, Hd


# ------------------------------------------------------------------ figure assembly
FW, FH = 7.1, 6.75
L_BOTTOM = 4.55
S = 0.0145                                    # inches per data-qubit site
fig = plt.figure(figsize=(FW, FH))

# (a) L  (size taken from the layout itself)
bw_ = (L["dX"] + 4 * L["dZ"]) * U
WL = (LIT["box"][2] + bw_ - (LIT["box"][0] - bw_)) / U
HL = (max(BAND_L[1], BAND_R[1]) + 3 * L["dX"] * U - min(BAND_L[0], BAND_R[0]) + 3 * L["dX"] * U) / U
axL = fig.add_axes([0.15 / FW, L_BOTTOM / FH, WL * S / FW, HL * S / FH])
draw_L(axL, L)

# (b) D1 and (c) D2
axD1 = fig.add_axes([0.15 / FW, 0.9475 / FH, W1 * S / FW, H1 * S / FH])
draw_D(axD1, D1, W1, H1, False, "level 2: (25,15,13)", f"{tot_D1:,} qubits", "(b) D1: in-place supply")
axD2 = fig.add_axes([2.15 / FW, 0.69 / FH, W2 * S / FW, H2 * S / FH])
draw_D(axD2, D2, W2, H2, True, "level 2: (25,15,17)", f"{tot_D2:,} qubits", "(c) D2: local buffers")

# (d) stacked bars
axB = fig.add_axes([4.75 / FW, 1.0 / FH, 2.2 / FW, 4.3 / FH])
names = ["L", "D1", "D2"]
sup = [sup_L, sup_1, sup_2]
cor = [core_L, core_1, core_2]
itf = [int_L, int_1, int_2]
unu = [0, un_1, un_2]
tot = [tot_L, tot_D1, tot_D2]
b = [0, 0, 0]
for vals, col, ec in ((sup, BEIGE, BEIGE_E), (cor, NAVY, NAVY), (itf, YEL, YEL_E), (unu, GREYU, GREYU_E)):
    axB.bar(names, [v / 1000 for v in vals], bottom=[x / 1000 for x in b], color=col, edgecolor=ec, lw=0.6, width=0.62)
    b = [x + v for x, v in zip(b, vals)]
for i, t in enumerate(tot):
    axB.text(i, t / 1000 + 0.5, f"{t / 1000:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
axB.set_ylim(0, 44)
axB.set_ylabel("Physical qubits (thousands)", fontsize=7.5)
axB.tick_params(labelsize=7, length=2.5)
for s_ in ("top", "right"):
    axB.spines[s_].set_visible(False)
axB.set_title("(d) Footprint components", fontsize=8.4, color=INK, pad=16)

# legend
from matplotlib.patches import Patch
handles = [Patch(fc=BEIGE, ec=BEIGE_E, label="Level-1 suppliers"), Patch(fc=NAVY, ec=NAVY, label="Level-2 core"),
           Patch(fc=YEL, ec=YEL_E, label="Interface / buffers"), Patch(fc=GREYU, ec=GREYU_E, label="Unused rectangle")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.77, 0.0), ncol=2, frameon=False, fontsize=7,
           handlelength=1.6, columnspacing=1.2)

out = sys.argv[1]
fig.savefig(out)
print("saved", out, "L width (sites)", round(WL, 1))
