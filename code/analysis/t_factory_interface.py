"""Interface share of Litinski's published two-level (15-to-1)x(15-to-1) factories [Table 1 of arXiv:1905.06903] using
his footprint formula (twolevel15to1.py), and the geometric footprint of the same distances with direct coupling (D1
rectangle: two supplier banks on the two level-2 ancilla rows). Timing and output error are NOT re-evaluated."""
rows = [("1e-3", (11, 5, 5), (25, 11, 11), 6), ("1e-3", (13, 5, 5), (29, 11, 13), 6), ("1e-3", (17, 7, 7), (41, 17, 17), 6),
        ("1e-4", (9, 3, 3), (25, 9, 9), 4)]
for p, (dx, dz, dm), (dx2, dz2, dm2), n in rows:
    core = (dx2 + 4 * dz2) * 3 * dx2; l1 = n * ((dx + 4 * dz) * 3 * dx + 2 * dm)
    iface = n * (dx + 4 * dz) * dm2 / 2 + 20 * dm2 ** 2 + 2 * dx2 * dm2
    QL = 2 * int(core + l1 + iface)
    W = max(dx2 + 4 * dz2, (n // 2) * (dx + 4 * dz)); H = 6 * dx + 3 * dx2
    QD = 2 * W * H + 4 * n * dm
    print(f"p={p} n={n} ({dx},{dz},{dm})x({dx2},{dz2},{dm2}): L {QL}, interface {round(2*iface)} ({iface/(core+l1+iface):.1%}), D1 geometric {QD} ({QD/QL:.3f}), W={W}, H={H}")
