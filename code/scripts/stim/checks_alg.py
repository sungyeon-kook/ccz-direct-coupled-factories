"""Algebraic construction of merged-region checks:
X checks: full X plaquettes inside R, plus weight-2 X half-plaquettes on the far edges of patches only
          (D tops, O bottom) -> X strings may end only there.
Z checks: every Z plaquette truncated to R (weight >= 2) that commutes with all X checks."""
def alg_checks(R, xedge_rows, x3=False):
    xs = [x for x, _ in R]; ys = [y for _, y in R]
    X, Zc = {}, {}
    for cx in range(min(xs) - 1, max(xs) + 1):
        for cy in range(min(ys) - 1, max(ys) + 1):
            quad = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)]
            ins = [q in R for q in quad]; n = sum(ins)
            typ = "X" if (cx + cy) % 2 == 0 else "Z"
            sup = [q if i else None for q, i in zip(quad, ins)]
            if typ == "X":
                if n == 4 or (x3 and n == 3): X[(cx + .5, cy + .5)] = ("X", sup)
                elif n == 2:
                    pair = [q for q in quad if q in R]
                    if pair[0][1] == pair[1][1] and xedge_rows(pair): X[(cx + .5, cy + .5)] = ("X", sup)
            else:
                if n >= 2: Zc[(cx + .5, cy + .5)] = ("Z", sup)
    xs_sets = [frozenset(q for q in c[1] if q) for c in X.values()]
    Z = {}
    for k, c in Zc.items():
        zs = frozenset(q for q in c[1] if q)
        if all(len(zs & x) % 2 == 0 for x in xs_sets): Z[k] = c
    out = dict(X); out.update(Z)
    return out
