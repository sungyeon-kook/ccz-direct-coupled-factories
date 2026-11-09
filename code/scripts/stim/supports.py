"""Stabilizer-level check of every rotation support P_v (x) Z_out on the v6 geometry:
data patches (d2) sit inside a longer d2-tall ancilla row; the out tile (d1) can be anywhere below it.
For each support: (1) all checks commute, (2) M = prod Z_L(merged) Z_L(out) is in the merged stabilizer group,
(3) number of logical qubits of the merged code = #merged data patches (nothing else gets measured)."""
import numpy as np, itertools, sys
from merge_circuit import region_checks, support
from merge_circuit2 import gf2_solve
def rank(rows):
    if not rows: return 0
    M = np.array(rows, np.uint8); r = 0
    for col in range(M.shape[1]):
        p = next((i for i in range(r, M.shape[0]) if M[i, col]), None)
        if p is None: continue
        M[[r, p]] = M[[p, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, col]: M[i] ^= M[r]
        r += 1
    return r
def check(d1, d2, merged, ox, x0, g, Wtot, s=None, a3=("Z",)):
    s = s or d2
    D = [{(x, y) for x in range(x0 + j*(d2+g), x0 + j*(d2+g) + d2) for y in range(d2)} for j in range(4)]
    S = {(x, y) for x in range(Wtot) for y in range(d2, d2+s)}
    O = {(x, y) for x in range(ox, ox+d1) for y in range(d2+s, d2+s+d1)}
    Dm = set().union(*[D[j] for j in merged]); P = Dm | O
    ht = lambda pair: "X" if all(q in P for q in pair) else "Z"
    R = P | S
    ch = region_checks(R, ht, a3)
    allv = sorted(R); ix = {q: i for i, q in enumerate(allv)}
    vec = lambda ss: np.array([1 if q in ss else 0 for q in allv], np.uint8)
    Z = [vec(support(c)) for c in ch.values() if c[0] == "Z"]; X = [vec(support(c)) for c in ch.values() if c[0] == "X"]
    Zm = np.array(Z); Xm = np.array(X)
    comm = not ((Zm.astype(int) @ Xm.T.astype(int)) % 2).any()
    k = len(allv) - rank(Z) - rank(X)
    zr = lambda reg: {q for q in reg if q[1] == min(y for _, y in reg)}
    try: gf2_solve(Z, vec(set().union(*[zr(D[j]) for j in merged]) | zr(O))); inM = True
    except ValueError: inM = False
    return comm and inM and k == len(merged), (comm, k, inM)
if __name__ == "__main__":
    d1, d2 = int(sys.argv[1]), int(sys.argv[2])
    pos4 = sys.argv[3]             # 'left' or 'right' : where the syndrome patch 4 sits
    supports = [tuple(sorted(set(S) | {3 if pos4 == "right" else 0})) for S in
                [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]] if pos4 == "right" else \
               [tuple(sorted({0} | {j + 1 for j in S})) for S in [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]]
    for g in (1, 2, 3):
        for x0 in (0, 1, 2, 3):
            Wtot = x0 + 4*(d2+g) - g + 2*d1 + 1      # row longer than the data region on both... right side
            ok = tot = 0; fails = []
            for m in supports:
                for ox in range(0, Wtot - d1 + 1):
                    good, info = check(d1, d2, m, ox, x0, g, Wtot)
                    ok += good; tot += 1
                    if not good: fails.append((m, ox, info))
            print(f"pos4={pos4} g={g} x0={x0}: valid {ok}/{tot}" + (f"  e.g. {fails[:2]}" if fails else "  ALL VALID"), flush=True)
