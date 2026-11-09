"""Per-rotation ancilla shape repair: greedily toggle single strip qubits at corners until the merged code
measures exactly M (k = #merged data patches and M in the stabilizer group)."""
import numpy as np, itertools, sys
from checks_alg import alg_checks
from merge_circuit import support
from merge_circuit2 import gf2_solve
from supports import rank
def evaluate(Dm_list, O, S, d_top_rows, o_bot):
    D = set().union(*Dm_list); R = D | O | S
    far = lambda pair: all(q in D and q[1] in d_top_rows for q in pair) or all(q in O and q[1] == o_bot for q in pair)
    ch = alg_checks(R, far)
    allv = sorted(R); vec = lambda ss: np.array([1 if q in ss else 0 for q in allv], np.uint8)
    Z = [vec(support(c)) for c in ch.values() if c[0] == "Z"]; X = [vec(support(c)) for c in ch.values() if c[0] == "X"]
    k = len(allv) - rank(Z) - rank(X)
    zr = lambda reg: {q for q in reg if q[1] == min(y for _, y in reg)}
    try: gf2_solve(Z, vec(set().union(*[zr(d) for d in Dm_list]) | zr(O))); inM = True
    except ValueError: inM = False
    return k, inM
def corner_candidates(P, S):
    R = P | S; cand = set()
    for (x, y) in S:
        nb = sum((x + dx, y + dy) in R for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)))
        if nb <= 2: cand.add((x, y))                    # convex corner of the strip
    for (x, y) in S:                                     # missing qubits next to concave corners
        for dx, dy in itertools.product((-1, 0, 1), repeat=2):
            q = (x + dx, y + dy)
            if q not in R and abs(dx) + abs(dy) == 2:
                if (x + dx, y) in R and (x, y + dy) in R: cand.add(q)
    return cand
def repair(Dm_list, O, S, d_top_rows, o_bot, target_k, max_steps=12):
    P = set().union(*Dm_list) | O
    k, inM = evaluate(Dm_list, O, S, d_top_rows, o_bot)
    score = lambda k, m: abs(k - target_k) + (0 if m else 1)
    cur = score(k, inM); S = set(S); steps = 0
    while cur > 0 and steps < max_steps:
        best = None
        for q in corner_candidates(P, S):
            S2 = S ^ {q}
            k2, m2 = evaluate(Dm_list, O, S2, d_top_rows, o_bot)
            sc = score(k2, m2)
            if best is None or sc < best[0]: best = (sc, q)
        if best is None or best[0] >= cur: break
        S ^= {best[1]}; cur = best[0]; steps += 1
    return cur == 0, S, steps
if __name__ == "__main__":
    d1, d2 = int(sys.argv[1]), int(sys.argv[2]); g = 2; x0 = 0
    Wtot = x0 + 4*(d2+g) - g + d1 + 2
    D = [{(x, y) for x in range(x0 + j*(d2+g), x0 + j*(d2+g) + d2) for y in range(d2)} for j in range(4)]
    sups = [tuple(sorted({0} | {j + 1 for j in S})) for S in [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]]
    total = okc = 0; maxsteps = 0
    for m in sups:
        for ox in range(0, Wtot - d1 + 1, max(1, (Wtot - d1) // 6)):
            S0 = {(x, y) for x in range(Wtot) for y in range(d2, 2*d2)}
            O = {(x, y) for x in range(ox, ox+d1) for y in range(2*d2, 2*d2+d1)}
            ok, S, steps = repair([D[j] for j in m], O, S0, {0}, 2*d2+d1-1, len(m))
            total += 1; okc += ok; maxsteps = max(maxsteps, steps)
            if not ok: print("unrepaired:", m, ox)
    print(f"(d1,d2)=({d1},{d2}): repaired {okc}/{total}, max single-qubit edits {maxsteps}")
