"""
Stim model of the v4 core interface: a multi-patch Z-type lattice-surgery measurement
    M = Z_D1 Z_D2 Z_O
between two data patches D1, D2 (distance d2) and one d1 'out' patch O, through an ancilla strip.

Geometry (rotated surface code, data qubits on integer points, y grows downward):
    rows [0, d2)            : D1 at x in [0, d2), D2 at x in [d2+g, 2 d2+g)    (g = gap columns)
    rows [d2, d2+s)         : ancilla strip, x in [0, 2 d2+g)                 (present only during merge)
    rows [d2+s, d2+s+d1)    : O at x in [ox, ox+d1)
Boundaries: horizontal edges carry weight-2 X checks, vertical edges weight-2 Z checks
  -> Z_L = a full row (horizontal), X_L = a full column (vertical).
  Merging across horizontal edges measures the product of the Z_L's.

Time: noiseless MPP init (code states Bell-paired with noiseless reference qubits R1, R2, RO)
      -> `pre` noisy rounds -> `r` merge rounds -> `post` noisy rounds -> noiseless MPP readout.
Observables (reference qubits make each logical class separately visible):
  BAD : obs0 = Z_D1 Z_R1   (X error on D1)        obs1 = Z_D2 Z_R2   (X error on D2)
        obs2 = X_D1X_R1 X_D2X_R2   (Z_D1 alone or Z_D2 alone -- not equivalent to any faulty |T>)
  T-like (detected by the [[8,3,2]] syndrome, verified in verify_gadgets.py):
        obs3 = Z_O Z_RO (X on out)  obs4 = X_D2X_R2 X_OX_RO (Z_O ~ Z_D1 Z_D2)
"""
import stim, itertools

def region_checks(R, htype=lambda pair: "X", allow3=False):
    """weight-4 bulk + weight-2 boundary checks of a rotated code on data set R (set of (x,y)).
    htype(pair) gives the boundary type of a horizontal edge pair; vertical edges are Z-type."""
    checks = {}
    xs = [x for x, _ in R]; ys = [y for _, y in R]
    for cx in range(min(xs) - 1, max(xs) + 1):
        for cy in range(min(ys) - 1, max(ys) + 1):
            quad = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)]   # NW, NE, SW, SE
            inside = [q in R for q in quad]
            typ = "X" if (cx + cy) % 2 == 0 else "Z"
            n = sum(inside)
            if n == 4:
                checks[(cx + .5, cy + .5)] = (typ, quad)
            elif n == 3 and allow3 and typ in allow3:
                checks[(cx + .5, cy + .5)] = (typ, [q if i else None for q, i in zip(quad, inside)])
            elif n == 2:
                nw, ne, sw, se = inside
                horiz = (nw and ne) or (sw and se)          # pair lies on a horizontal edge
                vert = (nw and sw) or (ne and se)
                pair = [q for q, i in zip(quad, inside) if i]
                if horiz and typ == htype(pair):
                    checks[(cx + .5, cy + .5)] = (typ, [q if i else None for q, i in zip(quad, inside)])
                if vert and typ == "Z":
                    checks[(cx + .5, cy + .5)] = (typ, [q if i else None for q, i in zip(quad, inside)])
    return checks

def support(chk): return frozenset(q for q in chk[1] if q is not None)

def strip_nullspace(zchecks, strip):
    """basis of subsets T of the strip with even overlap with all Z checks (GF(2) null space)."""
    import numpy as np
    st = sorted(strip); n = len(st); ix = {q: i for i, q in enumerate(st)}
    A = np.array([[1 if q in sp else 0 for q in st] for sp in zchecks if sp & strip], dtype=np.uint8)
    r = 0; piv = []
    for col in range(n):
        p = next((i for i in range(r, A.shape[0]) if A[i, col]), None)
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(A.shape[0]):
            if i != r and A[i, col]: A[i] ^= A[r]
        piv.append(col); r += 1
    free = [j for j in range(n) if j not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i, col in enumerate(piv): v[col] = A[i, f]
        basis.append([st[i] for i in range(n) if v[i]])
    return basis

def gauge_basis(null, inner_x, strip):
    """vectors of `null` that are independent modulo the span of X checks lying inside the strip"""
    import numpy as np
    st = sorted(strip); ix = {q: i for i, q in enumerate(st)}
    def vec(sset):
        v = np.zeros(len(st), dtype=np.uint8)
        for q in sset: v[ix[q]] = 1
        return v
    basis = []
    rows = [vec(x) for x in inner_x]
    def rank(rs):
        if not rs: return 0
        M = np.array(rs, dtype=np.uint8); r = 0
        for col in range(M.shape[1]):
            p = next((i for i in range(r, M.shape[0]) if M[i, col]), None)
            if p is None: continue
            M[[r, p]] = M[[p, r]]
            for i in range(M.shape[0]):
                if i != r and M[i, col]: M[i] ^= M[r]
            r += 1
        return r
    cur = rank(rows)
    gens = []
    for T in null:
        v = vec(T)
        if rank(rows + [v]) > cur:
            rows.append(v); cur += 1; gens.append(v)
    # greedy weight reduction with local X checks and the other generators -> local gauge detectors
    local = [vec(x) for x in inner_x]
    changed = True
    while changed:
        changed = False
        for a in range(len(gens)):
            for w in local + [g for b, g in enumerate(gens) if b != a]:
                cand = gens[a] ^ w
                if cand.sum() < gens[a].sum():
                    gens[a] = cand; changed = True
    return [[st[i] for i in range(len(st)) if g[i]] for g in gens]

def strip_completion(base, zchecks, strip):
    """find T subset of strip such that base xor T has even overlap with every Z check (GF(2) solve)."""
    import numpy as np
    st = sorted(strip); ix = {q: i for i, q in enumerate(st)}
    rows, rhs = [], []
    for sp in zchecks:
        rows.append([1 if q in sp else 0 for q in st]); rhs.append(len(sp & base) % 2)
    A = np.array(rows, dtype=np.uint8); b = np.array(rhs, dtype=np.uint8)
    A = np.concatenate([A, b[:, None]], axis=1); n = len(st); r = 0; piv = []
    for col in range(n):
        p = next((i for i in range(r, A.shape[0]) if A[i, col]), None)
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(A.shape[0]):
            if i != r and A[i, col]: A[i] ^= A[r]
        piv.append(col); r += 1
    if any(A[i, n] for i in range(r, A.shape[0])): raise ValueError("no strip completion: operator is not a logical of the merged code")
    t = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(piv): t[col] = A[i, n]
    return [st[i] for i in range(n) if t[i]]

def build(d1, d2, s, g=1, ox=0, pre=1, r=None, post=1, p=1e-3, a3=("Z",), bot="Z"):
    r = r if r is not None else d2
    D1 = {(x, y) for x in range(d2) for y in range(d2)}
    D2 = {(x, y) for x in range(d2 + g, 2 * d2 + g) for y in range(d2)}
    S = {(x, y) for x in range(2 * d2 + g) for y in range(d2, d2 + s)}
    O = {(x, y) for x in range(ox, ox + d1) for y in range(d2 + s, d2 + s + d1)}
    sep = {}
    for P in (D1, D2, O): sep.update(region_checks(P))
    patches = D1 | D2 | O
    def htype(pair):
        if all(q in patches for q in pair): return "X"                 # far edges of D1, D2, O
        if all(q[1] == d2 + s - 1 for q in pair): return bot             # strip bottom edge (outside O)
        return "Z"                                                       # strip top edge in the gap
    merged = region_checks(D1 | D2 | S | O, htype=htype, allow3=a3)
    allq = sorted(D1 | D2 | S | O, key=lambda q: (q[1], q[0]))
    anc_pos = sorted(set(sep) | set(merged), key=lambda c: (c[1], c[0]))
    qid = {}
    for q in allq: qid[q] = len(qid)
    for a in anc_pos: qid[("a",) + a] = len(qid)
    R1, R2, RO = len(qid), len(qid) + 1, len(qid) + 2
    c = stim.Circuit()
    for q, i in qid.items():
        if q[0] == "a": c.append("QUBIT_COORDS", [i], [q[1], q[2], 1])
        else: c.append("QUBIT_COORDS", [i], [q[0], q[1], 0])
    meas = []                                   # list of keys, index = measurement record position
    def rec(key): meas.append(key)
    def pauli_target(pairs):                    # [(qubit index, 'X'/'Z'), ...] -> MPP targets
        t = []
        for k, (i, P) in enumerate(pairs):
            if k: t.append(stim.target_combiner())
            t.append(stim.target_x(i) if P == "X" else stim.target_z(i))
        return t
    def logical(P, region, kind):
        if kind == "Z":   # top row
            y0 = min(y for _, y in region); return [(qid[q], "Z") for q in sorted(region) if q[1] == y0]
        x0 = min(x for x, _ in region); return [(qid[q], "X") for q in sorted(region) if q[0] == x0]
    # ---------------- noiseless init: project D1, D2, O onto code states Bell-paired with refs
    c.append("R", list(range(len(qid) + 3)))
    for key, chk in sep.items():
        c.append("MPP", pauli_target([(qid[q], chk[0]) for q in support(chk)])); rec(("init", key))
    for name, reg, ref in (("D1", D1, R1), ("D2", D2, R2), ("O", O, RO)):
        for kind in "ZX":
            c.append("MPP", pauli_target(logical(None, reg, kind) + [(ref, kind)])); rec(("initL", name, kind))
    # ---------------- noisy rounds
    def do_round(checks, tag):
        anc = [qid[("a",) + k] for k in checks]
        datas = [qid[q] for q in allq if any(q in support(ch) for ch in checks.values())]
        c.append("R", anc); c.append("X_ERROR", anc, p)
        c.append("DEPOLARIZE1", datas, p)
        xa = [qid[("a",) + k] for k, ch in checks.items() if ch[0] == "X"]
        c.append("H", xa); c.append("DEPOLARIZE1", xa, p)
        orderX, orderZ = [0, 1, 2, 3], [0, 2, 1, 3]   # X: NW NE SW SE ; Z: NW SW NE SE
        for layer in range(4):
            pairs = []
            for k, (typ, quad) in checks.items():
                q = quad[(orderX if typ == "X" else orderZ)[layer]]
                if q is None: continue
                a = qid[("a",) + k]
                pairs += [a, qid[q]] if typ == "X" else [qid[q], a]
            if pairs:
                c.append("CX", pairs); c.append("DEPOLARIZE2", pairs, p)
        c.append("H", xa); c.append("DEPOLARIZE1", xa, p)
        c.append("X_ERROR", anc, p)
        c.append("M", anc)
        for k in checks: rec((tag, k))
        c.append("TICK")
    rounds = []
    for t in range(pre): do_round(sep, ("pre", t)); rounds.append((("pre", t), sep))
    strip = sorted(S, key=lambda q: (q[1], q[0]))
    c.append("RX", [qid[q] for q in strip]); c.append("Z_ERROR", [qid[q] for q in strip], p)
    for t in range(r): do_round(merged, ("mrg", t)); rounds.append((("mrg", t), merged))
    c.append("X_ERROR", [qid[q] for q in strip], p)     # (Z error before MX)
    c.append("Z_ERROR", [qid[q] for q in strip], p)
    c.append("MX", [qid[q] for q in strip])
    for q in strip: rec(("stripX", q))
    for t in range(post): do_round(sep, ("post", t)); rounds.append((("post", t), sep))
    for key, chk in sep.items():
        c.append("MPP", pauli_target([(qid[q], chk[0]) for q in support(chk)])); rec(("final", key))
    finals = {}
    for name, reg, ref in (("D1", D1, R1), ("D2", D2, R2), ("O", O, RO)):
        c.append("MPP", pauli_target(logical(None, reg, "Z") + [(ref, "Z")])); rec(("finL", name, "Z"))
    c.append("MPP", pauli_target(logical(None, D1, "X") + [(R1, "X")] + logical(None, D2, "X") + [(R2, "X")])); rec(("finXX", "D1D2"))
    c.append("MPP", pauli_target(logical(None, D2, "X") + [(R2, "X")] + logical(None, O, "X") + [(RO, "X")])); rec(("finXX", "D2O"))
    idx = {k: i for i, k in enumerate(meas)}
    N = len(meas)
    def T(*keys): return [stim.target_rec(idx[k] - N) for k in keys]
    # ---------------- detectors
    prev = {key: ("init", key) for key in sep}           # last measurement key of each check (by support)
    prev_supp = {key: support(chk) for key, chk in sep.items()}
    for (tag, checks) in rounds:
        is_merge = tag[0] == "mrg"
        first_merge = tag == ("mrg", 0)
        first_post = tag == ("post", 0)
        for key, chk in checks.items():
            cur = (tag, key); sp = support(chk)
            if key in prev and prev_supp.get(key) == sp:
                c.append("DETECTOR", T(cur, prev[key]), [key[0], key[1], 0])
            elif first_merge:
                outside = sp - S
                if chk[0] == "X":
                    if not outside: c.append("DETECTOR", T(cur), [key[0], key[1], 0])
                    elif key in prev and prev_supp[key] == outside:
                        c.append("DETECTOR", T(cur, prev[key]), [key[0], key[1], 0])
            elif first_post:
                pass
            prev[key] = cur; prev_supp[key] = sp
        if first_post:
            # previous merged X checks touching the strip: value = strip MX product * restored boundary check
            for key, chk in rounds[pre + r - 1][1].items():
                if chk[0] != "X": continue
                sp = support(chk); ins = sp & S
                if not ins: continue
                outside = sp - S
                keys = [("mrg", r - 1), ]
                tg = [(("mrg", r - 1), key)] + [("stripX", q) for q in ins]
                if not outside: c.append("DETECTOR", T(*tg), [key[0], key[1], 1])
                elif key in sep and support(sep[key]) == outside:
                    c.append("DETECTOR", T(*(tg + [(("post", 0), key)])), [key[0], key[1], 1])
    for key in sep:
        c.append("DETECTOR", T(("final", key), prev[key]), [key[0], key[1], 2])
    # gauge detectors: X-type operators on the strip that commute with every merged Z check are +1 at
    # initialization (|+> strip) and read out by the strip MX at the split (lattice-surgery gauge fixing).
    inner_x = [support(ch) for ch in merged.values() if ch[0] == "X" and support(ch) <= S]
    for Tset in gauge_basis(strip_nullspace([support(ch) for ch in merged.values() if ch[0] == "Z"], S), inner_x, S):
        c.append("DETECTOR", T(*[("stripX", q) for q in Tset]), [0, 0, 3])
    # ---------------- observables
    c.append("OBSERVABLE_INCLUDE", T(("initL", "D1", "Z"), ("finL", "D1", "Z")), 0)
    c.append("OBSERVABLE_INCLUDE", T(("initL", "D2", "Z"), ("finL", "D2", "Z")), 1)
    zsupp = [support(ch) for ch in merged.values() if ch[0] == "Z"]
    colrep = lambda reg: {q for q in reg if q[0] == min(x for x, _ in reg)}
    T12 = strip_completion(colrep(D1) | colrep(D2), zsupp, S)
    T2O = strip_completion(colrep(D2) | colrep(O), zsupp, S)
    c.append("OBSERVABLE_INCLUDE", T(("initL", "D1", "X"), ("initL", "D2", "X"), ("finXX", "D1D2"), *[("stripX", q) for q in T12]), 2)
    c.append("OBSERVABLE_INCLUDE", T(("initL", "O", "Z"), ("finL", "O", "Z")), 3)
    c.append("OBSERVABLE_INCLUDE", T(("initL", "D2", "X"), ("initL", "O", "X"), ("finXX", "D2O"), *[("stripX", q) for q in T2O]), 4)
    return c

def keep_observables(c, keep):
    out = stim.Circuit()
    for inst in c.flattened():
        if inst.name == "OBSERVABLE_INCLUDE" and int(inst.gate_args_copy()[0]) not in keep: continue
        out.append(inst)
    return out
