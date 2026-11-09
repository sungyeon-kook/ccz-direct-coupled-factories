"""
Generalized Stim model of the v5 core rotation: P_v (x) Z_out measured by lattice surgery through a thin strip.
  * nD data patches (distance d2) in a row, D_j at x in [j(d2+g), j(d2+g)+d2), rows [0, d2)
  * `merged` = indices of data patches in P_v (the others stay idle as separate memories)
  * strip: rows [d2, d2+s), spans all data patches; O (distance d1) below the strip at x in [ox, ox+d1)
Observables (noiseless reference qubits R_j, R_O make every logical class separately visible):
  BAD (would corrupt the CCZ output, not equivalent to a faulty |T>):
     ZZ_j  = Z_Dj Z_Rj                         (X error on data patch j), all j
     XX_jk = X_Dj X_Rj X_Dk X_Rk (+strip)      consecutive merged j<k  (Z on a strict subset of P_v)
     X_j   = X_Dj X_Rj                         unmerged j              (Z on an idle patch)
  T-LIKE (equivalent to a faulty |T> for this rotation -> detected by the [[8,3,2]] syndrome):
     ZO    = Z_O Z_RO                          (X on out)
     XO    = X_Dlast X_Rlast X_O X_RO (+strip) (Z on out ~ Z on all of P_v)
     M     = recovered outcome of P_v (x) Z_out   (measurement-outcome fault)
"""
import stim, numpy as np
from merge_circuit import region_checks, support, strip_nullspace, gauge_basis, strip_completion

def gf2_solve(rows, target):
    """find x with sum_i x_i rows_i = target over GF(2); rows: list of 0/1 arrays"""
    A = np.array(rows, dtype=np.uint8).T                    # columns = generators
    b = np.array(target, dtype=np.uint8)
    M = np.concatenate([A, b[:, None]], axis=1); n = A.shape[1]; r = 0; piv = []
    for col in range(n):
        p = next((i for i in range(r, M.shape[0]) if M[i, col]), None)
        if p is None: continue
        M[[r, p]] = M[[p, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, col]: M[i] ^= M[r]
        piv.append(col); r += 1
    if any(M[i, n] for i in range(r, M.shape[0])): raise ValueError("no solution")
    x = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(piv): x[col] = M[i, n]
    return x

def build(d1, d2, s, nD=4, merged=(0, 1, 2, 3), g=2, ox=0, pre=1, r=None, post=1, p=1e-3, bot="Z", top_free="Z", a3=("Z",), S_override=None, use_alg=False, zorder_fn=None, D_override=None, strict_layers=False):
    r = r if r is not None else d1
    merged = tuple(sorted(merged))
    D = [{(x, y) for x in range(j * (d2 + g), j * (d2 + g) + d2) for y in range(d2)} for j in range(nD)]
    if D_override is not None:
        D = [set(q) for q in D_override]
        assert len(D) == nD
    width = nD * (d2 + g) - g
    S = {(x, y) for x in range(width) for y in range(d2, d2 + s)} if S_override is None else set(S_override)
    O = {(x, y) for x in range(ox, ox + d1) for y in range(d2 + s, d2 + s + d1)}
    assert S_override is not None or ox + d1 <= width
    sep = {}
    for P in D + [O]: sep.update(region_checks(P))
    Dm = set().union(*[D[j] for j in merged]); idle = [j for j in range(nD) if j not in merged]
    patches = Dm | O
    def htype(pair):
        if all(q in patches for q in pair): return "X"
        if all(q[1] == d2 + s - 1 for q in pair): return bot
        return top_free
    mreg = Dm | S | O
    if use_alg:
        from checks_alg import alg_checks
        far = lambda pair: all(q in Dm and q[1] == 0 for q in pair) or all(q in O and q[1] == d2 + s + d1 - 1 for q in pair)
        merged_checks = alg_checks(mreg, far)
    else:
        merged_checks = region_checks(mreg, htype=htype, allow3=a3)
    idle_checks = {}
    for j in idle: idle_checks.update(region_checks(D[j]))
    during = dict(idle_checks); during.update(merged_checks)
    allq = sorted(set().union(*D) | S | O, key=lambda q: (q[1], q[0]))
    anc_pos = sorted(set(sep) | set(merged_checks), key=lambda c: (c[1], c[0]))
    qid = {}
    for q in allq: qid[q] = len(qid)
    for a in anc_pos: qid[("a",) + a] = len(qid)
    Rq = {j: len(qid) + j for j in range(nD)}; RO = len(qid) + nD
    c = stim.Circuit()
    for q, i in qid.items():
        c.append("QUBIT_COORDS", [i], [q[1], q[2], 1] if q[0] == "a" else [q[0], q[1], 0])
    meas = []
    def rec(key): meas.append(key)
    def pt(pairs):
        t = []
        for k, (i, P) in enumerate(pairs):
            if k: t.append(stim.target_combiner())
            t.append(stim.target_x(i) if P == "X" else stim.target_z(i))
        return t
    def zrow(reg): y0 = min(y for _, y in reg); return {q for q in reg if q[1] == y0}
    def xcol(reg): x0 = min(x for x, _ in reg); return {q for q in reg if q[0] == x0}
    def L(reg, kind): return [(qid[q], kind) for q in sorted(zrow(reg) if kind == "Z" else xcol(reg))]
    regs = {("D", j): D[j] for j in range(nD)}; regs[("O",)] = O
    refs = {("D", j): Rq[j] for j in range(nD)}; refs[("O",)] = RO
    # init
    c.append("R", list(range(len(qid) + nD + 1)))
    for key, chk in sep.items():
        c.append("MPP", pt([(qid[q], chk[0]) for q in support(chk)])); rec(("init", key))
    for nm, reg in regs.items():
        for kind in "ZX":
            c.append("MPP", pt(L(reg, kind) + [(refs[nm], kind)])); rec(("initL", nm, kind))
    def do_round(checks, tag):
        anc = [qid[("a",) + k] for k in checks]
        datas = [qid[q] for q in allq if any(q in support(ch) for ch in checks.values())]
        c.append("R", anc); c.append("X_ERROR", anc, p); c.append("DEPOLARIZE1", datas, p)
        xa = [qid[("a",) + k] for k, ch in checks.items() if ch[0] == "X"]
        c.append("H", xa); c.append("DEPOLARIZE1", xa, p)
        for layer in range(4):
            pairs = []
            for k, (typ, quad) in checks.items():
                order = [0, 1, 2, 3] if typ == "X" else (zorder_fn(k, S, Dm) if zorder_fn else [0, 2, 1, 3])
                q = quad[order[layer]]
                if q is None: continue
                a = qid[("a",) + k]; pairs += [a, qid[q]] if typ == "X" else [qid[q], a]
            if strict_layers and len(pairs) != len(set(pairs)):
                raise ValueError("CNOT layer contains a shared qubit")
            if pairs: c.append("CX", pairs); c.append("DEPOLARIZE2", pairs, p)
        c.append("H", xa); c.append("DEPOLARIZE1", xa, p)
        c.append("X_ERROR", anc, p); c.append("M", anc)
        for k in checks: rec((tag, k))
        c.append("TICK")
    rounds = []
    for t in range(pre): do_round(sep, ("pre", t)); rounds.append((("pre", t), sep))
    strip = sorted(S, key=lambda q: (q[1], q[0]))
    c.append("RX", [qid[q] for q in strip]); c.append("Z_ERROR", [qid[q] for q in strip], p)
    for t in range(r): do_round(during, ("mrg", t)); rounds.append((("mrg", t), during))
    c.append("Z_ERROR", [qid[q] for q in strip], p); c.append("MX", [qid[q] for q in strip])
    for q in strip: rec(("stripX", q))
    for t in range(post): do_round(sep, ("post", t)); rounds.append((("post", t), sep))
    for key, chk in sep.items():
        c.append("MPP", pt([(qid[q], chk[0]) for q in support(chk)])); rec(("final", key))
    for nm, reg in regs.items():
        c.append("MPP", pt(L(reg, "Z") + [(refs[nm], "Z")])); rec(("finL", nm, "Z"))
        # Individual X correlations of merged patches are destroyed by the
        # parity measurement. Measuring them here also disturbs finRZ and
        # makes the merge-outcome observable artificially nondeterministic.
        # Retain only idle X correlations; use finXX for merged patches.
        if nm[0] == "D" and nm[1] in idle:
            c.append("MPP", pt(L(reg, "X") + [(refs[nm], "X")])); rec(("finL", nm, "X"))
    mrg_names = [("D", j) for j in merged]
    for a, b in zip(mrg_names, mrg_names[1:] + [("O",)]):
        c.append("MPP", pt(L(regs[a], "X") + [(refs[a], "X")] + L(regs[b], "X") + [(refs[b], "X")])); rec(("finXX", a, b))
    c.append("MPP", pt([(refs[nm], "Z") for nm in mrg_names + [("O",)]])); rec(("finRZ",))
    idx = {k: i for i, k in enumerate(meas)}; N = len(meas)
    def T(*keys): return [stim.target_rec(idx[k] - N) for k in keys]
    # detectors (same construction as merge_circuit.py, verified there)
    prev = {k: ("init", k) for k in sep}; prev_supp = {k: support(ch) for k, ch in sep.items()}
    for tag, checks in rounds:
        fm, fp = tag == ("mrg", 0), tag == ("post", 0)
        for key, chk in checks.items():
            cur = (tag, key); sp = support(chk)
            if key in prev and prev_supp.get(key) == sp:
                c.append("DETECTOR", T(cur, prev[key]), [key[0], key[1], 0])
            elif fm and chk[0] == "X":
                out = sp - S
                if not out: c.append("DETECTOR", T(cur), [key[0], key[1], 0])
                elif key in prev and prev_supp[key] == out: c.append("DETECTOR", T(cur, prev[key]), [key[0], key[1], 0])
            prev[key] = cur; prev_supp[key] = sp
        if fp:
            for key, chk in rounds[pre + r - 1][1].items():
                if chk[0] != "X": continue
                sp = support(chk); ins = sp & S
                if not ins: continue
                out = sp - S; tg = [(("mrg", r - 1), key)] + [("stripX", q) for q in ins]
                if not out: c.append("DETECTOR", T(*tg), [key[0], key[1], 1])
                elif key in sep and support(sep[key]) == out: c.append("DETECTOR", T(*(tg + [(("post", 0), key)])), [key[0], key[1], 1])
    for key in sep: c.append("DETECTOR", T(("final", key), prev[key]), [key[0], key[1], 2])
    zsupp = [support(ch) for ch in merged_checks.values() if ch[0] == "Z"]
    inner_x = [support(ch) for ch in merged_checks.values() if ch[0] == "X" and support(ch) <= S]
    for Tset in gauge_basis(strip_nullspace(zsupp, S), inner_x, S):
        c.append("DETECTOR", T(*[("stripX", q) for q in Tset]), [0, 0, 3])
    # observables
    obs, names = [], []
    def add(name, keys):
        c.append("OBSERVABLE_INCLUDE", T(*keys), len(names)); names.append(name)
    for j in range(nD): add(f"BAD:ZZ_D{j}", [("initL", ("D", j), "Z"), ("finL", ("D", j), "Z")])
    for j in idle: add(f"BAD:X_D{j}", [("initL", ("D", j), "X"), ("finL", ("D", j), "X")])
    for a, b in zip(mrg_names, mrg_names[1:] + [("O",)]):
        comp = strip_completion(xcol(regs[a]) | xcol(regs[b]), zsupp, S)
        kind = "TLIKE" if b == ("O",) else "BAD"
        add(f"{kind}:XX_{a}_{b}", [("initL", a, "X"), ("initL", b, "X"), ("finXX", a, b)] + [("stripX", q) for q in comp])
    add("TLIKE:ZZ_O", [("initL", ("O",), "Z"), ("finL", ("O",), "Z")])
    # measurement outcome M: product of round-0 merged Z checks + init Z checks equal to prod Z_rows
    allv = sorted(mreg); ix = {q: i for i, q in enumerate(allv)}
    def vec(ss):
        v = np.zeros(len(allv), dtype=np.uint8)
        for q in ss: v[ix[q]] ^= 1
        return v
    gens = [(("mrg", 0), k) for k, ch in merged_checks.items() if ch[0] == "Z"] + \
           [("init", k) for k, ch in sep.items() if ch[0] == "Z" and support(ch) <= mreg
            and all(len(support(ch) & support(xc)) % 2 == 0 for xc in merged_checks.values() if xc[0] == "X")]
    gvec = [vec(support(merged_checks[k] if t == ("mrg", 0) else sep[k])) for t, k in gens]
    target = vec(set().union(*[zrow(regs[nm]) for nm in mrg_names]) | zrow(O))
    xsol = gf2_solve(gvec, target)
    mkeys = [gens[i] for i in range(len(gens)) if xsol[i]]
    add("TLIKE:M", mkeys + [("finRZ",)] + [("initL", nm, "Z") for nm in mrg_names + [("O",)]])
    return c, names

def keep(c, ids):
    out = stim.Circuit()
    for inst in c.flattened():
        if inst.name == "OBSERVABLE_INCLUDE" and int(inst.gate_args_copy()[0]) not in ids: continue
        out.append(inst)
    return out

def distances(c, names):
    bad = [i for i, n in enumerate(names) if n.startswith("BAD")]
    tl = [i for i, n in enumerate(names) if n.startswith("TLIKE")]
    m = [i for i, n in enumerate(names) if n == "TLIKE:M"]
    g = lambda ids: len(keep(c, set(ids)).shortest_graphlike_error(ignore_ungraphlike_errors=True, canonicalize_circuit_errors=True))
    return g(bad), g(tl), g(m)
