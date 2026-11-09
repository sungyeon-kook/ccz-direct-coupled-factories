"""Space-like vs time-like logical failure rates of the rotated surface code under circuit-level noise in which the
measurement flip probability q = r * p is varied independently of the gate/idle/reset error p.

  memory    : d x d rotated patch, 3d rounds, Z basis           -> space-like failure per round  eps_s(d)
  stability : n x n patch whose boundaries are all Z type, so the product of all Z checks is the identity; data
              initialised and read out in X.  The observable is the parity of all Z-check outcomes of one round,
              which is exactly the quantity a lattice-surgery merge reports.  Its time-like distance is the number
              of rounds r_t (checked with Stim)                     -> time-like failure of an r_t-round merge P_t(r_t)

Noise (per round): reset X_ERROR(p); H DEPOLARIZE1(p); every CX layer DEPOLARIZE2(p) on pairs and DEPOLARIZE1(p) on
idle qubits; X_ERROR(q) before each measurement; DEPOLARIZE1(p) on data during measure/reset.
Decoder: PyMatching (correlated-free MWPM) via sinter.  Writes ../../tables/stim_timelike.csv."""
import stim, sinter, sys, csv, math
from pathlib import Path

def layout(n, kind):
    data = {(2 * i + 1, 2 * j + 1) for i in range(n) for j in range(n)}
    checks = []                                  # (pos, type, [data])
    for i in range(n + 1):
        for j in range(n + 1):
            pos = (2 * i, 2 * j); typ = "X" if (i + j) % 2 == 0 else "Z"
            ds = [(pos[0] + a, pos[1] + b) for a in (-1, 1) for b in (-1, 1) if (pos[0] + a, pos[1] + b) in data]
            if len(ds) == 4: checks.append((pos, typ, ds)); continue
            if len(ds) != 2: continue
            top_bot = j in (0, n); left_right = i in (0, n)
            if kind == "memory":
                if (top_bot and typ == "X") or (left_right and typ == "Z"): checks.append((pos, typ, ds))
            else:  # stability: Z-type boundaries everywhere
                if typ == "Z": checks.append((pos, typ, ds))
    return sorted(data), checks

ORDER = {"X": [(1, 1), (-1, 1), (1, -1), (-1, -1)], "Z": [(1, 1), (1, -1), (-1, 1), (-1, -1)]}

def circuit(kind, n, rounds, p, q):
    data, checks = layout(n, kind)
    idx = {}; coords = data + [c[0] for c in checks]
    for k, c in enumerate(coords): idx[c] = k
    D = [idx[d] for d in data]; M = [idx[c[0]] for c in checks]; MX = [idx[c[0]] for c in checks if c[1] == "X"]
    allq = D + M
    c = stim.Circuit()
    for k, xy in enumerate(coords): c.append("QUBIT_COORDS", [k], list(xy))
    data_basis = "Z" if kind == "memory" else "X"
    c.append("R" if data_basis == "Z" else "RX", D); c.append("Z_ERROR" if data_basis == "X" else "X_ERROR", D, p)
    nchk = len(checks); rec = []           # rec[r] = measurement-record offsets for round r
    det_type = "Z" if kind == "memory" else "X"   # check type deterministic in the first round
    for r in range(rounds):
        c.append("R", M); c.append("X_ERROR", M, p); c.append("DEPOLARIZE1", D, p)
        c.append("H", MX); c.append("DEPOLARIZE1", MX, p)
        for layer in range(4):
            pairs = []; busy = set()
            for pos, typ, ds in checks:
                a, b = ORDER[typ][layer]; d = (pos[0] + a, pos[1] + b)
                if d not in ds: continue
                m, dd = idx[pos], idx[d]
                assert m not in busy and dd not in busy; busy |= {m, dd}
                pairs += [m, dd] if typ == "X" else [dd, m]
            c.append("CX", pairs); c.append("DEPOLARIZE2", pairs, p)
            c.append("DEPOLARIZE1", [x for x in allq if x not in busy], p)
        c.append("H", MX); c.append("DEPOLARIZE1", MX, p)
        c.append("X_ERROR", M, q); c.append("M", M); c.append("DEPOLARIZE1", D, p)
        for k, (pos, typ, ds) in enumerate(checks):
            here = stim.target_rec(-nchk + k)
            if r == 0:
                if typ == det_type: c.append("DETECTOR", [here], [pos[0], pos[1], r])
            else:
                c.append("DETECTOR", [here, stim.target_rec(-2 * nchk + k)], [pos[0], pos[1], r])
        if kind == "stability" and r == 0:
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-nchk + k) for k, ch in enumerate(checks) if ch[1] == "Z"], 0)
    c.append("Z_ERROR" if data_basis == "X" else "X_ERROR", D, q); c.append("M" if data_basis == "Z" else "MX", D)
    nd = len(D); dpos = {d: k for k, d in enumerate(data)}
    for k, (pos, typ, ds) in enumerate(checks):
        if typ != det_type: continue
        tg = [stim.target_rec(-nd + dpos[d]) for d in ds] + [stim.target_rec(-nd - nchk + k)]
        c.append("DETECTOR", tg, [pos[0], pos[1], rounds])
    if kind == "memory":   # logical Z along the bottom row (commutes with all X checks)
        col = [d for d in data if d[1] == 1]
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-nd + dpos[d]) for d in col], 0)
    return c

def tasks(p, ratios, ds, ts, n_stab):
    out = []
    for r in ratios:
        for d in ds:
            out.append(sinter.Task(circuit=circuit("memory", d, 3 * d, p, r * p), json_metadata=dict(kind="memory", d=d, rounds=3 * d, r=r, p=p)))
        for t in ts:
            out.append(sinter.Task(circuit=circuit("stability", n_stab, t, p, r * p), json_metadata=dict(kind="stability", d=t, rounds=t, n=n_stab, r=r, p=p)))
    return out

if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        for kind, n, R in (("memory", 3, 3), ("memory", 5, 5), ("stability", 6, 3), ("stability", 6, 5), ("stability", 8, 7)):
            c = circuit(kind, n, R, 1e-3, 1e-3)
            dem = c.detector_error_model(decompose_errors=True)
            w = len(c.shortest_graphlike_error())
            print(kind, n, R, "qubits", c.num_qubits, "dets", c.num_detectors, "graphlike distance", w)
        sys.exit()
    p = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
    stats = sinter.collect(num_workers=2, tasks=tasks(p, (1, 2, 3, 5), (3, 5, 7, 9), (3, 5, 7, 9), 10),
                           decoders=["pymatching"], max_shots=3_000_000, max_errors=200, print_progress=False, save_resume_filepath=str(Path(__file__).resolve().parent / "timelike_stats.csv"))
    rows = []
    for s in stats:
        m = s.json_metadata; P = s.errors / s.shots
        per_round = 1 - (1 - P) ** (1 / m["rounds"]) if m["kind"] == "memory" else P
        rows.append(dict(kind=m["kind"], r=m["r"], d=m["d"], rounds=m["rounds"], shots=s.shots, errors=s.errors, P=f"{P:.4e}", rate=f"{per_round:.4e}"))
        print(rows[-1], flush=True)
    rows.sort(key=lambda x: (x["kind"], x["r"], x["d"]))
    out = Path(__file__).resolve().parents[2] / "tables" / "stim_timelike.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
