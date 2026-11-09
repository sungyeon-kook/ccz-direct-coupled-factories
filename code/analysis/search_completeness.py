"""Search-completeness check by random sampling.  Draw random designs from the documented search domain (level-1
candidates x n x level-2 distances), evaluate each exactly as the search does (screening timing, logical model), and
check that every feasible sample is dominated (fewer-or-equal qubits AND shorter-or-equal period) by a design on the
retained screening front front_lit_<kind>.csv.  A dominated sample means the search could not have missed it.
Usage: HOOKH=<1|2> python3 search_completeness.py <path to audit dir> <path to corrected_results dir> [samples]"""
import sys, os, csv
sys.path.insert(0, sys.argv[1])
import numpy as np
from model import rates
from timing import simulate
import search as S
R = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
p, target = 1e-3, 5.3e-11
ps, pt = rates(p)
cands = np.load(os.path.join(R, "candidates_lit.npy"))
qcap = 65000 + (20000 if S.HH > 1 else 0) if hasattr(S, "HH") else 65000
rng = np.random.default_rng(2026)
for kind, name in enumerate(["L", "D1", "D2"]):
    front = [(float(r["qubits"]), float(r["cycles"])) for r in csv.DictReader(open(os.path.join(R, f"front_lit_{name}.csv")))]
    feas = miss = 0
    for _ in range(N):
        c = cands[rng.integers(len(cands))]; n = int(rng.choice([2, 4, 6, 8, 10, 12]))
        dm2 = int(rng.choice(range(9, 26, 2))); dp2 = dm2 if kind == 1 else int(rng.choice(range(9, 24, 2)))
        dx2 = int(rng.choice(range(19, 34, 2))); dz2 = int(rng.choice(range(5, dx2 + 1, 2)))
        tm = simulate(kind, int(c[0]), int(c[2]), dm2, dp2, n // 2, c[5], batches=96)
        if tm[0] > 500: continue
        q, t, po = S.evaluate(kind, c, dx2, dz2, dm2, dp2, n, ps, pt, tm)
        if po > target or t > 500 or q > qcap: continue
        feas += 1
        if not any(fq <= q + 1e-9 and ft <= t + 1e-9 for fq, ft in front): miss += 1
    print(f"{name}: {N} random designs, {feas} feasible, {miss} not dominated by the retained front", flush=True)
