"""Smallest footprint under period caps and minimum-volume designs from validated fronts (validated_front_lit_*.csv)."""
import csv, sys
from pathlib import Path
R = Path(sys.argv[1])
F = {k: [r for r in csv.DictReader(open(R / f"validated_front_lit_{k}.csv")) if r["valid"] in ("1", "1.0")] for k in ("L", "D1", "D2")}
def best(k, cap):
    c = [r for r in F[k] if float(r["cycles"]) <= cap]
    return min(c, key=lambda r: (float(r["qubits"]), float(r["cycles"]))) if c else None
fmt = lambda r: f"{int(float(r['qubits'])):,} @ {float(r['cycles']):.2f} (n={int(float(r['n']))}, ({int(float(r['dx1']))},{int(float(r['dz1']))},{int(float(r['dm1']))});{int(float(r['dp1']))} / ({int(float(r['dx2']))},{int(float(r['dz2']))},{int(float(r['dm2']))});{int(float(r['dp2']))}, p={float(r['pout']):.2e})"
for cap in (80, 100, 120, 160, 200, 300, 1e9):
    b = {k: best(k, cap) for k in F}
    L = float(b["L"]["qubits"]) if b["L"] else None
    s = [f"cap {cap:g}:"] + [f"{k} " + (fmt(b[k]) + (f" saving {1-float(b[k]['qubits'])/L:+.1%}" if L and k != 'L' else "") if b[k] else f"{k} -") for k in F]
    print("\n   ".join(s))
v = {k: min(F[k], key=lambda r: float(r["qubits"]) * float(r["cycles"])) for k in F}
VL = float(v["L"]["qubits"]) * float(v["L"]["cycles"])
for k in F: print("minV", k, fmt(v[k]), f"vol {float(v[k]['qubits'])*float(v[k]['cycles'])/1e6:.3f}e6 ratio {float(v[k]['qubits'])*float(v[k]['cycles'])/VL:.3f}")
