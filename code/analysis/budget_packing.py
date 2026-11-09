"""Fixed qubit budget for the factory region: pack k identical copies of one retained design (k*qubits <= budget)
and maximise CCZ throughput k/period.  Uses the validated baseline fronts (p=1e-3, target 5.3e-11)."""
import csv, sys
from pathlib import Path
R = Path(sys.argv[1])
def load(k):
    return [(int(float(r["qubits"])), float(r["cycles"])) for r in csv.DictReader(open(R / f"validated_front_lit_{k}.csv")) if r["valid"] in ("1", "1.0", "True")]
F = {k: load(k) for k in ("L", "D1", "D2")}
def best(k, B):
    c = [(m / t, m, q, t) for q, t in F[k] for m in [B // q] if m >= 1]
    return max(c) if c else None
print("budget  " + "  ".join(f"{k:>24}" for k in F) + "   best direct / L")
rows = []
for B in list(range(30000, 72001, 2000)) + [80000, 90000, 100000, 120000, 150000]:
    b = {k: best(k, B) for k in F}
    s = "  ".join(f"{(f'{x[1]}x{x[2]}@{x[3]:.1f} -> {1000*x[0]:.2f}') if x else '-':>24}" for x in b.values())
    d = max([b[k][0] for k in ("D1", "D2") if b[k]] or [0]); l = b["L"][0] if b["L"] else 0
    ratio = (d / l) if l else float("inf")
    rows.append((B, ratio)); print(f"{B:>7}  {s}   {ratio:.2f}")
