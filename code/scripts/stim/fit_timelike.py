"""Fit the Stim memory/stability data (timelike_stats.csv, written by timelike_rates.py).

For each measurement-error ratio r = q/p:
  space-like : eps_s(d)  = per-round failure of a d x d memory           ~ exp(A_s + B_s (d+1)/2)
  time-like  : u_t(d)    = failure of a d-round stability experiment per (d x d) area, per d rounds, divided by d
                           = P_t(d) * d^2 / n^2 / d                        ~ exp(A_t + B_t (d+1)/2)
  (u_t is normalised exactly like Litinski's time-like term 0.5 * (area/dm^2) * pL(dm) * dm, so u_t == eps_s means
   'time-like errors are as likely as space-like ones', which is what pL(dm) == pL(d) assumes.)
Writes ../../tables/stim_fit.json with multipliers relative to r = 1 (used by litinski_interface/factory_opt.py),
and ../../tables/stim_timelike.csv with the raw rates and fits."""
import sinter, json, math, csv, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; TABLES = HERE.parents[1] / "tables"

stats = sinter.read_stats_from_csv_files(str(HERE / "timelike_stats.csv"))
agg = {}
for s in stats:
    m = s.json_metadata; k = (m["kind"], m["r"], m["d"])
    a = agg.setdefault(k, dict(shots=0, errors=0, rounds=m["rounds"], n=m.get("n")))
    a["shots"] += s.shots; a["errors"] += s.errors
rows = []; fits = {}
for r in sorted({k[1] for k in agg}):
    out = {}
    for kind in ("memory", "stability"):
        ds, ys, ws = [], [], []
        for (kd, rr, d), a in sorted(agg.items()):
            if kd != kind or rr != r or a["errors"] == 0: continue
            P = a["errors"] / a["shots"]
            rate = 1 - (1 - P) ** (1 / a["rounds"]) if kind == "memory" else P * d ** 2 / a["n"] ** 2 / d
            rel = 1 / math.sqrt(a["errors"])
            rows.append(dict(kind=kind, r=r, d=d, rounds=a["rounds"], shots=a["shots"], errors=a["errors"], rate=f"{rate:.4e}", rel_err=f"{rel:.3f}"))
            if d >= 5: ds.append((d + 1) / 2); ys.append(math.log(rate)); ws.append(1 / rel)
        B, A = np.polyfit(ds, ys, 1, w=ws)
        out["space" if kind == "memory" else "time"] = (float(A), float(B))
    fits[r] = out
    print(f"r={r}: space ln eps = {out['space'][0]:.2f} + {out['space'][1]:.3f} (d+1)/2  (Lambda_s = {math.exp(-out['space'][1]):.2f}); "
          f"time ln u = {out['time'][0]:.2f} + {out['time'][1]:.3f} (d+1)/2  (Lambda_t = {math.exp(-out['time'][1]):.2f})")
rel = {}
for r, f in fits.items():
    rel[f"r{r}"] = dict(space=[f["space"][0] - fits[1]["space"][0], f["space"][1] - fits[1]["space"][1]],
                        time=[f["time"][0] - fits[1]["time"][0], f["time"][1] - fits[1]["time"][1]],
                        abs=f)
json.dump(rel, open(TABLES / "stim_fit.json", "w"), indent=1)
with open(TABLES / "stim_timelike.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# extrapolated multipliers at relevant distances
for r, f in rel.items():
    S = lambda d: math.exp(f["space"][0] + f["space"][1] * (d + 1) / 2); T = lambda d: math.exp(f["time"][0] + f["time"][1] * (d + 1) / 2)
    print(r, " ".join(f"d={d}: S={S(d):.2g} T={T(d):.2g}" for d in (7, 15, 25)))
# Litinski's assumption at r = 1: time-like vs space-like per unit
A1s, B1s = fits[1]["space"]; A1t, B1t = fits[1]["time"]
print("r=1 ratio u_t/eps_s:", " ".join(f"d={d}: {math.exp(A1t - A1s + (B1t - B1s) * (d + 1) / 2):.2f}" for d in (5, 7, 9, 15, 25)))
