"""Re-evaluate output error and period of the Table-6 designs when the level-2 merge-ancilla rows are 2*d_X2 tall.
Uses the revision's logical model (code/audit/model.py, timing.py). A taller ancilla row doubles the merge area, so the
time-like outcome error (A p_t / 2 d_m) and the ancilla Z-storage term both double: equivalent to doubling every ancilla
length passed to level2. Timing is unchanged (same merge rounds). Validation protocol as validate_fronts.py.
Usage: python3 hook_height_reeval.py <path to code/audit>"""
import sys, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from model import rates, level1, level2
from timing import simulate
P, TARGET = 1e-3, 5.3e-11
ps, pt = rates(P)
def ev(kind, d1, dp1, d2, dp2, n, h, timing):
    dx, dz, dm = d1; dx2, dz2, dm2 = d2
    pl, pf = level1(P, dx, dz, dm, dp1, ps, pt)
    period, src, buf, transfer = timing
    W = max(3*dx2+dz2, n/2*(dx+4*dz))
    if kind == 0:
        core = (3*dx2+dz2)*(1+2*h)*dx2
        Q = 2*(core+n*((dx+4*dz)*(3*dx+dp2/2)+2*dp1)+20*dp2**2+2*dx2*dp2)
        path = 10*dp2+n/4*(dx+4*dz)
        inp = .5*(path/dp2)*ps[dp2]*dm2+.5*src*ps[int(dx)]+.5*max(0., buf-2*dm2)*ps[dp2]
        lens = np.array([3*dx2+dz2+dp2, dz2+dp2, 3*dx2+dz2+dp2, 3*dx2+dz2+dp2, 3*dx2+dz2+dp2, 2*dx2+dz2+dp2, 3*dx2+dz2+dp2, dx2+dz2+dp2], float)
    else:
        H = 6*dx+(1+2*h)*dx2+(2*dp2 if kind == 2 else 0)
        Q = 2*W*H+4*n*dp1
        inp = .5*src*ps[int(dx)]
        if kind == 2: inp += .5*buf*ps[dp2]+.5*transfer*ps[min(int(dx), dp2)]
        lens = np.full(8, W+(dx if kind == 1 else dp2))
    po, pf2 = level2(pl, dx2, dz2, dm2, ps, pt, inp, h*lens, period/4)
    return Q, period/(1-pf2), po
designs = [("120-cycle", "L", 0, (15,5,7), 5, (25,13,17), 15, 4), ("120-cycle", "D1", 1, (15,7,5), 7, (25,15,13), 13, 4),
           ("120-cycle", "D2", 2, (13,7,7), 7, (25,15,17), 15, 4), ("min footprint", "L", 0, (15,5,7), 5, (25,13,17), 15, 2),
           ("min footprint", "D1", 1, (15,7,7), 5, (25,11,15), 15, 4)]
for obj, name, kind, d1, dp1, d2, dp2, n in designs:
    pl, pf = level1(P, *d1, dp1, ps, pt)
    times = np.array([simulate(kind, d1[0], d1[2], d2[2], dp2, n//2, pf, batches=4096, stochastic=True, seed=s) for s in range(1, 9)])
    mean = times.mean(0); se = times.std(0, ddof=1)/math.sqrt(8)
    out = []
    for h in (1, 2):
        Q, T, po = ev(kind, d1, dp1, d2, dp2, n, h, mean); _, _, pohi = ev(kind, d1, dp1, d2, dp2, n, h, mean+3*se)
        out.append(f"h={h}: Q={Q:.0f} T={T:.2f} p_out={po:.3e} (+3SE {pohi:.3e}) {'OK' if pohi <= TARGET else 'FAIL'}")
    print(obj, name, " | ".join(out), flush=True)
