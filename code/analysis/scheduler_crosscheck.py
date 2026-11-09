"""Independent re-implementation of the reference dispatch policy, written from the rules stated in the manuscript
(Sec. III-D and Appendix B), compared with the scheduler used for all results (code/audit/timing.py, deterministic
mean-supply mode).  Also checks the stated rules on an event trace of the re-implementation.

Rules (manuscript):
  * a level-1 attempt lasts 6 d_m/(1-p_fail) on average (mean-supply mode);
  * each CCZ batch has four stages; per stage one supplied state per ancilla row (side); a pair of merges starts when
    both selected states and the core rows are available; each side takes the supplier whose state is ready first;
  * a merge lasts d_m2, decoding R, readout one cycle (X) or t_Y=floor(d_read/2)+2 (Y); in mean mode the readout is
    the average (1+t_Y)/2;  d_read = d_X (D1), d_p2 (D2); L reads |m> in X one cycle after its merge;
  * D1: the state stays in the supplier output slot until its readout is done;
  * D2: handoff h = max(d_X, d_p2) into the supplier's buffer once the buffer is free; the output slot is released at
    the end of the handoff; the buffer is released when the readout is done;
  * L: two |m> spots per side; the state departs when a spot is free, is transported for d_m2 cycles, and the merge
    also needs the spot's correction patch, which is busy until the previous decoding (R) plus one cycle is done;
    the |m> spot is cleared one cycle after its merge; the supplier output slot is released at departure;
  * a new attempt may start at most d_m cycles before the output slot is released;
  * the next batch starts after all readouts of the batch and one final syndrome-readout cycle.
Usage: python3 scheduler_crosscheck.py <path to code/audit>"""
import sys, itertools, math
sys.path.insert(0, sys.argv[1])
from timing import simulate

def reimpl(kind, dx, dm, dm2, dp, nb, pf, R=10.0, batches=128, trace=None):
    T1 = 6 * dm / (1 - pf)
    ty = (dp if kind == 2 else dx) // 2 + 2
    rd_mean = (1 + ty) / 2
    h = max(dx, dp)
    ready = [[T1] * nb for _ in range(2)]           # finish time of current attempt
    buffer_free = [[0.0] * nb for _ in range(2)]
    spot_free = [[0.0, 0.0] for _ in range(2)]; corr_free = [[0.0, 0.0] for _ in range(2)]
    core = 0.0; warm = batches // 4; t_warm = 0.0
    for b in range(batches):
        if b == warm: t_warm = core
        latest = 0.0
        for stage in range(4):
            pick = []
            for side in range(2):
                best = None
                s_idx = 0 if spot_free[side][0] <= spot_free[side][1] else 1
                for i in range(nb):
                    f = ready[side][i]
                    if kind == 0:
                        depart = max(f, spot_free[side][s_idx]); avail = max(depart + dm2, corr_free[side][s_idx])
                    elif kind == 2:
                        depart = max(f, buffer_free[side][i]); avail = depart + h
                    else:
                        depart = f; avail = f
                    if best is None or avail < best[0]: best = (avail, i, f, depart, s_idx)
                pick.append(best)
            start = max(core, pick[0][0], pick[1][0]); end = start + dm2
            for side in range(2):
                avail, i, f, depart, s_idx = pick[side]
                rd = 1.0 if kind == 0 else rd_mean
                done = end + R + rd; latest = max(latest, done)
                if kind == 0:
                    spot_free[side][s_idx] = end + 1; corr_free[side][s_idx] = done; release = depart
                elif kind == 2:
                    buffer_free[side][i] = done; release = depart + h
                else:
                    release = done
                restart = max(f, release - dm)
                if trace is not None: trace.append(dict(kind=kind, side=side, sup=i, f=f, depart=depart, start=start, end=end, done=done, release=release, restart=restart))
                ready[side][i] = restart + T1
            core = end
        core = max(core, latest) + 1.0
    return (core - t_warm) / (batches - warm)

grid = list(itertools.product((0, 1, 2), (13, 15), (5, 7), (13, 15, 17), (13, 15), (1, 2, 3, 4, 6), (0.02, 0.04)))
worst = 0.0
for kind, dx, dm, dm2, dp, nb, pf in grid:
    a = simulate(kind, dx, dm, dm2, dp, nb, pf)[0]; b = reimpl(kind, dx, dm, dm2, dp, nb, pf)
    worst = max(worst, abs(a - b) / a)
print(f"{len(grid)} configurations: max relative difference of the period = {worst:.2e}")

# rule checks on traces
viol = 0
for kind, nb in ((0, 2), (1, 2), (2, 2), (1, 3), (2, 3)):
    tr = []; reimpl(kind, 15, 7, 15, 15, nb, 0.04, trace=tr)
    for e in tr:
        if e["start"] < e["depart"] - 1e-9 and kind != 0: viol += 1                    # merge after the state is ready
        if e["restart"] < e["release"] - 7 - 1e-9: viol += 1                             # at most d_m before release
        if kind == 1 and e["release"] < e["done"] - 1e-9: viol += 1                      # D1 holds slot through readout
    by = {}
    for e in tr: by.setdefault((e["side"]), []).append(e)
    for side, ev in by.items():
        for a, b in zip(ev, ev[1:]):
            if b["start"] < a["end"] - 1e-9: viol += 1                                   # one merge at a time per row
print("rule violations in traces:", viol)
