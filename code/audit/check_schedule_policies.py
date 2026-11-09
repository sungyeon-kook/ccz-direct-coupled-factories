"""Analytic scheduling checks for counterfactual policies; no FT certification."""
import json
from pathlib import Path
import numpy as np
from timing import simulate

records = []
# With plentiful suppliers and no rejection, only merge depth and tail latency
# limit the core. This independently predicts the two limiting initiation times.
for kind in range(3):
    rd = 1 if kind == 0 else 2  # d_X=d_p=3: mean X/Y readout (1+3)/2
    for release in range(3):
        wait = simulate(kind, 3, 1, 15, 3, 16, 0., batches=256, release_policy=release)
        overlap = simulate(kind, 3, 1, 15, 3, 16, 0., batches=256, barrier=False, release_policy=release)
        # L's two spots each require transport+merge+one release cycle. Their
        # capacity is 2/(2*dm2+1) inputs per cycle, adding an internal stall.
        wait_expected = 4*15+10+rd+1+(1 if kind == 0 else 0)
        overlap_expected = 2*(2*15+1) if kind == 0 else 4*15+1
        assert abs(wait[0]-wait_expected) < 1e-10, wait
        assert abs(overlap[0]-overlap_expected) < 1e-10, overlap
records.append('18 abundant-supply limiting periods match analytic merge/tail and two-slot capacity formulas')

for kind in range(3):
    for nb in [1, 2, 3, 4, 6]:
        for barrier in [False, True]:
            for release in range(3):
                period, src, buf, transfer = simulate(kind, 15, 7, 15, 13, nb, .04,
                                                     batches=512, barrier=barrier, release_policy=release)
                assert period >= 4*15+1-1e-8
                assert period >= 4*42/(nb*.96)-1e-8
                assert min(src, buf, transfer) >= 0
                assert transfer == (15 if kind == 2 else 0)
records.append('90 policies satisfy merge-depth and mean supply-rate lower bounds; handoff exposure retained')
# D1 has no transport handoff and must be invariant to the release toggle.
for barrier in [False, True]:
    samples = [simulate(1, 15, 7, 15, 13, 3, .04, batches=1024,
                        stochastic=True, seed=23, barrier=barrier, release_policy=r) for r in range(3)]
    assert np.array_equal(samples[0], samples[1]) and np.array_equal(samples[0], samples[2])
records.append('D1 is invariant to the L/D2 source-release toggle')
path = Path(__file__).resolve().parents[1]/'corrected_results'/'schedule_sensitivity'
path.mkdir(exist_ok=True)
(path/'checks.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
