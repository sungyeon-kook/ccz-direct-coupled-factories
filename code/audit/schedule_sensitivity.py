"""Factorial timing-policy sensitivity on the original retained candidate pool.

This is NOT a re-optimization of the distance domain, nor a physically validated
overlapped factory. The pool is front_*.csv, including candidates rejected in
the original stochastic validation. Batch-overlap cases do not charge extra
output storage. To avoid also granting an unsupported reduction in level-2
storage error, each candidate keeps at least its reference-policy storage
interval. Transfer latency/error and input-slot occupancy remain explicit.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from model import rates, level1
from timing import simulate
from search import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'corrected_results' / 'schedule_sensitivity'
POLICIES = [(label, barrier, release) for barrier, prefix in [(True, 'wait'), (False, 'overlap')]
            for release, label in [(0, prefix+'_mixed'), (1, prefix+'_departure'), (2, prefix+'_arrival')]]
PARAMS = ['n', 'dx1', 'dz1', 'dm1', 'dp1', 'dx2', 'dz2', 'dm2', 'dp2']

def load_rows(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

def write_rows(path, rows):
    with open(path, 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def ident(row):
    return tuple(int(row[k]) for k in PARAMS)

def measure(kind, r, pf, barrier, release):
    return np.array([simulate(kind, int(r['dx1']), int(r['dm1']), int(r['dm2']), int(r['dp2']),
                             int(r['n'])//2, pf, batches=4096, stochastic=True, seed=seed,
                             barrier=barrier, release_policy=release) for seed in range(1, 9)])

def score(kind, r, c, ps, pt, timing, reference_period):
    """Keep logical storage time at least as large as the reference policy.

    evaluate returns period/(1-p_fail2). Rescale that period, not the error
    exposure, after extracting the acceptance factor from its returned time.
    """
    exposure = timing.copy()
    exposure[0] = max(timing[0], reference_period)
    q, accepted_period, po = evaluate(kind, c, int(r['dx2']), int(r['dz2']), int(r['dm2']),
                                     int(r['dp2']), int(r['n']), ps, pt, exposure)
    return q, accepted_period * timing[0]/exposure[0], po, exposure[0]

def choose(rows, cap=None, volume=False):
    good = [r for r in rows if r['valid'] and (cap is None or r['cycles_exposure_margin'] <= cap)]
    if not good:
        return None
    return min(good, key=lambda r: (r['qubits'] * (r['cycles'] if volume else 1), r['cycles']))

def run(key):
    domain = json.loads((ROOT/'corrected_results'/f'domain_{key}.json').read_text())
    ps, pt = rates(domain['p'], domain['S'], domain['T'])
    all_rows = {label: {} for label, _, _ in POLICIES}
    for kind, name in enumerate(['L', 'D1', 'D2']):
        old = {ident(r): r for r in load_rows(ROOT/'corrected_results'/f'validated_{key}_{name}.csv')}
        results = {label: [] for label, _, _ in POLICIES}
        for r in load_rows(ROOT/'corrected_results'/f'front_{key}_{name}.csv'):
            pl, pf = level1(domain['p'], int(r['dx1']), int(r['dz1']), int(r['dm1']), int(r['dp1']), ps, pt)
            c = np.array([r['dx1'], r['dz1'], r['dm1'], r['dp1'], pl, pf])
            reference = measure(kind, r, pf, True, 0)
            refmean = reference.mean(axis=0)
            refse = reference.std(axis=0, ddof=1)/math.sqrt(len(reference))
            for label, barrier, release in POLICIES:
                samples = reference if label == 'wait_mixed' else measure(kind, r, pf, barrier, release)
                mean = samples.mean(axis=0)
                se = samples.std(axis=0, ddof=1)/math.sqrt(len(samples))
                q, t, po, exposure = score(kind, r, c, ps, pt, mean, refmean[0])
                _, thi, pohi, _ = score(kind, r, c, ps, pt, mean+3*se, refmean[0]+3*refse[0])
                row = dict(r, cycles=t, pout=po, cycles_sem=se[0], cycles_exposure_margin=thi,
                           pout_exposure_margin=pohi, valid=int(pohi<=domain['target']),
                           raw_batch_period=mean[0], storage_period=exposure,
                           source_hold=mean[1], buffer_hold=mean[2], transfer=mean[3],
                           barrier=int(barrier), release_policy=release)
                if label == 'wait_mixed':
                    original = old[ident(r)]
                    assert abs(t-original['cycles']) < 1e-8
                    assert abs(po-original['pout']) < 1e-20
                    assert row['valid'] == original['valid']
                results[label].append(row)
        for label, rows in results.items():
            write_rows(OUT/f'{key}_{label}_{name}.csv', rows)
            all_rows[label][name] = rows
        print(key, name, len(old), 'reference regressions passed', flush=True)
    summary = {}
    for label, _, _ in POLICIES:
        selected = {}
        for name in ['L', 'D1', 'D2']:
            rows = all_rows[label][name]
            selected[name] = {'min_volume': choose(rows, volume=True), 'min_qubits': choose(rows),
                              'caps': {str(cap): choose(rows, cap=cap) for cap in [80, 100, 120, 160, 200, 300]}}
        vl = selected['L']['min_volume']; v0 = vl['qubits']*vl['cycles']
        summary[label] = {'selected': selected,
                          'min_volume_ratios': {n: selected[n]['min_volume']['qubits']*selected[n]['min_volume']['cycles']/v0 for n in ['L', 'D1', 'D2']}}
        # Paired comparisons keep the reference best-volume designs fixed.
        fixed = {}
        for name in ['L', 'D1', 'D2']:
            base = choose(all_rows['wait_mixed'][name], volume=True)
            row = next(r for r in all_rows[label][name] if ident(r) == ident(base))
            fixed[name] = {'qubits': row['qubits'], 'cycles': row['cycles'], 'pout': row['pout'],
                           'valid': row['valid'], 'period_ratio_to_reference': row['cycles']/base['cycles']}
        summary[label]['fixed_reference_min_volume_designs'] = fixed
        print(key, label, 'minV ratios', summary[label]['min_volume_ratios'], flush=True)
    (OUT/f'summary_{key}.json').write_text(json.dumps(summary, indent=2))

if __name__ == '__main__':
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    for key in sys.argv[1:] or ['lit', 'q1_abs_d7', 'q5_abs_d7', 'p4']:
        run(key)
