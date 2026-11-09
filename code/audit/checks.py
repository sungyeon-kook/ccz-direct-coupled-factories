"""Regression checks for the numerical model, dispatch constraints, and omissions.
Run after package construction. Does not certify a physical factory.
"""
import sys,json
from pathlib import Path
import numpy as np
import mpmath as mp
from model import rates,level1,level2
from timing import simulate
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy_reference'))
if not (ROOT/'legacy_reference').exists():sys.path.insert(0,str(ROOT/'litinski_interface'))
import fastmodel as old
mp.mp.dps=60
rng=np.random.default_rng(47);records=[]
for i in range(40):
    dx=int(rng.choice([11,13,15,17]));dz=int(rng.choice([5,7,9]));dm=int(rng.choice([5,7,9]));dp=int(rng.choice([5,7,9]))
    ps,pt=rates(.001,1.7,3.)
    a=level1(.001,dx,dz,dm,dp,ps,pt)
    b=old.level1(.001,dx,dz,dm,lambda d:ps[d],lambda d:pt[d],dp)
    assert max(abs(a[j]-b[j]) for j in [0,1])<5e-14,(a,b)
records.append('40 level-1 comparisons with the supplied density-matrix implementation passed')
zero=np.zeros(65)
assert abs(level1(0.,15,7,7,5,zero,zero)[0])<1e-25
for p in [1e-3,1e-6]:
    q=mp.mpf(p)
    acc=(1+(1-2*q)**8)/2
    correct=(1-q)**8+14*q**4*(1-q)**4+q**8
    exact=float(1-correct/acc)
    got=level2(p,25,15,15,zero,zero,0.,np.full(8,100.),15.)[0]
    assert abs(got-exact)<1e-13*max(p,1e-6),(got,exact)
records.append('8-input distillation matches its exact accepted-error polynomial')
for k in range(3):
    for n in [1,2,3,4,6]:
        period,src,buf,tr=simulate(k,15,7,15,13,n,.04)
        assert period>=8*42/(2*n*.96)-1e-7
        assert min(src,buf,tr)>=0
records.append('all 15 scheduling checks satisfy supply-rate and nonnegative-residency constraints')
for dp in [5,7]:
    ps,pt=rates(.001);records.append({'dp1':dp,'level1':level1(.001,15,5,7,dp,ps,pt)})
(ROOT/'corrected_results/checks.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
