"""Absolute calibration and exact binomial confidence intervals from saved counts.
No fitted low-distance coefficient is represented as a high-distance validation.
"""
import csv,json,math
from pathlib import Path
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'tables/stim_timelike.csv')))
out=[]
for r in rows:
    k=int(r['errors']);n=int(r['shots']);d=int(r['d']);rounds=int(r['rounds'])
    P=k/n
    lo=0 if k==0 else beta.ppf(.025,k,n-k+1)
    hi=1 if k==n else beta.ppf(.975,k+1,n-k)
    # Binary logical flips can cancel, unlike a survival-probability process.
    def normalize(x):
        if r['kind']=='memory':return (1-(1-2*x)**(1/rounds))/2
        return x*d/100
    rate=normalize(P);low=normalize(lo);high=normalize(hi)
    baseline=.05*.1**((d+1)/2)
    out.append(dict(kind=r['kind'],q_over_p=int(r['r']),d=d,shots=n,errors=k,raw_probability=P,rate=rate,rate_ci95_low=low,rate_ci95_high=high,absolute_multiplier=rate/baseline,multiplier_ci95_low=low/baseline,multiplier_ci95_high=high/baseline,relative_count_se=math.sqrt((1-P)/k)))
dst=ROOT/'corrected_results/calibration_absolute.csv'
with open(dst,'w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(dst)
for r in out:
    if r['d'] in (7,9) and r['q_over_p'] in (1,5):print(r)
