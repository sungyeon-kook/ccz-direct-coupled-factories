"""Re-evaluate retained mean-supply fronts with synchronized stochastic timing.

This is a validation/re-ranking of the retained set, not a global stochastic
optimization. Error estimates still use mean exposures in the logical model.
The +3 SEM exposure calculation is a numerical margin check, not a rigorous
confidence bound on the physical factory error probability.
"""
import csv,json,math
from pathlib import Path
import numpy as np
from model import rates,level1
from timing import simulate
from search import evaluate,OUT,RR

def run(key):
    domain=json.loads((OUT/f'domain_{key}.json').read_text());p=domain['p'];target=domain['target']
    ps,pt=rates(p,domain['S'],domain['T'])
    for kind,name in enumerate(['L','D1','D2']):
        rows=[]
        for r in csv.DictReader(open(OUT/f'front_{key}_{name}.csv')):
            r={k:float(v) for k,v in r.items()}
            dx,dz,dm,dp=[int(r[k]) for k in ['dx1','dz1','dm1','dp1']]
            dx2,dz2,dm2,dp2,n=[int(r[k]) for k in ['dx2','dz2','dm2','dp2','n']]
            pl,pf=level1(p,dx,dz,dm,dp,ps,pt);c=np.array([dx,dz,dm,dp,pl,pf],float)
            times=np.array([simulate(kind,dx,dm,dm2,dp2,n//2,pf,R=RR,batches=4096,stochastic=True,seed=seed) for seed in range(1,9)])
            mean=times.mean(axis=0);se=times.std(axis=0,ddof=1)/math.sqrt(len(times))
            q,t,po=evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,mean)
            _,thi,pohi=evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,mean+3*se)
            row=dict(r,screen_cycles=r['cycles'],screen_pout=r['pout'],cycles=t,pout=po,cycles_sem=se[0],pout_exposure_margin=pohi,cycles_exposure_margin=thi,valid=int(pohi<=target),source_hold=mean[1],buffer_hold=mean[2],transfer=mean[3])
            rows.append(row)
        with open(OUT/f'validated_{key}_{name}.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        good=[r for r in rows if r['valid']]
        front=[];best=float('inf')
        for r in sorted(good,key=lambda r:(r['qubits'],r['cycles'])):
            if r['cycles']<best:front.append(r);best=r['cycles']
        with open(OUT/f'validated_front_{key}_{name}.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(front)
        print(key,name,'valid',len(good),'/',len(rows),'minQ',min(good,key=lambda x:x['qubits']),'minV',min(good,key=lambda x:x['qubits']*x['cycles']),flush=True)

if __name__=='__main__':
    import sys
    for key in sys.argv[1:] or ['lit','q1_abs_d7','q5_abs_d7','p4']:run(key)
