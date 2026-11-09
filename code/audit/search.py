"""Bounded unpruned design enumeration, explicit dispatch policy.

No cross-geometry Pareto pruning, first-dp break, or dz bisection.
All candidates within the documented box and explicit level-1 quality cap
are retained. 'Best found' never means a global optimum over all architectures.
"""
import sys,json,csv,time,os
from pathlib import Path
HH=float(os.environ.get('HOOKH','1'))  # merge-ancilla height in units of d_X2 (1 = reference, 2 = doubled)
KY=float(os.environ.get('KY','0'))  # in-place Y readout fault factor (0 = original model)
RR=float(os.environ.get('RREACT','10'))  # decoding reaction time R
TGT=float(os.environ.get('TARGET','0'))  # optional output-error target override
OUT=Path(os.environ.get('OUTDIR', str(Path(__file__).resolve().parents[1]/'corrected_results')))
from pathlib import Path
import numpy as np
from numba import njit
from model import level1,level2,rates
from timing import simulate


@njit(cache=False)
def candidates(p,ps,pt,lo,hi,target):
    out=[]
    for dx in range(lo,hi+1,2):
      for dz in range(3,dx+1,2):
       for dm in range(3,16,2):
        if dx>3*dm:continue
        for dp in range(3,min(dx,15)+1,2):
         po,pf=level1(p,dx,dz,dm,dp,ps,pt)
         # Explicit screening domain, not a claimed mathematical bound.
         if po<=1.5*np.sqrt(target/28) and pf<.25:
          out.append((float(dx),float(dz),float(dm),float(dp),po,pf))
    return np.array(out)

@njit(cache=False)
def evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,timing):
    dx,dz,dm,dp1=c[:4];pl,pf=c[4:]
    period,src,buf,transfer=timing
    W=max(3*dx2+dz2,n/2*(dx+4*dz))
    if kind==0:
        Q=2*((3*dx2+dz2)*(1+2*HH)*dx2+n*((dx+4*dz)*(3*dx+dp2/2)+2*dp1)+20*dp2**2+2*dx2*dp2)
        path=10*dp2+n/4*(dx+4*dz)
        # Original transport exposure plus source waiting and decoding residency.
        inp=.5*(path/dp2)*ps[dp2]*dm2+.5*src*ps[int(dx)]+.5*max(0.,buf-2*dm2)*ps[dp2]
        lens=np.array([3*dx2+dz2+dp2,dz2+dp2,3*dx2+dz2+dp2,3*dx2+dz2+dp2,3*dx2+dz2+dp2,2*dx2+dz2+dp2,3*dx2+dz2+dp2,dx2+dz2+dp2],np.float64)
    else:
        H=6*dx+(1+2*HH)*dx2+(2*dp2 if kind==2 else 0)
        Q=2*W*H+4*n*dp1
        inp=.5*src*ps[int(dx)]
        if kind==2:inp+=.5*buf*ps[dp2]+.5*transfer*ps[min(int(dx),dp2)]
        # In-place D1 output has spatial size dx, independent of merge rounds.
        lens=np.full(8,W+(dx if kind==1 else dp2))
    # In-place Y readout fault (D1/D2): a d_read x d_read patch for t_Y rounds, sensitive to X and Z
    # logical errors, scaled by KY for the excess error of the transition round (Gidney 2024: 1.5-2.5).
    # A wrong Y outcome is a wrong Pauli frame, i.e. a P_v error: added to the P channel, X/Y branch-averaged.
    if kind!=0 and KY>0:
        dr=int(dx) if kind==1 else dp2
        pl=pl+.5*KY*ps[dr]*(dr//2+2)
    po,pf2=level2(pl,dx2,dz2,dm2,ps,pt,inp,HH*lens,period/4)
    return Q,period/(1-pf2),po

@njit(cache=False)
def scan(kind,cands,ps,pt,target,dxlo,dxhi,dmlo,dmhi,dplo,dphi,qcap,tcap):
    # Best period for each even qubit count; retains a dense budget frontier.
    times=np.full(qcap//2+1,np.inf);pars=np.zeros((qcap//2+1,12));tested=0
    for ci in range(len(cands)):
      c=cands[ci];dx,dz,dm,dp1=c[:4]
      for n in (2,4,6,8,10,12):
       for dm2 in range(dmlo,dmhi+1,2):
        for dp2 in (range(dplo,dphi+1,2) if kind!=1 else range(dm2,dm2+1,2)):
         tm=simulate(kind,int(dx),int(dm),dm2,dp2,n//2,c[5],R=RR,batches=96)
         if tm[0]>tcap:continue
         for dx2 in range(dxlo,dxhi+1,2):
          for dz2 in range(5,dx2+1,2):
           # Geometry bound only; does not discard candidates by error or shape.
           W=max(3*dx2+dz2,n/2*(dx+4*dz))
           if kind==0:Q=2*((3*dx2+dz2)*(1+2*HH)*dx2+n*((dx+4*dz)*(3*dx+dp2/2)+2*dp1)+20*dp2**2+2*dx2*dp2)
           else:Q=2*W*(6*dx+(1+2*HH)*dx2+(2*dp2 if kind==2 else 0))+4*n*dp1
           if Q>qcap:continue
           # Even with zero rejection this candidate cannot improve this Q cell.
           ix=int(Q)//2
           if tm[0]>=times[ix]:continue
           q,t,po=evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,tm);tested+=1
           if po<=target and t<=tcap and t<times[ix]:
            times[ix]=t
            pars[ix]=np.array([n,dx,dz,dm,dp1,dx2,dz2,dm2,dp2,q,t,po])
    return times,pars,tested

def run(key,kinds=(0,1,2)):
    p=1e-3;target=5.3e-11
    if key=='lit':S,T=1.,1.
    elif key=='q5_abs_d7':S,T=2.8434,5.0148
    elif key=='q1_abs_d7':S,T=1.5937,.70466
    elif key=='p4':p=1e-4;target=7.3e-14;S=T=1.
    else:raise ValueError(key)
    if TGT>0:target=TGT
    ps,pt=rates(p,S,T);low=5 if p<.001 else 9;high=15 if p<.001 else 23
    c=candidates(p,ps,pt,low,high,target)
    np.save(OUT/f'candidates_{key}.npy',c)
    box=dict(p=p,target=target,S=S,T=T,KY=KY,R=RR,HOOKH=HH,dx1=[low,high],dz1='odd 3..dx1',dm1=[3,15],dp1='odd 3..min(dx1,15)',nl1=[2,4,6,8,10,12],quality_cap='pL1 <= 1.5 sqrt(target/28), rejection < 0.25',Qcap=(65000 if p==.001 else 26000)+(20000 if HH>1 else 0),Tcap=500,level1_candidates=len(c))
    bounds=(19,33,9,25,9,23) if p==.001 else (11,21,5,15,5,15)
    box['level2_ranges_dx_dm_dp']=bounds
    (OUT/f'domain_{key}.json').write_text(json.dumps(box,indent=2))
    print(key,'candidates',len(c),flush=True)
    for kind,name in enumerate(['L','D1','D2']):
        if kind not in kinds:continue
        start=time.time();ts,par,nt=scan(kind,c,ps,pt,target,*bounds,box['Qcap'],box['Tcap'])
        rows=[];best=float('inf')
        for i,t in enumerate(ts):
            if t<best:
                best=t;rows.append(dict(zip(['n','dx1','dz1','dm1','dp1','dx2','dz2','dm2','dp2','qubits','cycles','pout'],par[i])))
        with open(OUT/f'front_{key}_{name}.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print(key,name,'evaluated',nt,'secs',round(time.time()-start,1),'minQ',rows[0],'minV',min(rows,key=lambda x:x['qubits']*x['cycles']),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    args=sys.argv[1:]
    kinds=(0,1,2)
    if '--d1-only' in args:
        args.remove('--d1-only');kinds=(1,)
    for key in args or ['lit']:run(key,kinds)
