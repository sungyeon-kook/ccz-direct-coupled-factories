"""Independent direct density-matrix calculation with 60-digit mpmath.
Uses the original level2 channel sequence, not the delta representation.
"""
import sys,json
from pathlib import Path
import mpmath as mp
import numpy as np
import model
mp.mp.dps=60
I=mp.eye(2);Z=mp.diag([1,-1]);X=mp.matrix([[0,1],[1,0]]);plus=mp.matrix([[.5,.5],[.5,.5]])
def kron(*items):
    r=mp.matrix([[1]])
    for b in items:
        r=mp.matrix([[r[i//b.rows,j//b.cols]*b[i%b.rows,j%b.cols] for j in range(r.cols*b.cols)] for i in range(r.rows*b.rows)])
    return r
def trace(a):return sum(a[i,i] for i in range(a.rows)).real
def level2(pl,dx,dz,dm,ps,pt,inp,lens,T):
    px=mp.mpf(float(ps[dx]));pz=mp.mpf(float(ps[dz]));pl=mp.mpf(float(pl));inp=mp.mpf(float(inp));T=mp.mpf(float(T))
    a=kron(plus,plus,plus,plus)
    def rot(mask,sign,p1,p2):
        nonlocal a
        P=[sign*(-1)**((i&mask).bit_count()) for i in range(16)]
        a=mp.matrix([[a[i,j]*sum(prob*mp.exp(1j*phi*(P[i]-P[j])) for prob,phi in [(1-p1-p2,mp.pi/8),(p1,5*mp.pi/8),(p2,-mp.pi/8)]) for j in range(16)] for i in range(16)])
    def store(probs,isx):
        nonlocal a
        for k,p in enumerate(probs):
            if not p:continue
            op=kron(*[(X if isx else Z) if l==k else I for l in range(4)])
            a=(1-p)*a+p*op*a*op
    zz=[mp.mpf(float(l)) /dx*px*dm/2 for l in lens]
    masks=[9,1,13,11,15,7,5,3];signs=[1,-1,-1,-1,1,-1,1,1]
    for st in range(4):
        for k in [2*st,2*st+1]:rot(masks[k],signs[k],pl+inp,inp+mp.mpf(float(lens[k]))*dx/dm*mp.mpf(float(pt[dm]))/2)
        if st==0:
            store([zz[0],0,0,0],False);sx=[px*T/2,0,0,mp.mpf(dz)/dx*px*T/2];sz=[px*T/2,0,0,mp.mpf(dx)/dz*pz*T/2]
        elif st==1:
            store([zz[2]+zz[3],zz[2],zz[3],0],False);sx=[px*T/2]*3+[mp.mpf(dz)/dx*px*T/2];sz=[px*T/2]*3+[mp.mpf(dx)/dz*pz*T/2]
        elif st==2:
            store([zz[4],zz[4]+zz[5],zz[4]+zz[5],0],False);sx=[px*(dm+2*dx)/2,px*T/2,px*T/2,mp.mpf(dz)/dx*px*T/2];sz=[px*(dm+2*dx)/2,px*T/2,px*T/2,mp.mpf(dx)/dz*pz*T/2]
        else:
            store([0,zz[6],zz[7],0],False);sx=[0,px*(dm+2*dx)/2,px*(T+dm+2*dx)/2,mp.mpf(dz)/dx*px*T/2];sz=[0,px*(dm+2*dx)/2,px*(T+dm+2*dx)/2,mp.mpf(dx)/dz*pz*T/2]
        store(sx,True);store(sz,False)
    proj=kron(I,I,I,(I+X)/2);acc=trace(proj*a)
    v=mp.matrix([mp.mpf(1)/4 if i<14 else -mp.mpf(1)/4 for i in range(16)])
    fid=(v.T*a*v)[0].real/acc
    return 1-fid,1-acc
if __name__=='__main__':
    rows=[]
    for p in [.001,.0001]:
        ps,pt=model.rates(p);l1=model.level1(p,13,7,7,7,ps,pt);lens=np.array([105,30,105,105,105,80,105,55],float)
        inp=.5*(150+1.5*41)*ps[15]
        precise=level2(l1[0],25,15,15,ps,pt,inp,lens,15.)
        fast=model.level2(l1[0],25,15,15,ps,pt,inp,lens,15.)
        row=dict(p=p,precise=str(precise[0]),delta=fast[0],absolute_difference=abs(float(precise[0])-fast[0]));rows.append(row);print(row,flush=True)
    (Path(__file__).resolve().parents[1]/'corrected_results/precision_check.json').write_text(json.dumps(rows,indent=2))
