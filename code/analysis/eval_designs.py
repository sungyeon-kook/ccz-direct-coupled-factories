import os,sys,math,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from model import rates,level1
from timing import simulate
from search import evaluate,KY,RR
p=1e-3;target=5.3e-11;ps,pt=rates(p)
D=[('minQ','L',0,2,(15,5,7,5),(25,13,17,15)),('minQ','D1',1,4,(15,7,7,5),(25,11,15,15)),('minQ','D2',2,4,(13,7,7,7),(25,15,17,15)),
   ('120','L',0,4,(15,5,7,5),(25,13,17,15)),('120','D1',1,4,(15,7,5,7),(25,15,13,13)),('120','D2',2,4,(13,7,7,7),(25,15,17,15)),
   ('minV','L',0,4,(15,7,5,7),(25,15,13,15)),('minV','D1',1,6,(15,5,7,7),(25,25,13,13)),('minV','D2',2,6,(15,5,7,7),(25,25,13,15))]
print('KY',KY,'R',RR)
for obj,name,kind,n,l1,l2 in D:
    dx,dz,dm,dp=l1;dx2,dz2,dm2,dp2=l2
    pl,pf=level1(p,dx,dz,dm,dp,ps,pt);c=np.array([dx,dz,dm,dp,pl,pf],float)
    times=np.array([simulate(kind,dx,dm,dm2,dp2,n//2,pf,R=RR,batches=4096,stochastic=True,seed=s) for s in range(1,9)])
    mean=times.mean(0);se=times.std(0,ddof=1)/math.sqrt(8)
    q,t,po=evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,mean);_,_,pohi=evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,mean+3*se)
    print(f"{obj:5s} {name:3s} Q={q:7.0f} T={t:7.2f} pout={po*1e11:5.2f}e-11 margin={pohi*1e11:5.2f} {'OK' if pohi<=target else 'FAIL'}")
