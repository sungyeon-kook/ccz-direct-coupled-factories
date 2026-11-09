import os,sys,math,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','audit'))
from model import rates,level1
from timing import simulate
from search import evaluate,KY,RR
p=1e-3;ps,pt=rates(p)
kind,n=int(sys.argv[1]),int(sys.argv[2]);dx,dz,dm,dp,dx2,dz2,dm2,dp2=map(int,sys.argv[3:11])
pl,pf=level1(p,dx,dz,dm,dp,ps,pt);c=np.array([dx,dz,dm,dp,pl,pf],float)
tm=simulate(kind,dx,dm,dm2,dp2,n//2,pf,R=RR,batches=96)
print('screen',evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,tm))
times=np.array([simulate(kind,dx,dm,dm2,dp2,n//2,pf,R=RR,batches=4096,stochastic=True,seed=s) for s in range(1,9)])
mean=times.mean(0);se=times.std(0,ddof=1)/math.sqrt(8)
print('KY',KY,'valid',evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,mean),evaluate(kind,c,dx2,dz2,dm2,dp2,n,ps,pt,mean+3*se)[2])
