"""Cancellation-resistant implementation of the supplied logical error model.

We propagate delta=rho-rho_ideal, not 1-fidelity of a rounded density matrix.
No approximation in the channel probabilities is introduced by this change.
Distances and rates are inputs; physical circuit correctness is not implied.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def rotation(a, e, mask, sign, p1, p2, p3):
    n = len(a)
    c = 2.0**-0.5
    for i in range(n):
        pi = sign * (1 if (binparity(i & mask) == 0) else -1)
        for j in range(n):
            pj = sign * (1 if (binparity(j & mask) == 0) else -1)
            if pi == pj:
                continue
            u = c + 1j*c*pi
            du = (-2*p1-p2-p3)*u + (p2-p3)*u.conjugate()
            e[i,j] = (u+du)*e[i,j] + du*a[i,j]
            a[i,j] *= u

@njit(cache=True)
def binparity(x):
    r=0
    while x:
        r ^= x & 1
        x >>= 1
    return r

@njit(cache=True)
def error(a,e,bit,p,is_x):
    if p == 0: return
    n=len(a)
    if is_x:
        for i in range(n):
            ii=i^bit
            for j in range(n):
                jj=j^bit
                if i*n+j < ii*n+jj:
                    u=e[i,j]; v=e[ii,jj]
                    shift=p*(a[ii,jj]-a[i,j])
                    e[i,j]=(1-p)*u+p*v+shift
                    e[ii,jj]=(1-p)*v+p*u-shift
    else:
        for i in range(n):
            for j in range(n):
                if (i^j)&bit:
                    e[i,j] -= 2*p*(a[i,j]+e[i,j])

@njit(cache=True)
def storage(a,e,probs,is_x):
    for k in range(len(probs)):
        error(a,e,1 << (len(probs)-1-k),probs[k],is_x)

@njit(cache=True)
def rot1(a,e,mask,length,dx,dm,p,ps,pt,dp):
    q=.5*ps[dp]*dm
    rotation(a,e,mask,1,p/3+q,p/3+q+.5*length*dx/dm*pt[dm],p/3)

@njit(cache=True)
def level1(p,dx,dz,dm,dp,ps,pt):
    a=np.full((32,32),1/32+0j);e=np.zeros((32,32),np.complex128)
    px=ps[dx];pz=ps[dz]
    s1=(p/3+.5*dp/dz*pz*dm,p/3+.5*dz/dp*ps[dp]*dm,p/3)
    for mask in (8,4,2):rotation(a,e,mask,1,*s1)
    rot1(a,e,14,3*dz,dx,dm,p,ps,pt,dp)
    sx=[.5*px*dm]+[.5*dz/dx*px*dm]*3
    sz=[.5*px*dm]+[.5*dx/dz*pz*dm]*3
    storage(a,e,[0.]+sx[1:]+[0.],True);storage(a,e,[0.]+sz[1:]+[0.],False)
    rot1(a,e,28,dx+2*dz,dx,dm,p,ps,pt,dp);rot1(a,e,26,dx+3*dz,dx,dm,p,ps,pt,dp)
    error(a,e,16,.5*(2*dx+5*dz)/dx*px*dm,False)
    storage(a,e,sx+[0.],True);storage(a,e,sz+[0.],False)
    rot1(a,e,22,dx+3*dz,dx,dm,p,ps,pt,dp);rot1(a,e,19,dx+4*dz,dx,dm,p,ps,pt,dp)
    rotation(a,e,1,1,*s1)
    error(a,e,16,.5*(2*dx+7*dz)/dx*px*dm,False)
    sx5=sx+[.5*dz/dx*px*dm];sz5=sz+[.5*dx/dz*pz*dm]
    storage(a,e,sx5,True);storage(a,e,sz5,False)
    rot1(a,e,25,dx+4*dz,dx,dm,p,ps,pt,dp);rot1(a,e,21,dx+4*dz,dx,dm,p,ps,pt,dp)
    error(a,e,16,(dx+4*dz)/dx*px*dm,False)
    storage(a,e,sx5,True);storage(a,e,sz5,False)
    rot1(a,e,31,dx+4*dz,dx,dm,p,ps,pt,dp);rot1(a,e,7,3*dz,dx,dm,p,ps,pt,dp)
    error(a,e,16,.5*(dx+4*dz)/dx*px*dm,False)
    storage(a,e,[.5*px*(dm+2*dx)]+sx5[1:],True)
    storage(a,e,[.5*px*(dm+2*dx)]+sz5[1:],False)
    rot1(a,e,11,4*dz,dx,dm,p,ps,pt,dp);rot1(a,e,13,4*dz,dx,dm,p,ps,pt,dp)
    storage(a,e,[0.]+sx5[1:],True);storage(a,e,[0.]+sz5[1:],False)
    da=0.+0j;df=0.+0j;w=np.exp(1j*np.pi/4)
    for i in range(32):
        for j in range(32):
            if i//16==j//16:da+=e[i,j]/16
            ui=1.+0j if i<16 else w
            uj=1.+0j if j<16 else w
            df+=ui.conjugate()*e[i,j]*uj/32
    return (da.real-df.real)/(1+da.real),-da.real

@njit(cache=True)
def level2(pl1,dx,dz,dm,ps,pt,inp,lanc,T):
    a=np.full((16,16),1/16+0j);e=np.zeros((16,16),np.complex128)
    px=ps[dx];pz=ps[dz]
    masks=(9,1,13,11,15,7,5,3);signs=(1,-1,-1,-1,1,-1,1,1)
    zz=.5*lanc/dx*px*dm
    for step in range(4):
        for k in (2*step,2*step+1):
            at=.5*lanc[k]*dx/dm*pt[dm]
            rotation(a,e,masks[k],signs[k],pl1+inp,inp+at,0.)
        if step==0:
            storage(a,e,[zz[0],0.,0.,0.],False)
            storage(a,e,[.5*px*T,0.,0.,.5*dz/dx*px*T],True)
            storage(a,e,[.5*px*T,0.,0.,.5*dx/dz*pz*T],False)
        elif step==1:
            storage(a,e,[zz[2]+zz[3],zz[2],zz[3],0.],False)
            storage(a,e,[.5*px*T]*3+[.5*dz/dx*px*T],True)
            storage(a,e,[.5*px*T]*3+[.5*dx/dz*pz*T],False)
        elif step==2:
            storage(a,e,[zz[4],zz[4]+zz[5],zz[4]+zz[5],0.],False)
            storage(a,e,[.5*px*(dm+2*dx),.5*px*T,.5*px*T,.5*dz/dx*px*T],True)
            storage(a,e,[.5*px*(dm+2*dx),.5*px*T,.5*px*T,.5*dx/dz*pz*T],False)
        else:
            storage(a,e,[0.,zz[6],zz[7],0.],False)
            storage(a,e,[0.,.5*px*(dm+2*dx),.5*px*(T+dm+2*dx),.5*dz/dx*px*T],True)
            storage(a,e,[0.,.5*px*(dm+2*dx),.5*px*(T+dm+2*dx),.5*dx/dz*pz*T],False)
    da=0.+0j;df=0.+0j
    for i in range(16):
        da+=.5*(e[i,i]+e[i,i^1])
        for j in range(16):
            si=1 if i<14 else -1;sj=1 if j<14 else -1
            df+=si*sj*e[i,j]/16
    return (da.real-df.real)/(1+da.real),-da.real

def rates(p,S=1.,T=1.):
    ds=np.arange(65)
    base=.1*(100*p)**((ds+1)/2)
    return base*S,base*T
