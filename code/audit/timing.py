"""Explicit, deliberately conservative two-sided dispatch policy.

Both sides perform one rotation per stage; a CCZ batch has four stages.
The next batch waits for outstanding readouts and the syndrome decision.
For D2 the source slot is retained through the handoff (max(dx,dp) rounds).
The next level-1 attempt may start dm rounds before that slot is released.
L retains the published overlapped transport assumption. Output consumption
is assumed to meet the same pipeline contract as in Litinski's resource model.
This is a scheduling policy, not a proof of an optimal physical schedule.
Optional policies are counterfactual sensitivity cases: barrier=False overlaps
batch boundaries without charging extra output buffering. release_policy=1
releases both L/D2 at departure; =2 retains both through arrival; =0 is the
reference asymmetric policy. Handoff latency and exposure are never removed.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def finish(t,dm,pf,stochastic):
    if not stochastic:return t+6*dm/(1-pf)
    t+=6*dm
    while np.random.random()<pf:t+=6*dm
    return t

@njit(cache=True)
def simulate(kind,dx,dm,dm2,dp,nb,pf,R=10.,batches=128,stochastic=False,seed=1,barrier=True,release_policy=0):
    np.random.seed(seed)
    nxt=np.zeros((2,nb));buf=np.zeros((2,nb));spots=np.zeros((2,2));zeros=np.zeros((2,2))
    for s in range(2):
        for j in range(nb):nxt[s,j]=finish(0.,dm,pf,stochastic)
    now=0.;latest=0.;beg=0.;hs=0.;hb=0.;ht=0.;cnt=0
    warm=batches//4
    ty=(dp if kind==2 else dx)//2+2
    hand=max(dx,dp)
    for b in range(batches):
        if b==warm:beg=now
        for stage in range(4):
            ids=np.zeros(2,np.int64);ss=np.zeros(2,np.int64)
            fs=np.zeros(2);moves=np.zeros(2);arr=np.zeros(2)
            for side in range(2):
                best=1e100
                spot=0 if spots[side,0]<=spots[side,1] else 1
                for i in range(nb):
                    f=nxt[side,i]
                    if kind==0:
                        move=max(f,spots[side,spot]);avail=max(move+dm2,zeros[side,spot])
                    elif kind==2:
                        move=max(f,buf[side,i]);avail=move+hand
                    else:move=f;avail=f
                    if avail<best:
                        best=avail;ids[side]=i;fs[side]=f;moves[side]=move;arr[side]=avail;ss[side]=spot
            start=max(now,arr[0],arr[1]);end=start+dm2
            for side in range(2):
                i=ids[side];f=fs[side];move=moves[side]
                rd=1. if kind==0 else (1. if stochastic and np.random.random()<.5 else (float(ty) if stochastic else (1.+ty)/2))
                done=end+R+rd
                latest=max(latest,done)
                if kind==0:
                    spot=ss[side];spots[side,spot]=end+1;zeros[side,spot]=done
                    release=move+(dm2 if release_policy==2 else 0.)
                    restart=max(f,release-dm)
                    src=move-f;bufhold=done-move
                elif kind==2:
                    buf[side,i]=done
                    release=move+(0. if release_policy==1 else hand)
                    restart=max(f,release-dm)
                    src=move-f;bufhold=done-(move+hand)
                else:
                    restart=max(f,done-dm);src=done-f;bufhold=0.
                nxt[side,i]=finish(restart,dm,pf,stochastic)
                if b>=warm:
                    hs+=src;hb+=bufhold;ht+=(hand if kind==2 else 0);cnt+=1
            now=end
        # Includes decoding/readout barrier; one final X-syndrome readout cycle.
        now=(max(now,latest) if barrier else now)+1.
    return (now-beg)/(batches-warm),hs/cnt,hb/cnt,ht/cnt
