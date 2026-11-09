"""Finite-size one-side probes of the asymmetric direct-coupled merge motif.

Includes a rectangular syndrome patch and the merge observable M (previously
excluded by final_check.py). Not a full D1/D2 factory or proof of scaling.
Gaps and corner repairs are explicitly included in the physical geometry.
"""
import sys,json,itertools,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/stim'))
from merge_circuit2 import build,keep
from autorepair import repair

def run(dx,dz,do,dm):
    out=[];gap=2;D=[];x=0
    for w in [dx,dx,dx,dz]:
        D.append({(i,j) for i in range(x,x+w) for j in range(dx)});x+=w+gap
    W=x-gap+do+2
    supports=[tuple([j for j in range(3) if (m>>j)&1]+[3]) for m in range(8)]
    for mult in [1,2]:
      s=mult*dx
      for sup in supports:
       for ox in [0,W-do]:
        S0={(i,j) for i in range(W) for j in range(dx,dx+s)}
        O={(i,j) for i in range(ox,ox+do) for j in range(dx+s,dx+s+do)}
        ok,S,edits=repair([D[j] for j in sup],O,S0,{0},dx+s+do-1,len(sup))
        row=dict(dx=dx,dz=dz,dout=do,dm=dm,strip=s,W=W,support=sup,ox=ox,repaired=ok,corner_edits=edits)
        try:
            if not ok:raise ValueError('algebraic repair failed')
            c,names=build(do,dx,s,merged=sup,ox=ox,r=dm,S_override=S,use_alg=True,D_override=D,strict_layers=True)
            c.detector_error_model(decompose_errors=True)
            groups={'data':[i for i,n in enumerate(names) if n.startswith('BAD')],
                    'output':[i for i,n in enumerate(names) if n.startswith('TLIKE') and n!='TLIKE:M'],
                    'merge':[names.index('TLIKE:M')]}
            for name,ids in groups.items():
                row[name]=len(keep(c,set(ids)).shortest_graphlike_error(ignore_ungraphlike_errors=True,canonicalize_circuit_errors=True))
            row['num_qubits']=c.num_qubits
            if sup==(0,1,2,3) and ox==0:c.to_file(ROOT/f'corrected_results/probe_{dx}_{dz}_{do}_strip{s}.stim')
        except Exception as e:row['error']=str(e).split('\n')[0]
        out.append(row);print(row,flush=True)
    (ROOT/f'corrected_results/geometry_{dx}_{dz}_{do}_{dm}.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':
    args=list(map(int,sys.argv[1:])) or [5,3,3,5]
    run(*args)
