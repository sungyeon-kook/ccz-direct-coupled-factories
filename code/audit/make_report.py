"""Generate all manuscript numbers from validated CSVs; no hand-edited results."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'corrected_results';DEST=ROOT.parent
(DEST/'generated').mkdir(exist_ok=True);(DEST/'figs').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def read(key,name):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(open(OUT/f'validated_front_{key}_{name}.csv'))]
data={key:{name:read(key,name) for name in ['L','D1','D2']} for key in ['lit','q1_abs_d7','q5_abs_d7','p4']}
def best(key,name,budget=None,volume=False):
    rs=data[key][name]
    if budget is not None:rs=[r for r in rs if r['cycles_exposure_margin']<=budget]
    if not rs:return None
    return min(rs,key=lambda r:(r['qubits']*(r['cycles'] if volume else 1),r['cycles']))
def comma(q):return f'{q:,.0f}'.replace(',','{,}')
def d(r,j):return '('+','.join(str(int(r[k+str(j)])) for k in ['dx','dz','dm'])+')'
lines=[];summary={}
for label,budget,vol in [('footprint',None,False),('120-cycle cap',120,False),('volume',None,True)]:
 for name in ['L','D1','D2']:
    r=best('lit',name,budget,vol);base=best('lit','L',budget,vol)
    lines.append(f"{label} & {name} & {int(r['n'])} & ${d(r,1)};{int(r['dp1'])}$ & ${d(r,2)};{int(r['dp2']) if name!='D1' else '-'}$ & {comma(r['qubits'])} & {r['cycles']:.2f} & ${r['pout']/1e-11:.2f}$ & {r['qubits']/base['qubits']:.3f} & {r['qubits']*r['cycles']/(base['qubits']*base['cycles']):.3f} \\")
    summary[f'lit_{label}_{name}']=r
(DEST/'generated/main_rows.tex').write_text('\n'.join(x.rstrip('\\')+'\\\\' for x in lines))
lines=[]
for cap in [80,100,120,160,200,300]:
 vals=[best('lit',n,cap) for n in ['L','D1','D2']]
 lines.append(str(cap)+' & '+' & '.join(comma(r['qubits']) if r else '--' for r in vals)+' & '+' & '.join(f"{100*(1-r['qubits']/vals[0]['qubits']):.1f}\\%" if r else '--' for r in vals[1:])+r' \\')
(DEST/'generated/budget_rows.tex').write_text('\n'.join(lines))
lines=[]
for key,label in [('lit','Baseline'),('q1_abs_d7',r'$q=p$ anchor'),('q5_abs_d7',r'$q=5p$ anchor'),('p4',r'$p=10^{-4}$')]:
    L=best(key,'L',120);D1=best(key,'D1',120);D2=best(key,'D2',120)
    vL=best(key,'L',volume=True);v1=best(key,'D1',volume=True);v2=best(key,'D2',volume=True)
    lines.append(label+' & '+' & '.join(comma(r['qubits']) for r in [L,D1,D2])+f" & {v1['qubits']*v1['cycles']/(vL['qubits']*vL['cycles']):.3f} & {v2['qubits']*v2['cycles']/(vL['qubits']*vL['cycles']):.3f}"+r' \\')
(DEST/'generated/scenario_rows.tex').write_text('\n'.join(lines))
policy=json.loads((OUT/'schedule_sensitivity/summary_lit.json').read_text())
lines=[]
for key,r in policy.items():
    barrier,release=key.split('_')
    vl=r['selected']['L']['min_volume']
    lines.append(('wait' if barrier=='wait' else 'overlap')+' & '+release+
                 f" & {vl['qubits']*vl['cycles']/1e6:.3f} & {r['min_volume_ratios']['D1']:.3f} & {r['min_volume_ratios']['D2']:.3f}"+r' \\')
(DEST/'generated/policy_rows.tex').write_text('\n'.join(lines))
(OUT/'report_summary.json').write_text(json.dumps(summary,indent=2))
colors={'L':'#243e60','D1':'#ad5134','D2':'#15847a'}
fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
for ax,key,title in zip(axes,['lit','q5_abs_d7'],['Baseline logical model','Absolute-rate anchor ($q=5p$, $d=7$)']):
 for name in ['L','D1','D2']:
    rs=data[key][name];ax.step([r['qubits']/1000 for r in rs],[r['cycles'] for r in rs],where='post',color=colors[name],label=name,lw=1.5)
 ax.set(xlabel='Physical qubits (thousands)',title=title,xlim=(27,60),ylim=(55,210));ax.grid(alpha=.18);ax.axhline(120,color='#777',ls=':',lw=.8)
axes[0].set_ylabel('Cycles per accepted CCZ');axes[0].legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(DEST/'figs/revised_fronts.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.4,2.35))
caps=np.linspace(60,220,321)
for name in ['L','D1','D2']:
    ys=[best('lit',name,cap) for cap in caps];ax.plot(caps,[r['qubits']/1000 if r else np.nan for r in ys],color=colors[name],label=name,lw=1.5)
ax.set(xlabel='Maximum allowed cycles per CCZ',ylabel='Footprint (k qubits)',xlim=(60,220));ax.grid(alpha=.18);ax.legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(DEST/'figs/budget_curve.pdf');plt.close(fig)
raw=list(csv.DictReader(open(OUT/'calibration_absolute.csv')))
fig,axes=plt.subplots(1,2,figsize=(7.1,2.6),sharey=True)
for ax,kind,title in zip(axes,['memory','stability'],['Space-like coefficient $S$','Time-like coefficient $T$']):
 for q,col in [(1,'#243e60'),(5,'#ad5134')]:
    rs=[r for r in raw if r['kind']==kind and int(r['q_over_p'])==q and int(r['d'])>=5]
    x=np.array([int(r['d']) for r in rs]);y=np.array([float(r['absolute_multiplier']) for r in rs]);lo=np.array([float(r['multiplier_ci95_low']) for r in rs]);hi=np.array([float(r['multiplier_ci95_high']) for r in rs])
    ax.errorbar(x+(q-3)*.025,y,yerr=[y-lo,hi-y],fmt='o-',ms=4,lw=1,color=col,label=f'$q={q}p$',capsize=3)
 ax.axhline(1,color='#777',ls='--',lw=.8);ax.set(title=title,xlabel='Distance / number of rounds',xticks=[5,7,9],ylim=(0,7.5));ax.grid(alpha=.18)
axes[0].set_ylabel('Relative to $p_L(d)/2$');axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(DEST/'figs/absolute_calibration.pdf');plt.close(fig)
print('Generated manuscript tables and figures.')
for name in ['L','D1','D2']:print('120 cycles',name,best('lit',name,120))

# Component-level interface schematic; one side shown, no geometric scale implied.
fig,axes=plt.subplots(1,3,figsize=(7.1,2.15))
for ax,name in zip(axes,['L','D1','D2']):
    ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off');ax.set_title(name,fontsize=11,color=colors[name])
    def box(x,y,w,h,label,color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#566273',lw=.7))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7)
    box(.4,7.8,4.2,1.2,'Supplier 1','#e8d9c6');box(5.4,7.8,4.2,1.2,'Supplier 2','#e8d9c6')
    box(.4,.6,9.2,1.3,'Level-2 patches','#dbe4ee')
    box(.4,2.4,9.2,1.2,'Shared merge ancilla','#f0d5cf')
    if name=='L':
        box(.4,4.6,9.2,1.5,'Transport + two buffer spots\n+ correction patches','#f6e5b8')
        mid=6.15
    elif name=='D2':
        box(.4,4.6,4.2,1.3,'Buffer 1','#f6e5b8');box(5.4,4.6,4.2,1.3,'Buffer 2','#f6e5b8');mid=5.95
    else:
        ax.text(5,5.5,'Output stays in supplier\nuntil adaptive readout',ha='center',va='center',fontsize=7);mid=3.65
    for x in ([.8,9.2] if name=='D1' else [2.5,7.5]):
        ax.annotate('',xy=(x,mid),xytext=(x,7.7),arrowprops=dict(arrowstyle='->',lw=.7,color=colors[name]))
        if name!='D1':ax.annotate('',xy=(x,3.65),xytext=(x,4.5),arrowprops=dict(arrowstyle='->',lw=.7,color=colors[name]))
fig.tight_layout();fig.savefig(DEST/'figs/interface_schematic.pdf');plt.close(fig)

# Expand generated rows before compilation (avoids input-boundary alignment issues).
template=DEST/'manuscript.tex'
if template.exists():
    tex=template.read_text()
    for path in (DEST/'generated').glob('*.tex'):
        body=path.read_text().rstrip()+'\n';path.write_text(body)
        tex=tex.replace('\\input{generated/'+path.name+'}',body)
    (DEST/'main.tex').write_text(tex)
