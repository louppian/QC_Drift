from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Result/Phase2/C_analysis'
OUT.mkdir(parents=True, exist_ok=True)
def calculate(d,B=0,seed=20261012):
 pats=d.groupby('pid',sort=True).year.first();pids=list(pats.index); point={};boots={};counts=[];mats={};rng=np.random.default_rng(seed)
 draws=np.zeros((B,len(pids)))
 if B:
  for y in [2024,2026]:
   ix=np.flatnonzero(pats.to_numpy()==y);draws[:,ix]=rng.multinomial(len(ix),np.repeat(1/len(ix),len(ix)),size=B)
 for y in [2024,2026]:
  for k in [2,3]:
   for direction in ['up','down','net']:
    sub=d[d.year.eq(y)].copy()
    if direction!='net':sub=sub[sub.old_RB.eq(k if direction=='up' else k+1)]
    event=(sub.new_RB.ge(k+1) if direction=='up' else sub.new_RB.le(k) if direction=='down' else sub.new_RB.ge(k+1).astype(int)-sub.old_RB.ge(k+1).astype(int)).astype(float)
    sub['num']=event*sub.w;sub['den']=sub.w
    mat=sub.groupby('pid')[['num','den']].sum().reindex(pids,fill_value=0).to_numpy();name=f'{direction}{k}{k+1}_{y}';mats[name]=mat
    point[name]=mat[:,0].sum()/mat[:,1].sum() if mat[:,1].sum() else np.nan
    if B:
     sums=draws@mat
     with np.errstate(divide='ignore',invalid='ignore'):boots[name]=sums[:,0]/sums[:,1]
    counts.append(dict(term=name,numerator=float(event.sum()),denominator=len(sub),weighted_num=float(mat[:,0].sum()),weighted_den=float(mat[:,1].sum()),patients=sub.pid.nunique()))
 specs={'T':{'up34_2024':1,'down34_2024':-1},'U':{'up34_2026':1,'down34_2026':-1},'D23_2024':{'up23_2024':1,'down23_2024':-1},'D23_2026':{'up23_2026':1,'down23_2026':-1},'delta_year':{'T':1,'U':-1},'delta_cutpoint_2024':{'T':1,'D23_2024':-1},'delta_cutpoint_2026':{'U':1,'D23_2026':-1},'C_down':{'down34_2026':1,'down34_2024':-1},'net_year':{'net34_2024':1,'net34_2026':-1},'net_cutpoint_2024':{'net34_2024':1,'net23_2024':-1},'net_cutpoint_2026':{'net34_2026':1,'net23_2026':-1}}
 for name,terms in specs.items():
  point[name]=sum(point[t]*v for t,v in terms.items())
  if B:boots[name]=sum(boots[t]*v for t,v in terms.items())
 return point,boots,counts
d=pd.read_csv(OUT/'c1_read_results.csv');assert len(d)==1155 and d.uid2.is_unique;d['w']=1.0
p,b,counts=calculate(d,20000)
names=['T','U','delta_year','delta_cutpoint_2024','delta_cutpoint_2026','D23_2024','D23_2026','C_down','net34_2024','net34_2026','net_year','net_cutpoint_2024','net_cutpoint_2026']
r={}
for name in names:
 v=b[name];v=v[np.isfinite(v)];q=np.quantile(v,[.025,.05,.95,.975]);r[name]=dict(estimate=p[name],ci95=[q[0],q[3]],one_sided_lower=q[1],one_sided_upper=q[2],valid=len(v))

rows=[]
conditions=[('H2024','T',1),('H2024','delta_year',1),('H2024','delta_cutpoint_2024',1),('HAI','U',-1),('HAI','delta_year',1),('HAI','delta_cutpoint_2026',-1)]
for name,v in r.items():
 for hypothesis,term,direction in [c for c in conditions if c[1]==name] or [('',name,0)]:
  passed=(v['one_sided_lower']>0 if direction==1 else v['one_sided_upper']<0) if direction else False
  status=('충족' if passed else '반대 방향' if direction*v['estimate']<0 else '불충분') if direction else '보조 지표'
  rows.append(dict(hypothesis=hypothesis,parameter=name,estimate=v['estimate'],ci_low=v['ci95'][0],ci_high=v['ci95'][1],lower_one_sided=v['one_sided_lower'],upper_one_sided=v['one_sided_upper'],direction=direction,status=status,bootstrap=20000,seed=20261012,valid=v['valid'],n_presentations=1155,weighting='equal presentation',cluster='patient within year'))
pd.DataFrame(rows).to_csv(OUT/'c2_hypothesis_results.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(counts).to_csv(OUT/'c2_transition_counts.csv',index=False,encoding='utf-8-sig')
print('Saved all-presentation hypothesis estimates and counts.')
