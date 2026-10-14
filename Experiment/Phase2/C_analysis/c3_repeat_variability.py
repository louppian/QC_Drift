from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Result/Phase2/C_analysis'
OUT.mkdir(parents=True, exist_ok=True)
j=pd.read_csv(OUT/'c1_read_results.csv').sort_values('read_order')
f=j.drop_duplicates('uid');r=j[j.uid.duplicated()].merge(f[['uid','new_RT','new_LT','new_RB','new_LB']],on='uid',suffixes=('','_first'),validate='many_to_one')
assert len(r)==324 and r.uid.nunique()==287
out=[];rng=np.random.default_rng(20261014)
for year in ['all',2024,2026]:
 d=r if year=='all' else r[r.year.eq(year)]
 pats=d.groupby('pid',sort=True).year.first();draw=np.zeros((20000,len(pats)))
 for y in pats.unique():
  ix=np.flatnonzero(pats.to_numpy()==y);draw[:,ix]=rng.multinomial(len(ix),np.repeat(1/len(ix),len(ix)),20000)
 for roi in ['RT','LT','RB','LB']:
  old=d['old_'+roi].to_numpy();first=d['new_'+roi+'_first'].to_numpy();rep=d['new_'+roi].to_numpy()
  a=first-old;b=rep-first
  metrics={'past_current_mae':abs(a),'repeat_mae':abs(b),'mae_excess':abs(a)-abs(b),'past_current_disagree':(a!=0).astype(float),'repeat_disagree':(b!=0).astype(float),'disagreement_excess':(a!=0).astype(float)-(b!=0).astype(float),'past_current_bias':a,'repeat_bias':b,'past_current_cross34':((old==4)!=(first==4)).astype(float),'repeat_cross34':((first==4)!=(rep==4)).astype(float),'cross34_excess':((old==4)!=(first==4)).astype(float)-((first==4)!=(rep==4)).astype(float)}
  item=dict(year=year,roi=roi,pairs=len(d),images=d.uid.nunique(),patients=len(pats),repeat_up=int((b>0).sum()),repeat_down=int((b<0).sum()),repeat_same=int((b==0).sum()),repeat_3to4_num=int(((first==3)&(rep==4)).sum()),repeat_3to4_den=int((first==3).sum()),repeat_4down_num=int(((first==4)&(rep<=3)).sum()),repeat_4down_den=int((first==4).sum()))
  for name,values in metrics.items():
   t=pd.DataFrame({'pid':d.pid.to_numpy(),'v':values,'n':1}).groupby('pid')[['v','n']].sum().reindex(pats.index);z=draw@t.to_numpy();v=z[:,0]/z[:,1];item[name]=dict(estimate=float(np.mean(values)),ci95=np.quantile(v,[.025,.975]).tolist())
  # Equal weight per unique image, averaging multiple repeat contrasts.
  u=pd.DataFrame({'uid':d.uid.to_numpy(),'mae_excess':abs(a)-abs(b),'cross34_excess':((old==4)!=(first==4)).astype(float)-((first==4)!=(rep==4)).astype(float)}).groupby('uid').mean();item['unique_image_sensitivity']=u.mean().to_dict()
  out.append(item)

counts=[];rows=[]
for item in out:
 base={k:v for k,v in item.items() if not isinstance(v,dict)};counts.append(base)
 for metric,v in item.items():
  if isinstance(v,dict) and 'estimate' in v:
   rows.append(dict(year=item['year'],roi=item['roi'],metric=metric,estimate=v['estimate'],ci_low=v['ci95'][0],ci_high=v['ci95'][1],pairs=item['pairs'],patients=item['patients'],bootstrap=20000,seed=20261014,cluster='patient within year'))
 for metric,v in item['unique_image_sensitivity'].items():
  rows.append(dict(year=item['year'],roi=item['roi'],metric='unique_image_'+metric,estimate=v,pairs=item['pairs'],patients=item['patients']))
pd.DataFrame(counts).to_csv(OUT/'c3_repeat_counts.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(OUT/'c3_repeat_variability.csv',index=False,encoding='utf-8-sig')
print('Saved paired past/current versus first/repeat comparisons.')
