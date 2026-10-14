"""Render final RB tables from verified c2/c3 outputs; no inference or relabeling."""
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Result/Phase2/C_analysis'
h = pd.read_csv(OUT/'c2_hypothesis_results.csv')
c = pd.read_csv(OUT/'c2_transition_counts.csv').set_index('term')
v = pd.read_csv(OUT/'c3_repeat_variability.csv')
v = v[v.roi.eq('RB')]

def interval(lo, hi, scale=1, digits=2):
    return f'[{lo*scale:.{digits}f}, {hi*scale:.{digits}f}]'

def fraction(num, den):
    return f'{int(num)}/{int(den)}'

rows1 = []
conditions = [('H2024','T','① 2024 상향 수정','비보조→비보조 이동 T>0'),
 ('H2024','delta_year','① 2024 상향 수정','원판독 조건 선택성 T−U>0'),
 ('H2024','delta_cutpoint_2024','① 2024 상향 수정','2024 경계 선택성 >0'),
 ('HAI','U','② 2026 상향 anchoring','AI 보조→비보조 이동 U<0'),
 ('HAI','delta_year','② 2026 상향 anchoring','원판독 조건 선택성 T−U>0'),
 ('HAI','delta_cutpoint_2026','② 2026 상향 anchoring','2026 경계 선택성 <0')]
for hypothesis, parameter, title, condition in conditions:
    matches = h[h.hypothesis.eq(hypothesis)&h.parameter.eq(parameter)]
    assert len(matches)==1
    r = matches.iloc[0]
    direction=int(r.direction)
    passed = r.lower_one_sided>0 if direction==1 else r.upper_one_sided<0
    status = '충족' if passed else '반대 방향' if direction*r.estimate<0 else '불충분'
    assert status==r.status
    up=down='—'
    if parameter in ['T','U']:
        year=2024 if parameter=='T' else 2026
        a=c.loc[f'up34_{year}'];b=c.loc[f'down34_{year}']
        up=f'{fraction(a.numerator,a.denominator)} ({a.numerator/a.denominator*100:.1f}%)'
        down=f'{fraction(b.numerator,b.denominator)} ({b.numerator/b.denominator*100:.1f}%)'
        np.testing.assert_allclose(r.estimate,a.numerator/a.denominator-b.numerator/b.denominator)
    bound=r.lower_one_sided if direction==1 else r.upper_one_sided
    rows1.append({'가설':title,'조건':condition,'상향':up,'하향':down,'추정값(%p)':f'{r.estimate*100:+.2f}',
        '양측 95% CI(%p)':interval(r.ci_low,r.ci_high,100),
        '단측 95% 한계(%p)':('하한 ' if direction==1 else '상한 ')+f'{bound*100:+.2f}','판정':status})

rows2=[]
for metric,label in [('ACC','RB 전체 ACC'),('MAE','RB 전체 MAE'),('boundary_disagreement','4 대 ≤3 경계 불일치율')]:
    block=v[v.metric.eq(metric)]
    assert block.repeat_denominator.eq(324).all() and block.repeat_estimate.nunique()==1
    for year,condition in [('2024','2024 비보조'),('2026','2026 AI 보조'),('all','전체')]:
        r=block[block.year.astype(str).eq(year)].iloc[0]
        scale=1 if metric=='MAE' else 100;digits=3 if metric=='MAE' else 2
        unit='' if metric=='MAE' else '%';difference_unit='' if metric=='MAE' else '%p'
        a=fraction(r.past_numerator,r.past_denominator);b=fraction(r.repeat_numerator,r.repeat_denominator)
        f=Fraction(int(r.past_numerator),int(r.past_denominator))-Fraction(int(r.repeat_numerator),int(r.repeat_denominator))
        np.testing.assert_allclose(float(f),r.difference,atol=1e-14)
        status='확인' if (r.difference_ci_high<0 if metric=='ACC' else r.difference_ci_low>0) else '판단 불충분'
        assert (status=='확인')==(r.exceeds_repeat=='confirmed')
        rows2.append({'지표':label,'원판독 조건':condition,
          '과거↔현재 첫 판독':f'{a} = {r.past_estimate*scale:.{digits}f}{unit}',
          '과거↔현재 95% CI':interval(r.past_ci_low,r.past_ci_high,scale,digits)+unit,
          '전체 재판독 내부 기준':f'{b} = {r.repeat_estimate*scale:.{digits}f}{unit}',
          '전체 반복 95% CI':interval(r.repeat_ci_low,r.repeat_ci_high,scale,digits)+unit,
          '차이 계산':f'{a} − {b} = {f}',
          '차이':f'{r.difference*scale:+.{digits}f}{difference_unit}',
          '차이 95% CI':interval(r.difference_ci_low,r.difference_ci_high,scale,digits)+difference_unit,
          '반복 불일치 초과':status})
assert len(rows1)==6 and len(rows2)==9
# Specific prior transcription error: pooled ACC upper CI must be negative.
assert v[v.metric.eq('ACC')&v.year.astype(str).eq('all')].iloc[0].difference_ci_high<0
pd.DataFrame(rows1).to_csv(OUT/'c4_table1.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows2).to_csv(OUT/'c4_table2.csv',index=False,encoding='utf-8-sig')
text='''최종 RB 표 1·2와 해석

표 1: 전체 1,155개 제시(831개 고유 영상); 반복 제시 포함, 행별 동일 가중치.
상향=원판독 3→재판독 4, 하향=원판독 4→재판독 ≤3.
방향성 이동=상향률−하향률. 경계 선택성=해당 조건의 3↔4 이동−2↔3 이동.
가설 ①은 공동조건 불충족, 가설 ②는 주검정의 세 단측 조건 충족.

표 2: 287개 고유 영상의 324개 첫–반복 쌍; 2024 249쌍, 2026 75쌍.
세 번 이상 제시된 영상은 각 반복을 최초 현재 판독과 연결하며, 과거–최초 현재 판독도 같은 쌍의 가중치로 비교한다.
반복 기준은 두 연도를 합친 전체 324쌍으로 고정한다.
ACC는 정답률이 아니라 정확한 등급 일치율이다.
MAE 분자는 절대 등급 차이의 합이며 단위는 등급이다.
경계 불일치는 한 판독이 4이고 다른 판독이 ≤3인 양방향 사건이다.
첫·반복 순서를 뒤집어도 이 세 지표는 동일하다.
분수 차이는 두 비율의 산술적 차이이며 새로운 관측 건수/표본 수가 아니다.
ACC 차이의 양측 95% CI 상한<0, MAE·경계 불일치율 차이의 하한>0이면 초과 확인, 나머지는 판단 불충분이다.

신뢰구간: 연도별 환자 군집 bootstrap 20,000회.
표 1 seed=20261012, 표 2 seed=20261014.
환자의 모든 해당 영상과 반복 쌍을 보존하고 각 표의 모수·공통 기준·차이를 같은 재표집에서 산출한다.
표 1의 단측 가설 판정과 표 2의 양측 차이 판정을 구분한다.

해석
2026 AI 보조 원판독을 현재 비보조로 재판독할 때 RB 상위 경계가 선택적으로 낮아져, 2024 대비 상대적 상향 부여 설명에 부합한다.
중복 영상에서 2024 비보조 원판독–현재 재판독의 불일치는 현재 평가자의 전체 반복 변동을 초과했고, 2026 AI 보조 원판독의 불일치는 반복 변동과 통계적으로 구별되지 않았다.
이 패턴에서 AI가 판독 기준을 일정하게 잡아줬을 가능성을 탐색적으로 제시할 수 있다. 기준 안정화는 가능성에 관한 해석이며, 반복 수준과의 동등성 또는 정확도 우위의 판정이 아니다.
RB 상위 등급의 상대적 상향은 가설 검정이 지지하는 방향성 결과로 구분한다.
기준 안정화의 해석은 중복 영상 비교에 한정한다. 전체 1,155개 제시의 RB 오차 SD는 2024 0.638, 2026 0.665로 산포 감소가 일관된 결과는 아니다.
기존의 조건부 방향성 반복 초과 지표를 표 2의 불일치 크기 또는 정답 기준으로 사용하지 않는다.
'''
(OUT/'c4_interpretation.txt').write_text(text,encoding='utf-8')
print('PASS: final table arithmetic, common reference, CI signs and decisions validated; exported 6 and 9 rows.')
