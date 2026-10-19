"""Export manuscript tables from existing statistics without rerunning inference."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'Result/Phase2/C_analysis'
OUT = ROOT / 'Result/Phase2/D_manuscript'


def markdown(frame):
    cells = [[str(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in cells])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c3 = pd.read_csv(SRC / 'c3_grade_boundary.csv')
    c7 = pd.read_csv(SRC / 'c7_sequence_context.csv')
    primary = c3[c3.analysis.eq('original800') & c3.roi.eq('RB')]
    transition = primary[primary.record_type.eq('transition')].copy()
    transition['row_total'] = transition.groupby(['year', 'old_grade']).n.transform('sum')
    transition['row_rate'] = transition.n / transition.row_total
    transition[['year', 'old_grade', 'new_grade', 'n', 'row_total', 'row_rate']].to_csv(
        OUT / 'table_s4_rb_transitions.csv', index=False, encoding='utf-8-sig')
    sections = ['### 보충표 S4. 연도별 RB 기존 등급과 재판독 등급\n']
    for year, total in [(2024, 500), (2026, 300)]:
        sub = transition[transition.year.eq(year)]
        assert sub.n.sum() == total and len(sub) == 25
        matrix = sub.pivot(index='old_grade', columns='new_grade', values='n').astype(int)
        matrix.columns = [f'재판독 {int(x)}' for x in matrix.columns]
        matrix.insert(0, '기존 등급', matrix.index.astype(int))
        matrix['합계'] = sub.groupby('old_grade').n.sum().astype(int)
        sections += [f'{year}년: {total}장. 셀은 실제 영상 수입니다.\n', markdown(matrix), '']
    sections += ['행 비율은 각 기존 등급의 영상 수를 분모로 하며, CSV에 함께 저장하였다.\n',
                 '### 보충표 S5. 최초 800장의 RB 전체 인접 경계 변화\n']
    counts = primary[primary.record_type.eq('boundary_count')].set_index('term')
    estimates = primary[primary.record_type.eq('boundary_estimate')].set_index('parameter')
    rows, display = [], []
    for year in [2024, 2026]:
        for k in range(4):
            for direction, label in [('up', '높은 쪽으로 변경'), ('down', '낮은 쪽으로 변경')]:
                term = f'{direction}{k}{k+1}_{year}'
                count, est = counts.loc[term], estimates.loc[term]
                rows.append(dict(year=year, boundary=f'{k}-{k+1}', direction=direction,
                                 numerator=int(count.numerator), denominator=int(count.denominator),
                                 weighted_num=count.weighted_num, weighted_den=count.weighted_den,
                                 estimate=est.estimate, ci_low=est.ci_low, ci_high=est.ci_high,
                                 weighting=est.weighting))
                display.append([year, f'{k}·{k+1}', label,
                                f'{int(count.numerator)}/{int(count.denominator)}',
                                f'{100*est.estimate:.2f}', f'[{100*est.ci_low:.2f}, {100*est.ci_high:.2f}]'])
        for k, name in [(2, f'D23_{year}'), (3, 'T' if year == 2024 else 'U')]:
            value = estimates.loc[f'up{k}{k+1}_{year}', 'estimate'] - estimates.loc[f'down{k}{k+1}_{year}', 'estimate']
            assert abs(value - estimates.loc[name, 'estimate']) < 1e-12
        high = estimates.loc['T' if year == 2024 else 'U', 'estimate']
        assert abs(high - estimates.loc[f'D23_{year}', 'estimate'] - estimates.loc[f'delta_cutpoint_{year}', 'estimate']) < 1e-12
    pd.DataFrame(rows).to_csv(OUT / 'table_s5_rb_boundaries.csv', index=False, encoding='utf-8-sig')
    sections += [markdown(pd.DataFrame(display, columns=['연도', '경계', '변화 방향', '실제 분자/분모', '가중 비율 (%)', '95% CI (%)'])),
                 '\n높은 쪽 변경의 분모는 기존 낮은 인접 등급, 낮은 쪽 변경의 분모는 기존 높은 인접 등급이다. 비율과 구간은 기존 역확률 가중 추정값이다. 두 조건부 비율의 차이를 사용한 T·U와 경계 선택성은 본문 표 4에 제시하였다.\n',
                 '### 보충표 S6. mini-sequence 반복 구성과 RB 판단 유지\n']
    c7.to_csv(OUT / 'table_s6_sequence_context.csv', index=False, encoding='utf-8-sig')
    labels = {'different_context': '다른 구간 구성', 'same_grade': '같은 등급 유지',
              'grade4_changed': '4등급 여부 변경', 'repeated_images': '대상 중 반복 영상',
              'single_presentation_images': '대상 중 단회 영상', 'all_repeats_grade3': '모든 반복에서 3 유지',
              'repeat_grade3': '반복 비교에서 3 유지'}
    display = []
    for r in c7.itertuples():
        if r.metric not in labels or str(r.year) == '2024':
            continue
        display.append([r.year, labels[r.metric], {'pair':'쌍', 'image':'장', 'sequence':'시퀀스'}[r.unit],
                        f'{int(r.numerator)}/{int(r.denominator)}', f'{100*r.estimate:.1f}'])
    sections += [markdown(pd.DataFrame(display, columns=['연도', '항목', '단위', '분자/분모', '비율 (%)'])),
                 '\n대상은 AI≤3·기존 4·최초 재판독 3인 23장이다. 반복된 6장에서 8쌍의 비교가 발생했다. 원본 길이 분포와 전체 반복 지표는 CSV에 저장하였다. 반복 비교는 서로 다른 7장 구간 사이의 판단 유지를 평가한다.\n']
    (OUT / 'supplemental_tables.md').write_text('\n\n'.join(sections), encoding='utf-8')
    mapping = [
        ('1','2.1','c1_read_results.csv','presentation','analysis populations','image/patient'),
        ('2','3.1','a2_label_image_auc.csv','','boundary 3-4','image'),
        ('3','3.2','b1_summary.csv','','delta_corrected','patient bootstrap'),
        ('4','3.4','c3_grade_boundary.csv','boundary_estimate','original800; T,U,delta_year,delta_cutpoint','weighted conditional grade'),
        ('5','3.5','c6_three_hypotheses.csv','statistic;ai_transition_rate','AI and old grade conditions','conditional image'),
        ('6','3.6','c6_three_hypotheses.csv','ai_boundary','old_above_ai','conditional image'),
        ('7','3.7','c4_repeat_variability.csv','comparison','RB','repeat pair'),
        ('8','3.9','c6_three_hypotheses.csv','statistic','AI agreement and MAE','300 images'),
        ('S1','supplement','b1_summary.csv','','raw,matched,ROI contrasts','patient bootstrap'),
        ('S2','supplement','c6_three_hypotheses.csv','ai_boundary;ai_boundary_contrast','boundary comparisons','conditional image'),
        ('S3','supplement','c3_grade_boundary.csv','boundary_count;boundary_estimate','T,U; all populations','population-specific'),
        ('S4','3.4','c3_grade_boundary.csv','transition','original800;RB','old grade row'),
        ('S5','3.4','c3_grade_boundary.csv','boundary_count;boundary_estimate','original800;RB;up/down','conditional grade'),
        ('S6','3.8','c7_sequence_context.csv','all','all','image/pair/sequence')]
    pd.DataFrame(mapping, columns=['table','section','source','record_type','selection','denominator']).to_csv(
        OUT / 'manuscript_mapping.csv', index=False, encoding='utf-8-sig')
    print('Generated three table CSVs, supplemental_tables.md, manuscript_mapping.csv')


if __name__ == '__main__':
    main()
