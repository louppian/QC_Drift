# Phase 2 export 이후 분석

분석 순서는 방향반전분해와 재판독의 상대적 차이 → 등급별 이동 및 RB 경계 → 반복판독 → 과거 AI 대조다. c7에서 서로 다른 mini-sequence의 반복 판단 유지와 원본 길이를 기술적으로 집계한다. 기존 코드와 결과가 있는 `C_analysis_old`는 수정하지 않는다.

## 코드와 산출물

각 `c*.py`는 같은 이름의 CSV 하나만 `Result/Phase2/C_analysis/`에 생성한다. 한 CSV 안에서 행의 종류는 `record_type`으로 구분한다. 수치는 숫자 열로 저장하며 JSON 문자열이나 보고서용 반올림 문자열로 바꾸지 않는다.

| 코드 / CSV | 역할 | record_type |
|---|---|---|
| c1_read_results | 공통 입력 준비. 반환 NPZ 검증, 최신 유효 저장본 선택, 원판독 연결, 원래 선정 영상의 포함확률 연결 | presentation, file_audit |
| c2_relative_difference | Phase 1 방향별 bias·분해·보정값과 재판독의 ROI별 평균 변화 및 연도 간 수정량 차이 | reread_change, year_contrast, phase1_run, phase1_decomposition, phase1_corrected |
| c3_grade_boundary | 기존×재판독 5×5 등급 교차표, 기존 등급별 변화, RB 0↔1부터 3↔4까지 경계 통과 및 기존 contrast | transition, grade_change, boundary_count, boundary_estimate, hypothesis_condition |
| c4_repeat_variability | 기준 안정화 평가: 기존–현재와 첫–반복 비교, 반복 건수, 연도 직접 대비, 전체 제시 RB 차이 SD | comparison, repeat_count, year_contrast, full_presentation_dispersion |
| c5_ai_linkage | 과거 AI 예측·원판독·재판독 직접 연결, 등급별 건수, AI 일치도 변화, 경계별 판단 유지·변경과 환자 bootstrap | statistic, ai_transition, down_ai_by_grade, count, observation, source, ai_boundary, ai_boundary_contrast, ai_transition_rate |

| c6_three_hypotheses | 공통 분모 B·C와 탐색적 가설 방향 비교 및 공개 AI 집계 | hypothesis_estimate, hypothesis_count, hypothesis_condition 및 c5 집계 |
| c7_sequence_context | 구간 구성 차이·반복 판단 유지·원본 길이 집계 | sequence_composition, repeat_grade, target_repeat, source_length |

c1은 실험 공통 입력이다. c2~c5가 기존 네 분석이고 c6는 공통 분모의 가설 방향 비교와 공개용 AI 집계다. 이전 c4의 보고서 표는 별도 실험이나 산출물로 생성하지 않는다. 표의 숫자와 판정은 c3·c4 CSV에서 재구성할 수 있다.

`c6_three_hypotheses.py`는 최초 800장과 c5의 검증된 영상별 연결을 사용한다. T는 2024년 기존 상향률−하향률이다. B와 C는 2026년 기존 RB 4등급 55장을 공통 분모로 하며, B는 AI 4·재판독 ≤3, C는 AI ≤3·재판독 ≤3의 비율이다. 같은 환자 추출에서 B−C와 C−B를 계산하고, 기존 Phase 2와 같은 양측 95% percentile CI 및 단측 95% 한계로 탐색적 방향 조건을 평가한다. 두 차이는 같은 대비의 반대 방향이다. 동시 CI나 다중비교 보정은 적용하지 않는다. `hypothesis_estimate`, `hypothesis_count`, `hypothesis_condition`에 결과를 저장한다. c5의 집계 행도 함께 내보내되 영상·환자 식별자, 원자료 경로·행 번호와 해시는 제외하므로 이 CSV는 Git에 포함할 수 있다. c1·c5의 로컬 원자료 연결 CSV는 계속 제외한다.

## 분석 대상과 수치 정의

- `analysis=original800`: 원래 선정한 첫 제시 800장(2024년 500장, 2026년 300장). 선정 노트북에서 `img_sel`과 `pi_uid`를 다시 계산한다. 단순히 전체 고유 영상의 앞 800개를 쓰지 않는다.
- `analysis=unique831`: 반복 구간에 추가된 31장을 포함한 고유 영상의 첫 판독 831장.
- `analysis=presentation1155`: 반복판독을 포함한 전체 1,155개 제시.
- c2의 평균·수정량과 c3의 등급 교차표·등급별 변화는 위 각 관찰 표본에서 동일 가중치로 계산한 기술통계다. 평균 변화는 `재판독−기존`, 수정량 차이는 `2024년 변화−2026년 변화`이며 단위는 등급이다. 새 평균 변화 CI는 추가하지 않았다.
- c2의 Phase 1 방향별 bias는 기존 a4 CSV의 5회 실행 평균이다. `phase1_corrected`는 기존 b1 CSV의 보정값과 CI 등을 그대로 연결한다. 분해를 재판독 수정량과 같은 추정량으로 취급하지 않는다.
- c3의 RB 경계 추정은 기존 계산을 유지한다. original800은 `1/pi_uid`, 나머지 두 분석 대상은 동일 가중치다. `boundary_count`는 실제 분자·분모와 가중 분자·분모를 모두 담는다. 조건부 상향률−하향률과 전체 평균 등급 변화를 구분한다.
- c3는 연도별 환자 군집 bootstrap 20,000회, seed 20261012를 유지한다. `ci_low/high`는 양측 95% CI, `lower/upper_one_sided`는 단측 95% 한계다. 확률·차이는 0~1 척도로 저장하며 %p 표시는 100을 곱한다.
- c3의 `hypothesis_condition`은 기존 H2024·HAI 조건별 판정을 보존한다. 해당 조건 충족을 AI anchoring의 원인 확인으로 해석하지 않는다. 두 연도의 수정이 함께 기여하는지는 c2 및 등급별 이동에서 확인한다.
- c4는 287개 영상의 324개 첫–반복 쌍을 유지한다. 세 번 이상 제시된 영상의 각 반복은 첫 판독과 연결한다. 두 연도의 전체 반복판독을 공통 기준으로 하며 bootstrap 20,000회, seed 20261014를 유지한다. ACC는 정확한 등급 일치율이며 외부 정답에 대한 정확도가 아니다.
- c4의 `full_presentation_dispersion`은 전체 1,155개 제시(2024년 763개, 2026년 392개)의 RB `재판독−기존` 표본 SD(ddof=1)다. 기존 보고서의 0.638·0.665를 원자료에서 계산하며 평균·변화 합계·제곱합을 함께 저장한다. 반복쌍 비교와 표본이 다르며 이 SD에는 CI나 가설 판정을 추가하지 않는다. 기준 안정화 가능성은 반복 비교의 관찰 양상으로 해석하고, 전체 산포 감소나 연도 간 안정성 우위가 입증됐다는 주장과 구분한다.
- c5는 original800 중 2026년 300장·33명의 원자료를 PNG 해시, 과거 NPZ, result.txt의 4 ROI 라벨과 AI 예측으로 대조한다. 과거 AI `inference` 순서 [RT, RB, LT, LB]를 [RT, LT, RB, LB]로 바꾼다. 환자 paired bootstrap 20,000회, seed 20261012와 조건부 분모가 비는 반복의 제외 규칙을 유지한다. 원자료 NPZ 경로·행 번호와 해시를 같은 CSV에 보존한다.

c1과 c5는 영상별 식별자를 포함하므로 기존 c1과 같이 로컬 CSV를 Git에서 제외한다. CSV 생성과 검증에는 영향이 없다.

### AI 경계별 추가 산출물

- `ai_boundary`: 2026년 original800의 RB 300장에서 0↔1, 1↔2, 2↔3, 3↔4를 각각 비교한다. `old_above_ai`는 AI가 경계 이하이고 원판독이 경계 초과인 영상, `old_below_ai`는 그 반대다. 해당 영상들을 분모로 재판독이 AI와 같은 경계 쪽으로 바뀐 `changed_to_ai_side`와 기존 판독과 같은 쪽을 유지한 `stayed_on_old_side`의 실제 분자·분모, 환자 수, 비율과 CI를 저장한다. 같은 쪽 유지가 정확한 등급 유지나 AI와 정확한 등급 일치를 뜻하지는 않는다.
- `ai_boundary_contrast`: 각 방향에서 3↔4의 `changed_to_ai_side` 비율에서 0↔1, 1↔2, 2↔3의 비율을 각각 뺀다. 양쪽 분자·분모와 차이 및 CI를 저장한다. 경계 사이에 같은 영상이 포함될 수 있으므로 같은 환자 bootstrap 추출을 사용한다. 양쪽 분모가 모두 존재하는 반복만 CI에 포함하며 `valid`에 그 수를 기록한다. 탐색적 비교이며 다중비교 보정은 적용하지 않는다.
- `ai_transition_rate`: AI 3·원판독 4인 영상을 분모로 재판독 3의 건수·비율과 CI를 저장한다. 기존 `ai_transition`의 조합별 건수와 구분한다.
- 추가 AI 통계는 기존과 같은 환자 paired bootstrap 20,000회, seed 20261012를 사용한다. 비율과 차이는 0~1 척도다. 이 통계는 재판독을 외부 정답으로 지정하거나 판독자의 심리·원인을 판정하지 않는다.

## 실행

저장소 루트에서 실행한다. Python에 numpy, pandas가 필요하다. 원래 선정 단계는 노트북을 읽어 메모리에서 복원하며 selection/export 파일이나 영상 NPZ를 새로 생성하지 않는다.

```powershell
python Experiment/Phase2/C_analysis/c1_read_results.py --returns "D:/InhaUH_CXR/2026.10 CXRs"
python Experiment/Phase2/C_analysis/c2_relative_difference.py
python Experiment/Phase2/C_analysis/c3_grade_boundary.py
python Experiment/Phase2/C_analysis/c4_repeat_variability.py
python Experiment/Phase2/C_analysis/c5_ai_linkage.py --historical-root "D:/InhaUH_CXR/2026.05 CXRs"
python Experiment/Phase2/C_analysis/c6_three_hypotheses.py
python Experiment/Phase2/C_analysis/c7_sequence_context.py
python Experiment/Phase2/D_manuscript/d1_manuscript_tables.py
```

c1과 c5의 `--unified-labels` 기본값은 `CXR/unified_labels.csv`다. 원자료 경로는 인자로 전달한다. 분석 코드는 old 코드·결과를 입력으로 사용하지 않는다.

## 기존 수치 유지 확인

정리 당시 `Result/Phase2/C_analysis_old/`를 기준으로 아래 항목 31,543개 값을 비교해 통과했다. 수치 허용오차는 절대·상대 각각 1e-12였다. 비교 완료 후 단순 실행용 run.py와 일회성 verify.py는 제거했다.

- c1의 전체 판독 행·등급·저장 시각·선택 파일과 전체 파일 감사 기록.
- Phase 1 실행별 bias·보정값, 기존 판독 행에서 독립 집계한 ROI별 재판독 평균과 등급 교차표.
- 기존 c2의 경계 통과 분자·분모·가중 분자·분모, 추정값·CI·조건별 판정.
- 기존 c5의 세 분석 대상별 800/831/1,155건 가중 추정값·CI와 환자 수.
- 기존 c3의 반복 건수·지표·CI·직접 연도 대비 전체.
- 기존 c5의 과거 AI 통계·CI 및 하향 수정의 AI별 건수. JSON과 CSV 두 기존 저장본을 대조.
- 기존 c4 보고서 표 두 개의 표시 숫자·반올림 CI·분수·판정.

제시 문맥 관련 기존 산출물은 비교 대상에서 제외한다. 기존 해석 TXT 및 최종 보고서는 덮어쓰지 않으며, 새 분석은 위 CSV의 숫자를 기준으로 읽는다.


## 별도 생성하지 않는 기존 출력

- c7은 구간 구성과 반복 판단 유지, 원본 길이 분포를 집계한다. 길이별 효과 검정과 제외 영상 수에 따른 수정 연관 분석은 별도 생성하지 않는다.
- 기존 보고서 표 두 개와 해석 TXT는 별도 생성하지 않는다. 수치와 조건 판정은 c3·c4에 보존했다.
- 기존 extra31의 연도·등급·선정 역할 요약은 별도 생성하지 않는다. 해당 31개 행은 c1의 record_type=presentation, read_type=first_read, pi_uid가 비어 있는 행으로 남아 있다.
- 기존 검증 요약 JSON은 별도 생성하지 않는다. 반환 파일 검증은 c1, 과거 AI 원자료 검증은 c5에 포함한다.

## 원고 표 연결

D_manuscript/d1_manuscript_tables.py는 기존 c3와 c7을 읽어 Result/Phase2/D_manuscript에 S4–S6 CSV, supplemental_tables.md, manuscript_mapping.csv를 생성한다. 새 통계 검정 없이 기존 추정값과 기술 집계를 연결한다. c7 공개 CSV에는 영상·환자 식별자와 원본 경로를 저장하지 않는다.
