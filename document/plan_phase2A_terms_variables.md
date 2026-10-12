# Phase 2A 용어 및 변수 사전

이 문서는 `plan_phase2A.md` 본문에서 분리한 용어와 데이터 변수명을 모은 사전이다. 연구 계획서 본문에는 통계 개념명을 쓰고, 실제 산출물 컬럼명과 코드값은 이 문서에서 관리한다.

씨발 무조건 미니시퀀스만 판독 대상이다 판독이면 무조건 분석분모다. 여기 나와있지 않는 용어 사용하지마라.

## 1. 주요 용어

| 용어 | 의미 |
|---|---|
| 원자료 영상 | 원본 데이터셋의 촬영 단위. 한 영상은 네 ROI 관측을 가진다 |
| ROI 관측 | 영상 하나의 한 ROI에 대한 기존 등급과 재판독 등급 쌍 |
| 환자 군집 | 같은 환자에서 나온 모든 mini-sequence와 ROI 관측의 군집 |
| 기존 등급 | 원판독 조건에서 부여된 ROI 등급 |
| 재판독 등급 | Phase 2A 정보 비노출 재판독에서 부여된 ROI 등급 |
| 원판독 조건 | 기존 등급이 생성된 판독 조건. 2024년 AI 비보조 또는 2026년 AI 보조 |
| 재판독 조건 | Phase 2A의 공통 재판독 조건. AI 출력 없이 정보 비노출로 수행 |
| 경계 | 순서형 등급 k와 k+1을 가르는 임계 경계 |
| 상향 경계 통과 | 기존 등급 k가 재판독에서 k+1 이상으로 이동한 사건 |
| 하향 경계 통과 | 기존 등급 k+1이 재판독에서 k 이하로 이동한 사건 |
| 경계 통과율 | 특정 경계에서 상향 또는 하향 경계 통과 사건의 조건부 확률 |
| 방향성 이동 | 상향 경계 통과율에서 하향 경계 통과율을 뺀 값 |
| 정보 비노출 재판독 | 판독 화면에서 기존 라벨, 모델 예측, 촬영 연도, 선정 경로, 분모 자격, 예상 수정 방향, 환자 식별자를 가린 재판독 |
| Phase 1 가설 비인지 | 판독자가 표적 연도, ROI, 경계, 예상 방향 및 Phase 1 결과를 모르는 상태 |
| 라벨 출처 | 각 ROI 등급이 기존 판독 또는 Phase 2A 재판독 중 어디에서 왔는지 나타내는 정보 |
| mini-sequence 후보 모집단 | 사전 지정 규칙에 따라 원자료에서 만들 수 있는 모든 mini-sequence 후보의 집합 |
| mini-sequence | 후보 모집단에서 선택되어 판독자에게 제시되는 연속 제시 단위 |
| 선정구간 | 특정 대상 조건을 만족해 mini-sequence 후보를 정의하게 하는 원자료상의 연속 부분 구간 |
| 선정 경로 | 영상이 어떤 규칙 또는 대상군 때문에 판독 세트에 포함되었는지 나타내는 정보 |
| 가중 층 | 결합 포함확률이 같은 관측들의 층 |
| 분모 자격 | ROI 관측이 어떤 조건부 확률의 분모에 속하는지 나타내는 분석 정보 |
| hidden duplicate 경로 | 겹침 유래 2회차 제시분 또는 층화표집 반복 제시분 |
| hidden duplicate 층화 기준 | ROI와 기존 등급의 교차층 |

## 2. 데이터 변수명

| 개념 | 변수명 | 값 또는 형식 |
|---|---|---|
| 원본 영상 식별자 | `uid` | 원자료 영상 단위의 고유 식별자 |
| 원본 환자 식별자 | `pid` | 원자료 환자 단위의 고유 식별자 |
| 원본 시퀀스 내 위치 | `seq` | 환자 내부 원자료 순서 |
| ROI별 기존 등급 | `orig_RT`, `orig_LT`, `orig_RB`, `orig_LB` | 0, 1, 2, 3, 4 |
| 재판독용 영상 식별자 | `uid2` | 정보 비노출 제시용 영상 식별자 |
| 재판독용 mini-sequence 식별자 | `ms_id` | 정보 비노출 제시용 mini-sequence 식별자 |
| mini-sequence 내 제시 순서 | `seq2` | mini-sequence 내부 순서 |
| 전체 재판독 제시 순번 | `read_order` | 최종 재판독 제시 목록의 순번 |
| ROI별 분모 자격 | `denom_role_RT`, `denom_role_LT`, `denom_role_RB`, `denom_role_LB` | 다중값 또는 분모별 불리언 구조 |
| 선정 경로 | `entry_path` | 복수 경로 허용 |
| 가중 층 | `weight_stratum` | 결합 포함확률 값으로 정의한 층 |
| 제시 유형 | `presentation_type` | `original`, `hidden_duplicate_target` |
| hidden duplicate 경로 | `dup_source` | `overlap`, `stratified` |
| hidden duplicate 층화 기준 | `dup_stratum` | ROI×기존 등급 교차층. `dup_source=stratified`에서 기록 |
| mini-sequence 선택확률 | `pi_ms` | 0부터 1 사이의 확률 |
| 최종 영상 포함확률 | `pi_uid` | 0부터 1 사이의 결합 포함확률 |
| ROI별 재판독 등급 | `new_RT`, `new_LT`, `new_RB`, `new_LB` | 0, 1, 2, 3, 4 |

## 3. 주요 코드값

| 변수명 | 코드값 | 의미 |
|---|---|---|
| `entry_path` | `target_census` | 표적군 전수 포함 |
| `entry_path` | `target_ms_interior` | 표적 mini-sequence 내부 포함 |
| `entry_path` | `lower_cutpoint_sampled` | 하위 경계 대조군 표집 포함 |
| `entry_path` | `far_control_sampled` | 원거리 음성 대조군 표집 포함 |
| `entry_path` | `year_control_census` | 원판독 조건 대조군 전수 포함 |
| `entry_path` | `year_control_sampled` | 원판독 조건 대조군 표집 포함 |
| `presentation_type` | `original` | 최초 제시 원본 영상 |
| `presentation_type` | `hidden_duplicate_target` | hidden duplicate 분석 대상 영상 |
| `dup_source` | `overlap` | 전수 포함 구성의 겹침 유래 2회차 제시분 |
| `dup_source` | `stratified` | ROI×기존 등급 층화표집 반복 제시분 |

## 4. 관리 원칙

- 본문 계획서에는 개념명을 사용한다.
- 산출물 생성 코드와 검증표에는 이 변수명을 사용한다.
- 변수명 변경이 필요하면 이 사전을 먼저 고치고, 산출물 생성 코드와 검증표를 같은 변경 묶음으로 갱신한다.
