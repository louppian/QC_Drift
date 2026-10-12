# 실험 계획서

영역 표시 순서: **[RT, LT, RB, LB]**. 복원한 E0(a9)·특징(a10/a11)의 내부 순서는 원본 **[RT, RB, LT, LB]**를 유지한다.

## 0. 잘못 기록 — cross 에 matched 를 섞은 오류

재현 대상은 `D:\npjDM2026\draftv11.md` §5.0 표다.

1. **cross 에 matching 을 섞은 게 오류다.** draftv11 §5.0: **Exp 1 cross(fwd/rev)는 raw 코호트**,
   정합은 **Exp 2 in-domain(같은 해)에만** 들어간다. cross 의 평균회귀(RTM)는 matching 이 아니라
   **Δγ 보정**(δ_corr = δ_obs + Δγ/2)이 흡수한다. 그런데 `e1_cross.cross_val_fold_matched` 를
   만들어 cross 에 정합을 붙였다 — 마크다운에 없는 단계다. (제거함. cross = raw 로 고정.)
2. **rev 불일치(+0.224 vs md +0.004)를 matched/백본으로 오판했다.** matched Δγ 는 작아(CI 0 포함)
   보정해도 δ_corr≈δ_obs 이므로 **matching 은 rev 을 못 고친다.** 실제 문제는 **raw cross rev 자체가
   md 의 raw cross rev 과 다른 것**이다.

**유지(오류 아님):** **val 5-fold(8:2) 비중복은 우리 의도 설계**다 — best_val 의 val-운을 없애려는
안정화. md 의 seed 3회와 방식만 다를 뿐 **둘 다 raw cross** 이며, 이건 이탈이 아니라 우리 설계 선택이다.

**교정 기준:** cross = raw + val 5-fold 유지. matching 은 Exp 2(in-domain)에서 Δγ 재는 용도로만.

## 1. 데이터

| 항목 | 값 |
|---|---|
| 경로 | `CXR/Merged/` |
| 라벨 | `labels.csv` — `uid, patient_id, RT, LT, RB, LB, image_path` |
| 연도 구분 | `patient_id` 접두어 `24_` / `26_` |
| 이미지 | `images_normalize/<uid>.png` — 512×512 grayscale, seg+STN 정렬 완료 |
| 마스크 | `masks/<uid>.png` — 512×512, 정렬 완료 |
| 등급 | 0~4 (5-class), ROI 4개 |
| 2024 | 68 환자 / 718 영상 |
| 2026 | 46 환자 / 587 영상 |

## 2. 학습 규약 (공통)

| 파라미터 | 값 |
|---|---|
| epochs | 50 |
| early-stop patience | 10 (val MAE 기준) |
| 최종 test 모델 | **best-val 체크포인트 reload** |
| 백본 동결 | FREEZE_BLOCKS = 6 |
| train : val | **8 : 2** (val 20%, 환자 단위) — E1·E2 는 5-fold 비중복 val |
| batch size | 32 |
| optimizer | AdamW |
| lr | encoder 1e-5 / head 1e-4 |
| weight decay | encoder 5e-5 / head 5e-4 |
| scheduler | CosineAnnealing (T_max = 50) |
| 정규화 | Normalize([0.56], [0.17]) |
| seeds | 1, 2, 42 |

## 3. 모델

| 모델 | 구성 |
|---|---|
| **DORGA** | MRM ViT-base/16-512 백본(`DORGA_Brixia.pth`, in_chans=1, drop_path 0.1, global_pool avg) + Dynamic prior, R=4·C=5·K=7, proj_dim=768, GNN heads=4·dropout=0.1 |
| **BSNet** | ResNet18 + 하드어텐션, BrixiaLoss |
| **PAFE** | ResNet34+ViT (3ch 내부복제), CE |

## 4. E 실험

| 실험 | 설계 | 산출 |
|---|---|---|
| **E1** cross | **val 5-fold(8:2) 비중복.** 방향마다 train연도를 5-fold → fold k=val(20%)·나머지=train(80%), **test=반대연도 전체(고정)**. split 1~5. fwd(2024→2026)·rev(2026→2024). δ_obs=(rev−fwd)/2, γ=(rev+fwd)/2, split 평균 | `checkpoint/E1/dorga/{24to26,26to24}_split{1..5}/` |
| **E2** in-domain matched | 코호트별 환자 5-fold(전 환자 1회 test) × init_seeds 42·1·2, **fold 내 8:2 val**. **matched**(두 코호트 등급분포 정합 서브샘플). Δg_matched = g₂₀₂₄ − g₂₀₂₆ | `checkpoint/E2/dorga/` + `e2_summary.json` |
| **E3** 양성대조 | 한 코호트(2024)를 정합 두 반쪽 H1·H2. H2에 오프셋 β ∈ {0, 0.25, 0.5, 1.0} 주입. reps 42·1·2. 복원곡선(β→δ) 기울기·절편 | `checkpoint/E3/` + `e3_summary.json` |
| **E4** 음성대조 | β=0, 학습량 비대칭 train_frac ∈ {1.0, 0.5, 0.25}. reps 42·1·2. δ_spurious 누출 | `checkpoint/E4/` + `e4_summary.json` |
| **부록 B** in-domain raw | E2와 동일하나 **raw**(전 환자, 정합 전). Δg_raw. 수축분 = Δg_raw − Δg_matched | `checkpoint/AppendixB/dorga/` + `appb_summary.json` |

판정: δ_corr = δ_obs + Δg/2 (E3 matched). `B_summary/b1_summary.py` → `A1_verdict.json` (+CI).

## 5. L 분석 (학습 무관: L1·L2·L4)

| 레벨 | 계산 | 산출 |
|---|---|---|
| **L1** 구조·분포 | 환자·영상수, 시퀀스길이, 평균등급, 등급 0~4 비율 (labels.csv) | `Result/Phase1/A_analysis/a1_structure.csv` |
| **L2** 라벨-영상 | ROI **6-특징**[1차 4(mean·median·uniformity·entropy) + GLRLM SRE + GLSZM SAE]. 그레이 Ng=16, ROI [min,max] 양자화, GLRLM 4방향 합산, GLSZM 8-연결. 인접등급 방향무관 AUC=max(a,1−a). 경계 3→4, 귀무 max 순열 1,500회 보정 | `Result/Phase1/A_analysis/a2_label_image_{features,auc}.csv` |
| **L3** 방향반전 분해 | E1 예측 → δ·γ (모델 의존) | (= E1) |
| **L4** 재현성 | 재측정 2세션 → 자기일치 ACC, weighted κ(quadratic), Wilcoxon | `Result/Phase1/A_analysis/a3_reproducibility.csv` |

## 6. 평가·통계

| 항목 | 값 |
|---|---|
| 지표 | ACC, MAE, bias(pred−label) |
| bias 단위 | 환자 (연속 영상 상관 → 환자를 독립 단위) |
| CI | 환자 단위 부트스트랩, n=5000, 95% |
| 방향반전 | δ=(rev−fwd)/2, γ=(rev+fwd)/2 |
| A1 판정 | Δg CI가 0 배제 시 A1 위반 |

## 7. 기존 checkpoint 예측에서 재현

재현 완료한 a1~a3 CSV는 유지한다. a4~a8은 checkpoint의 기존 test_preds.npz만 읽는다.
모델·가중치 로딩, 학습, 재추론, 예측 NPZ 생성은 하지 않는다. 필요한 NPZ가 없으면 경로를 표시하고 중단한다.

| 코드 | 입력 | 출력 |
|---|---|---|
| a4_cross | checkpoint/E1/dorga, 10개 예측 | Result/Phase1/A_analysis/a4_cross.csv |
| a5_indomain_matched | checkpoint/E2/dorga, 30개 예측 | Result/Phase1/A_analysis/a5_indomain_matched.csv |
| a6_positive_control | checkpoint/E3, 24개 예측 | Result/Phase1/A_analysis/a6_positive_control.csv |
| a7_negative_control | checkpoint/E4, 18개 예측 | Result/Phase1/A_analysis/a7_negative_control.csv |
| a8_indomain_raw | checkpoint/AppendixB/dorga, 30개 예측 | Result/Phase1/A_analysis/a8_indomain_raw.csv |
| b1_summary | a4~a8 CSV + checkpoint/E1 예측 | Result/Phase1/B_summary/b1_summary.csv |
| c1_confusion | checkpoint/E1 예측 + b1 CSV | Result/Phase1/C_figures/c1_confusion.png |

각 분석 CSV에는 개별 실행 지표와 집계 지표를 함께 담는다. cohort 환자 bias 벡터도 값별로 저장한다.
source는 비교할 Result_old의 상대 경로, json_path는 그 문서의 항목 위치다.
arm·direction·year·fold·seed·condition·roi·metric 열로 항목을 구분한다.
value는 타입과 정밀도를 보존한 JSON 스칼라 표기이며, 새 JSON 파일을 쓰는 것은 아니다.

```bash
# a4~a8 집계 → b1 → c1 → 비교. a1~a3은 재실행하지 않는다.
python Experiment/Phase1/compare.py --run

# 이미 생성한 CSV를 기존 JSON/CSV와 비교만 한다.
python Experiment/Phase1/compare.py
```

비교는 새 CSV만 읽으며 이전에 생성된 Result 하위의 JSON·NPZ는 사용하지 않는다.
Result_old는 비교 기준으로만 읽는다. 지표 생성 코드에서는 읽지 않는다.
개별 results.json의 ACC·MAE·bias·환자 bias/CI·ROI 지표와 cohort/summary/verdict를 대응 비교한다.
학습 history 및 NPZ로 확인할 수 없는 n_train/final_epoch/best_val_epoch/best_val_mae/tail_bias/mrm은 제외를 명시한다.
누락·중복·수치 차이·기준에 없는 항목은 통과시키지 않는다.

core.py·train.py는 기존 학습 코드로 보존한다. common.py는 집계 공통 함수와 복원 코드가 공유한다.
a9~a11 및 b2는 별도 분석이며 위 실행 목록에 포함하지 않는다.
