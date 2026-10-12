# -*- coding: utf-8 -*-
"""Checkpoint prediction aggregation, CSV serialization, and original statistical helpers."""
from __future__ import annotations
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

# ── Windows cp949 콘솔에서 base 스크립트의 유니코드 출력(✔ ━ 등) 크래시 방지 ──
#   base 로드(=wandb import) 전에 콘솔 캡처를 끄고 stdout 을 utf-8 로 재설정한다.
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

# ── 학습 코어: Private Experiment/A_analysis/core.py (舊 dorga_train_2026to2024.py 자립 대체) ──
_HERE = os.path.dirname(os.path.abspath(__file__))     # Experiment/E
_EXP_ROOT = os.path.dirname(_HERE)                     # Experiment
_REPO = os.path.dirname(_EXP_ROOT)                     # Private repo 루트
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# Core is loaded only by legacy model code, never by checkpoint CSV aggregation.
class _LazyCore:
    def __getattr__(self, name):
        from Experiment.A_analysis import core
        return getattr(core, name)
B = _LazyCore()
EARLYSTOP = 10
ROI = ["RT", "LT", "RB", "LB"]
C = 5
MANIFEST = str(Path(_REPO) / "CXR/Merged/labels.csv")
A1_OUT = Path(_REPO) / "checkpoint"
RESULT_OUT = Path(_REPO) / "Result"
_SPLITTERS = {}


def _prep(df):
    """core labels.csv(patient_id 24_/26_ 접두어) → 구 splitter 호환 year·patient 파생."""
    if "year" not in df.columns:
        df = df.copy()
        df["patient"] = df["patient_id"]
        df["year"] = df["patient_id"].astype(str).str[:2].map({"24": 2024, "26": 2026})
    return df

def _dispatch(df, mode, seed):
    df = _prep(df)
    if mode in _SPLITTERS:
        return _SPLITTERS[mode](df, seed)
    return B.make_split(df, mode, seed)



def register(mode: str, fn):
    """mode 이름에 커스텀 splitter 를 건다. run_one(mode,...) 가 이걸 타게 된다."""
    _SPLITTERS[mode] = fn


# ═══════════════════════════════════════════════════════════════
# split helper — 환자 단위 (누수 방지: 한 환자는 한 split 에만)
# ═══════════════════════════════════════════════════════════════
def _patients_of(df, year):
    return np.asarray(df.loc[df.year == year, "patient"].unique(), dtype=object)

def _mark(df, train_pat, val_pat, test_pat, year):
    """df 에서 해당 year 행만 남기고 split 을 채운다."""
    sub = df[df.year == year].copy()
    tr, va, te = set(train_pat), set(val_pat), set(test_pat)
    sub["split"] = np.where(sub.patient.isin(te), "test",
                    np.where(sub.patient.isin(va), "val",
                    np.where(sub.patient.isin(tr), "train", None)))
    return sub[sub.split.notna()].copy()

def kfold_patient_folds(df, year, n_folds, seed):
    """year 코호트 환자를 n_folds 로 분할 → [(fold_idx, test_patients_array), ...]"""
    pats = _patients_of(df, year)
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    return [(k, pats[k::n_folds]) for k in range(n_folds)]   # 인터리브 분할 (분포 균등)


# ═══════════════════════════════════════════════════════════════
# 라벨 주입 (E3 양성대조) — 순서형 오프셋, 천장/바닥 clip
#   b_half 오프셋을 '그 half 에 속한 모든 행'에 일관 적용한다(학습·평가 무관).
#   분수 β 는 stochastic rounding 으로 표현.  clip(0,C-1) 이라 극단 등급에서
#   복원 기울기가 감쇠하는데, 그 감쇠 자체가 E3 복원곡선으로 측정된다.
# ═══════════════════════════════════════════════════════════════
def inject_offset(df, mask, beta, seed):
    """mask(bool Series/array) 인 행의 ROI 라벨에 오프셋 beta 를 순서형으로 적용."""
    if beta == 0:
        return df
    rng = np.random.default_rng(seed)
    df = df.copy()
    idx = np.where(np.asarray(mask))[0]
    for col in ROI:
        g = df[col].to_numpy().astype(float)
        base = g[idx] + beta
        lo = np.floor(base)
        up = (rng.random(len(idx)) < (base - lo)).astype(float)
        g[idx] = np.clip(lo + up, 0, C - 1)
        df[col] = g.astype(int)
    return df


# ═══════════════════════════════════════════════════════════════
# 분포 정합 (E2) — 두 코호트를 공통 등급분포·공통 크기로 서브샘플
#   ROI-평균 등급을 환자 대리로 쓰고, 등급 히스토그램을 맞춰 환자를 표집한다.
# ═══════════════════════════════════════════════════════════════
def match_two_cohorts(df, seed, n_bins=5):
    """2024·2026 을 환자 단위로 공통 분포·공통 환자수로 맞춘 부분집합 반환."""
    rng = np.random.default_rng(seed)
    def pat_grade(year):
        sub = df[df.year == year]
        return sub.groupby("patient")[ROI].apply(lambda x: x.to_numpy().mean())
    g24, g26 = pat_grade(2024), pat_grade(2026)
    edges = np.linspace(0, C - 1, n_bins + 1)
    b24, b26 = np.digitize(g24.values, edges), np.digitize(g26.values, edges)
    keep24, keep26 = [], []
    for b in range(1, n_bins + 2):
        p24, p26 = g24.index[b24 == b], g26.index[b26 == b]
        k = min(len(p24), len(p26))                     # 각 bin 에서 min 만큼만
        if k == 0: continue
        keep24 += list(rng.choice(p24, k, replace=False))
        keep26 += list(rng.choice(p26, k, replace=False))
    return set(keep24), set(keep26)


# ═══════════════════════════════════════════════════════════════
# 실행 래퍼 — 등록된 mode 로 B.run_one 호출
# ═══════════════════════════════════════════════════════════════
def run_arm(mode, splitter, seed, epochs, root=None, model="dorga", arm=None, skip_existing=True):
    """Read saved predictions only. Training and inference are not allowed here."""
    if not skip_existing:
        raise ValueError("CSV aggregation reads existing test_preds.npz only")
    path = Path(root) / (arm or f"{mode}_s{seed}") / "test_preds.npz"
    data = load_predictions(path)
    return dict(arm_metrics(data, mode, seed), npz=str(path))


# ═══════════════════════════════════════════════════════════════
# per-patient bias & δ/s 분해 + bootstrap CI
# ═══════════════════════════════════════════════════════════════
def per_patient_bias(npz_path, roi=None):
    """test_preds.npz → 환자별 bias 배열. roi=None 이면 4-ROI 평균, 아니면 해당 ROI."""
    d = np.load(npz_path, allow_pickle=True)
    P, Y, pats = d["preds"], d["labels"], np.asarray(d["patients"])
    if roi is None:
        e = (P - Y).mean(axis=1)
    else:
        j = ROI.index(roi); e = (P[:, j] - Y[:, j]).astype(float)
    uniq = np.unique(pats)
    return np.array([e[pats == u].mean() for u in uniq])

def mean_ci(arr, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return float(arr.mean()), tuple(np.percentile(boots, [2.5, 97.5]))

def diff_ci(a, b, n_boot=5000, seed=0):
    """독립 두 표본 평균차 a-b 의 bootstrap CI (fwd/rev 는 test 셋이 달라 비대응)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    boots = a[ia].mean(1) - b[ib].mean(1)
    return float(a.mean() - b.mean()), tuple(np.percentile(boots, [2.5, 97.5]))

def decompose(fwd_npz, rev_npz, roi=None, n_boot=5000, seed=0):
    """δ=(rev-fwd)/2 (라벨 성분), s=(rev+fwd)/2 (모델 성분) + 각 CI. fwd/rev 독립 bootstrap."""
    f = per_patient_bias(fwd_npz, roi)
    r = per_patient_bias(rev_npz, roi)
    rng = np.random.default_rng(seed)
    bf = f[rng.integers(0, len(f), (n_boot, len(f)))].mean(1)
    br = r[rng.integers(0, len(r), (n_boot, len(r)))].mean(1)
    delta, s = (br - bf) / 2, (br + bf) / 2
    return dict(fwd=float(f.mean()), rev=float(r.mean()),
                delta=float((r.mean() - f.mean()) / 2), delta_ci=tuple(np.percentile(delta, [2.5, 97.5])),
                s=float((r.mean() + f.mean()) / 2),     s_ci=tuple(np.percentile(s,     [2.5, 97.5])))

def sig(ci):     # CI 가 0 을 배제하는가
    return ci[0] * ci[1] > 0


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════
# in-domain Δg 공용 코어 — E2(raw 전 환자)·E3(matched 정합)가 공유
#   in-domain(c→c) 은 라벨 오프셋 b_c 상쇄 → bias=g_c(순수 모델오차). Δg=g24−g26.
# ═══════════════════════════════════════════════════════════════
def indomain_fold_splitter(year, test_pat, seed, keep_pat=None):
    """year in-domain splitter. keep_pat 주면 그 정합 부분집합 안에서만 train/val/test."""
    def _fn(df, _seed):
        keep = set(keep_pat) if keep_pat is not None else None
        te = set(test_pat)
        pool = np.array([p for p in _patients_of(df, year)
                         if p not in te and (keep is None or p in keep)], dtype=object)
        rng = np.random.default_rng(seed); rng.shuffle(pool)
        n_val = max(2, int(round(len(pool) * B.VAL_FRAC)))     # 8:2 (core VAL_FRAC 단일 소스)
        return _mark(df, pool[n_val:], pool[:n_val], test_pat, year), year, year
    return _fn


def run_cohort(year, folds, seed, init_seeds, root, keep_pat=None, tag="raw",
               skip_existing=True, only_folds=None):
    """한 코호트 K-fold(전 환자 1회 test) × init_seeds → overall·ROI별 환자 bias 벡터.
    skip_existing(기본 True) 은 arm 별로 run_arm 에 전달(이미 학습된 fold·seed 는 건너뜀).
    only_folds(list, 기본 None=전체) 로 학습·집계할 fold 인덱스를 제한한다(부분 실행)."""
    df = _prep(pd.read_csv(MANIFEST))
    if keep_pat is None:
        fold_list = kfold_patient_folds(df, year, folds, seed)
    else:
        pats = np.array([p for p in _patients_of(df, year) if p in set(keep_pat)], dtype=object)
        rng = np.random.default_rng(seed); rng.shuffle(pats)
        fold_list = [(k, pats[k::folds]) for k in range(folds)]
    if only_folds is not None:
        want = set(only_folds)
        fold_list = [(k, tp) for (k, tp) in fold_list if k in want]
    keys = ["overall"] + list(ROI)
    bias_pat = {k: {} for k in keys}
    Pall, Yall = [], []
    for k, test_pat in fold_list:
        for isd in init_seeds:
            mode = f"{tag}_{year}_fold{k}"             # seed 는 core 가 _s{isd} 로 붙임
            res = run_arm(mode, indomain_fold_splitter(year, test_pat, seed, keep_pat), isd, B.EPOCHS, root,
                          skip_existing=skip_existing)
            d = np.load(res["npz"], allow_pickle=True)
            P, Y, pats_te = d["preds"], d["labels"], np.asarray(d["patients"])
            Pall.append(P); Yall.append(Y)
            for key in keys:
                e = (P - Y).mean(axis=1) if key == "overall" \
                    else (P[:, ROI.index(key)] - Y[:, ROI.index(key)]).astype(float)
                for u in np.unique(pats_te):
                    bias_pat[key].setdefault(u, []).append(float(e[pats_te == u].mean()))
    P, Y = np.vstack(Pall), np.vstack(Yall)
    out = {"year": int(year), "n_pat": len(bias_pat["overall"]),
           "acc": float((P == Y).mean()), "mae": float(np.abs(P - Y).mean())}
    for key in keys:
        vec = np.array([np.mean(bias_pat[key][u]) for u in sorted(bias_pat[key])])
        g, ci = mean_ci(vec, seed=seed)
        out[key] = dict(g=g, ci=list(ci), vec=vec.tolist())
    return out


def a1_test(out24, out26, keys):
    """Δg=g24−g26 (독립 두 코호트 환자 bias 벡터 평균차) + bootstrap CI + reject_A1."""
    a1 = {}
    for key in keys:
        a, b = np.array(out24[key]["vec"]), np.array(out26[key]["vec"])
        dg, dci = diff_ci(a, b, seed=0)
        a1[key] = dict(delta_g=dg, ci=list(dci), reject_A1=bool(sig(dci)))
    return a1


# Restored from npj a1_common.py and original dorga_train_2026to2024.py.
def legacy_manifest():
    df = _prep(pd.read_csv(MANIFEST))
    # Merged labels rewrites image_path; recover original raw paths by uid.
    source = pd.read_csv(Path(_REPO) / "CXR/unified_labels.csv")
    paths = source.set_index("uid", verify_integrity=True)["image_path"]
    df["image_path"] = df["uid"].map(paths)
    if df["image_path"].isna().any():
        raise ValueError("Merged labels contain uid absent from unified_labels.csv")
    return df


def legacy_raw_path(row):
    path = str(row["image_path"]).replace("\\", "/")
    for prefix in ("/shared/home/mai/JeongGeon/Private/CXR/", "/shared/home/mai/JeongGeon/QC_Drift/CXR/"):
        if path.startswith(prefix):
            path = str(Path(_REPO) / "CXR" / path[len(prefix):])
    local_prefix = "D:/InhaUH_CXR/2026.05 CXRs/Dataset/"
    if path.startswith(local_prefix):
        path = str(Path(_REPO) / "CXR/2026 CXR" / path[len(local_prefix):])
    candidate = Path(path)
    if int(row["year"]) == 2026 and candidate.parent.name.startswith("NPZ_IMAGE"):
        candidate = Path(raw_path_from_npz(str(candidate)))
    if candidate.is_file():
        return str(candidate)
    if int(row["year"]) == 2024:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = Path(_REPO) / "CXR/Merged/images_original" / (str(row["uid"]) + ext)
            if candidate.is_file(): return str(candidate)
    raise FileNotFoundError(f"Original image missing for {row.get('uid')}: {path}")


def raw_path_from_npz(path):
    import re
    p = Path(path)
    match = re.match(r"\d+_([A-Za-z]+\d+)\.png", p.name)
    if match is None: raise ValueError(f"Unexpected NPZ image name: {path}")
    return str(p.parent.parent / p.parent.name.replace("NPZ_IMAGE", "RAW_IMAGE") / (match.group(1).upper()+".jpg"))


def build_2026_cache(raw_paths):
    # Original algorithm: same theta and STN.transform for image AND mask.
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from Model.SegSTN.seg import LungSegmenter
    from Model.SegSTN.stn import STN
    seg = LungSegmenter(str(Path(_REPO)/"Model/SegSTN/weights/finetuned_9601.pt"), device=str(B.DEVICE))
    stn = STN(in_shape=(1,512,512)).to(B.DEVICE)
    stn.load_state_dict(torch.load(Path(_REPO)/"Model/SegSTN/weights/stn_weights.pth", map_location=B.DEVICE, weights_only=False))
    stn.eval()
    cache = {}
    with torch.no_grad():
        for p in raw_paths:
            arr = np.array(Image.open(p).convert("L")).astype(np.float32)/255.0
            t = torch.from_numpy(arr)[None,None]
            im1024 = F.interpolate(t,size=(1024,1024),mode="bilinear",align_corners=False)
            im512 = F.interpolate(t,size=(512,512),mode="bilinear",align_corners=False).to(B.DEVICE)
            mask1024 = seg(im1024.to(B.DEVICE))
            mask512 = F.interpolate(mask1024.float(),size=(512,512),mode="nearest").to(B.DEVICE)
            theta = stn(mask512)
            image = STN.transform(im512,theta).cpu()
            mask = STN.transform(mask512,theta).cpu()
            cache[p] = ((image[0,0].numpy()*255).clip(0,255).astype(np.uint8),
                        ((mask[0,0].numpy()>0.5)*255).astype(np.uint8))
    del seg, stn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return cache


def build_full_2026_cache():
    df = legacy_manifest()
    return build_2026_cache(sorted({legacy_raw_path(row) for _,row in df[df.year==2026].iterrows()}))

# CSV rows retain the exact baseline document and nested metric identity.
CSV_COLUMNS = ['source', 'level', 'arm', 'direction', 'year', 'fold', 'seed',
               'condition', 'roi', 'metric', 'json_path', 'value']
STAGES = {'a4_cross': 'E1', 'a5_indomain_matched': 'E2',
          'a6_positive_control': 'E3', 'a7_negative_control': 'E4',
          'a8_indomain_raw': 'AppendixB'}


def flatten(obj, path=()):
    if isinstance(obj, dict) and obj:
        for key, value in obj.items():
            yield from flatten(value, path + (key,))
    elif isinstance(obj, (list, tuple)) and obj:
        for i, value in enumerate(obj):
            yield from flatten(value, path + (i,))
    else:
        yield path, obj


def write_documents(path, documents):
    import csv
    import re
    rows = []
    for source, obj in documents.items():
        arm = Path(source).parent.name if source.endswith('/results.json') else ''
        seed = re.search(r'_s(\d+)$', arm)
        split = re.search(r'(?:fold|split)(\d+)', arm)
        direction = ('fwd' if '24to26' in arm or '_fwd_' in arm else
                     'rev' if '26to24' in arm or '_rev_' in arm else '')
        condition = re.search(r'(beta|frac)([\d.]+)', arm)
        for tokens, value in flatten(obj):
            roi = next((t for t in tokens if t in ['overall'] + ROI), '')
            year = next((str(t) for t in tokens if str(t) in ('2024', '2026')), '')
            if not year:
                year = str(obj.get('test_year', obj.get('year', ''))) if isinstance(obj, dict) else ''
            summary_condition = ''
            if not arm and len(tokens) > 1 and tokens[0] == 'curve':
                item = obj['curve'][tokens[1]]
                summary_condition = (f"beta{item['beta']}" if 'beta' in item else
                                     f"frac{item['train_frac']}" if 'train_frac' in item else '')
            rows.append(dict(source=source, level='arm' if arm else 'summary', arm=arm,
                             direction=direction, year=year, fold=split.group(1) if split else '',
                             seed=seed.group(1) if seed else (obj.get('seed', '') if arm else ''),
                             condition=condition.group(0) if condition else summary_condition, roi=roi,
                             metric='/'.join(map(str, tokens)),
                             json_path=json.dumps(tokens, ensure_ascii=False),
                             value=json.dumps(value, ensure_ascii=False)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)
    print(f'[save] {path} ({len(documents)} documents, {len(rows)} metrics)')


def read_documents(path):
    import csv
    documents, seen = {}, set()
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f'Unexpected aggregation CSV columns: {path}')
        for row in reader:
            source = row['source']; tokens = json.loads(row['json_path'])
            key = (source, tuple(tokens))
            if key in seen: raise ValueError(f'Duplicate CSV metric: {key}')
            seen.add(key)
            value = json.loads(row['value'])
            if not tokens:
                documents[source] = value
                continue
            node = documents.setdefault(source, [] if isinstance(tokens[0], int) else {})
            for i, token in enumerate(tokens):
                last = i == len(tokens)-1
                child = value if last else ([] if isinstance(tokens[i+1], int) else {})
                if isinstance(node, list):
                    while len(node) <= token: node.append(None)
                    if last or node[token] is None: node[token] = child
                    node = node[token]
                else:
                    if last: node[token] = child
                    else: node = node.setdefault(token, child)
    return documents


def load_predictions(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f'Missing saved predictions: {path}')
    with np.load(path, allow_pickle=True) as data:
        P, Y, patients = data['preds'], data['labels'], np.asarray(data['patients']).astype(str)
    if P.ndim != 2 or P.shape != Y.shape or P.shape[1] != len(ROI) or len(P) != len(patients) or not len(P):
        raise ValueError(f'Invalid prediction shapes: {path}')
    if not np.isfinite(P).all() or not np.isfinite(Y).all():
        raise ValueError(f'Nonfinite predictions: {path}')
    return P, Y, patients


def arm_metrics(data, mode, seed):
    P, Y, patients = data
    errors = (P-Y).mean(axis=1)
    vec = np.array([errors[patients == u].mean() for u in np.unique(patients)])
    mean, ci = mean_ci(vec, seed=seed)
    return dict(model='dorga', mode=mode, seed=seed, roi_order=ROI, n_test=len(P),
                acc=float((P == Y).mean()), mae=float(np.abs(P-Y).mean()), bias=float((P-Y).mean()),
                pat_bias=mean, ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                per_roi={r: dict(acc=float((P[:,i] == Y[:,i]).mean()),
                                 mae=float(np.abs(P[:,i]-Y[:,i]).mean()),
                                 bias=float((P[:,i]-Y[:,i]).mean())) for i,r in enumerate(ROI)})


def expected_arms(stage):
    if stage == 'a4_cross':
        return [(f'dorga/{mode}_split{k}', f'{mode}_split{k}', k)
                for mode in ('24to26', '26to24') for k in range(1,6)]
    if stage in ('a5_indomain_matched', 'a8_indomain_raw'):
        tag = 'matched' if stage == 'a5_indomain_matched' else 'raw'
        return [(f'dorga/{tag}_{year}_fold{k}_s{seed}', f'{tag}_{year}_fold{k}', seed)
                for year in (2024,2026) for k in range(5) for seed in (42,1,2)]
    positive = stage == 'a6_positive_control'
    prefix, knob, values = ('E3','beta',(0.0,0.25,0.5,1.0)) if positive else ('E4','frac',(1.0,0.5,0.25))
    return [(f'{prefix}_2024_{knob}{v}_{direction}_s{seed}', f'{prefix}_2024_{knob}{v}_{direction}', seed)
            for v in values for seed in (42,1,2) for direction in ('fwd','rev')]


def require_predictions(stages):
    missing = [str(A1_OUT/STAGES[stage]/arm/'test_preds.npz')
               for stage in stages for arm,_,_ in expected_arms(stage)
               if not (A1_OUT/STAGES[stage]/arm/'test_preds.npz').is_file()]
    if missing: raise FileNotFoundError('Missing saved predictions (no training/inference fallback):\n'+'\n'.join(missing))


def cohort_from_predictions(arrays, year):
    keys = ['overall'] + ROI
    biases = {key: {} for key in keys}
    ps, ys = [], []
    for P,Y,patients in arrays:
        ps.append(P); ys.append(Y)
        for key in keys:
            e = (P-Y).mean(axis=1) if key == 'overall' else (P[:,ROI.index(key)]-Y[:,ROI.index(key)]).astype(float)
            for patient in np.unique(patients):
                biases[key].setdefault(patient, []).append(float(e[patients == patient].mean()))
    P,Y=np.vstack(ps),np.vstack(ys)
    out=dict(year=year,n_pat=len(biases['overall']),acc=float((P==Y).mean()),mae=float(np.abs(P-Y).mean()))
    for key in keys:
        vec=np.array([np.mean(biases[key][p]) for p in sorted(biases[key])])
        g,ci=mean_ci(vec,seed=0)
        out[key]=dict(g=g,ci=list(ci),vec=vec.tolist())
    return out


def aggregate_stage(stage):
    require_predictions([stage])
    folder=STAGES[stage]; documents={}; arrays={}
    for arm,mode,seed in expected_arms(stage):
        data=load_predictions(A1_OUT/folder/arm/'test_preds.npz')
        arrays[arm]=data
        metrics=arm_metrics(data,mode,seed)
        if stage == 'a4_cross':
            ty,ey=(2024,2026) if mode.startswith('24to26') else (2026,2024)
        else:
            ty=ey=2026 if '2026' in mode else 2024
        metrics.update(train_year=ty,test_year=ey)
        documents[f'{folder}/{arm}/results.json']=metrics
    if stage in ('a5_indomain_matched','a8_indomain_raw'):
        matched=stage=='a5_indomain_matched';tag='matched' if matched else 'raw';prefix='e2' if matched else 'appb'
        cohorts={str(year):cohort_from_predictions([arrays[a] for a,_,_ in expected_arms(stage) if f'{year}_fold' in a],year)
                 for year in (2024,2026)}
        for year,cohort in cohorts.items(): documents[f'{folder}/{prefix}_cohort_{year}.json']=cohort
        group=dict(cohorts)
        if matched: group['n']={y:c['n_pat'] for y,c in cohorts.items()}
        documents[f'{folder}/{prefix}_summary.json']={tag:group,
            'A1_test_matched' if matched else 'A1_test':a1_test(cohorts['2024'],cohorts['2026'],['overall']+ROI)}
    elif stage in ('a6_positive_control','a7_negative_control'):
        positive=stage=='a6_positive_control';knob='beta' if positive else 'frac'
        values=(0.0,0.25,0.5,1.0) if positive else (1.0,0.5,0.25)
        curve=[]
        for value in values:
            ds={k:[] for k in ['overall']+ROI};ss={k:[] for k in ds}
            for seed in (42,1,2):
                for key in ds:
                    f=A1_OUT/folder/f'{folder}_2024_{knob}{value}_fwd_s{seed}'/'test_preds.npz'
                    r=A1_OUT/folder/f'{folder}_2024_{knob}{value}_rev_s{seed}'/'test_preds.npz'
                    dec=decompose(f,r,roi=None if key=='overall' else key,seed=seed)
                    ds[key].append(dec['delta']);ss[key].append(dec['s'])
            if positive:
                item=dict(beta=value,delta_mean=float(np.mean(ds['overall'])),delta_sd=float(np.std(ds['overall'])),
                          s_mean=float(np.mean(ss['overall'])),s_sd=float(np.std(ss['overall'])),deltas=ds['overall'],ss=ss['overall'],
                          delta_roi={k:float(np.mean(ds[k])) for k in ROI},s_roi={k:float(np.mean(ss[k])) for k in ROI})
            else:
                item=dict(train_frac=value,delta_spurious=float(np.mean(ds['overall'])),delta_sd=float(np.std(ds['overall'])),
                          s_mean=float(np.mean(ss['overall'])),deltas=ds['overall'])
            curve.append(item)
        # Test patients in reverse are H1, in forward are H2; derive sizes from saved predictions.
        first=values[0]
        h1=len(np.unique(arrays[f'{folder}_2024_{knob}{first}_rev_s42'][2]))
        h2=len(np.unique(arrays[f'{folder}_2024_{knob}{first}_fwd_s42'][2]))
        summary=dict(year=2024,n_H1=h1,n_H2=h2,curve=curve)
        if positive:
            slope,intercept=np.polyfit(values,[c['delta_mean'] for c in curve],1)
            summary.update(recovery_slope=float(slope),recovery_intercept=float(intercept),s_grand=float(np.mean([c['s_mean'] for c in curve])),
                           verdict='기울기≈1·절편≈0 이면 추정기 무편향 (A1-참 조건 검증됨)')
        else:
            summary['note']='참 δ=0. frac↓ 일수록 δ_spurious 가 0 에서 벗어나면 A1 위반이 δ 로 누출됨을 뜻함.'
        documents[f'{folder}/{"e3" if positive else "e4"}_summary.json']=summary
    write_documents(RESULT_OUT/'A_analysis'/f'{stage}.csv',documents)
    return documents
