# -*- coding: utf-8 -*-
r"""
E1 confusion matrix 그림 — 방향(fwd/rev) × ROI 2×4 그리드.

e1_cross.py 가 남긴 split 전부를 pool(summary.py 의 환자 부트스트랩과 동일 소스)해
ROI별 5×5 혼동행렬을 그린다. 색=행 정규화 비율(recall), 숫자=count.
패널 제목의 bias·ACC 는 core._metrics 와 같은 정의: (P−Y).mean(), (P==Y).mean().

ROI 헤더는 **정합 보정 후** 값 δ_corr = δ_obs + Δγ_matched/2 와 환자 부트스트랩
95% CI (A1_verdict.json, summary.build_verdict 산출). CI 가 0 을 배제하는 ROI 는
빨간 테두리로 강조한다 — 보정 후에도 방향 반전이 남는 문제 영역.

입력:  checkpoint/E1/dorga/{24to26,26to24}_split*/test_preds.npz  (preds·labels)
       Result/B_summary/b1_summary.csv  (delta_corrected: δ_corr·CI)
산출:  Result/C_figures/c1_confusion/e1_confusion_quadrant.png

실행:  python Experiment/C_figures/c1_confusion.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _s in (sys.stdout, sys.stderr):              # Windows cp949 콘솔 유니코드 크래시 방지
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


_HERE = Path(__file__).resolve().parent
_REPO = next(p for p in (_HERE, *_HERE.parents) if (p / "Experiment").is_dir())
CKPT_E1   = _REPO / "checkpoint" / "E1" / "dorga"    # 입력: {mode}_split{k}/test_preds.npz
VERDICT   = _REPO / "Result" / "B_summary" / "b1_summary.csv"     # 입력: δ_corr + CI (보정 후)
RESULT_E1 = _REPO / "Result" / "C_figures" / "c1_confusion"
ROI = ["RT", "LT", "RB", "LB"]                    # core.ROI = preds 열 순서
NGRADE = 5
MODES = [("24to26", "2024→2026 (2026 라벨)"),      # e1_cross.DIRS 의 tag
         ("26to24", "2026→2024 (2024 라벨)")]
FONT_PATH = Path("/shared/home/mai/JeongGeon/Fonts/SeoulNamsanL.ttf")


def set_font():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT_PATH))
    name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["font.family"] = [name, "DejaVu Sans"]     # δ 등 없는 글자만 폴백
    plt.rcParams["axes.unicode_minus"] = False


def load_pool(mode):
    """checkpoint/E1/dorga/{mode}_split*/test_preds.npz 를 전부 pool → (P, Y)."""
    files = sorted(CKPT_E1.glob(f"{mode}_split*/test_preds.npz"))
    if not files:
        return None, None, files
    P = np.concatenate([np.load(f)["preds"] for f in files]).astype(int)
    Y = np.concatenate([np.load(f)["labels"] for f in files]).astype(int)
    return P, Y, files


def load_delta():
    """b1_summary.csv → {roi: (δ_corr, lo, hi, CI가 0을 배제하는가)}."""
    sys.path.insert(0, str(_REPO))
    from Experiment.A_analysis.common import read_documents
    d = read_documents(VERDICT)["A1_verdict.json"]["delta_corrected"]
    out = {}
    for roi in ROI:
        r = d[roi]
        lo, hi = r["ci"]
        out[roi] = (r["delta_corr"], lo, hi, lo > 0 or hi < 0)
    return out


def confusion(y, p):
    m = np.zeros((NGRADE, NGRADE), dtype=int)     # 행=실제, 열=예측
    np.add.at(m, (y, p), 1)
    return m


def panel(ax, y, p, roi, delta, top, ylabel, xlabel):
    m = confusion(y, p)
    row = m.sum(axis=1, keepdims=True)
    recall = np.divide(m, row, out=np.zeros(m.shape, float), where=row > 0)

    ax.imshow(recall, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower")
    for i in range(NGRADE):
        for j in range(NGRADE):
            if m[i, j]:                            # 0 은 빈칸
                ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if recall[i, j] > 0.55 else "#1a1a1a")

    dc, lo, hi, hot = delta
    stat = f"bias(pred-true)={float((p - y).mean()):+.2f}   ACC={float((p == y).mean()):.2f}"
    if top:
        head = f"{roi}   δ_corr={dc:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
        if hot:
            head += "   ← 0 배제"
        ax.set_title(f"{head}\n{stat}", fontsize=10.5,
                     color="crimson" if hot else "black",
                     fontweight="bold" if hot else "normal")
    else:
        ax.set_title(f"{roi}   {stat}", fontsize=10.5,
                     color="crimson" if hot else "black",
                     fontweight="bold" if hot else "normal")

    if hot:                                        # 보정 후에도 반전이 남는 문제 영역
        for s in ax.spines.values():
            s.set(edgecolor="crimson", linewidth=2.6, zorder=5)

    ax.set_xticks(range(NGRADE)); ax.set_yticks(range(NGRADE))
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, NGRADE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NGRADE, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    if xlabel: ax.set_xlabel("예측 등급", fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    else:      ax.set_yticklabels([])


def main():
    set_font()
    if not VERDICT.exists():
        print(f"[중단] 보정 후 값 없음: {VERDICT} — summary.py 실행 후")
        return
    delta = load_delta()
    hot = [r for r in ROI if delta[r][3]]
    print(f"[load] {VERDICT.name}  δ_corr CI 0배제: {hot if hot else '없음'}")

    pools = {}
    for mode, _ in MODES:
        P, Y, files = load_pool(mode)
        if P is None:
            print(f"[중단] E1 npz 없음: {CKPT_E1}/{mode}_split* — e1_cross.py 실행 후")
            return
        print(f"[load] {mode}  npz {len(files)}개  n={len(P)} (ROI당)")
        pools[mode] = (P, Y)

    fig, axes = plt.subplots(2, 4, figsize=(19.5, 10.8))
    for r, (mode, ylabel) in enumerate(MODES):
        P, Y = pools[mode]
        for c, roi in enumerate(ROI):
            panel(axes[r, c], Y[:, c], P[:, c], roi, delta[roi], r == 0,
                  ylabel if c == 0 else None, r == len(MODES) - 1)

    fig.suptitle("E1 ROI별 confusion matrix — 방향 반전 (색=행 정규화 recall, 숫자=count)",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012,
             "헤더 δ_corr = δ_obs + Δγ_matched/2 (정합 보정 후, 환자 부트스트랩 95% CI)   ·   "
             "빨강 = CI 가 0 을 배제 → 보정 후에도 방향 반전이 남는 문제 영역",
             ha="center", fontsize=10.5, color="crimson")
    fig.tight_layout(rect=(0, 0.028, 0.93, 0.96))
    cax = fig.add_axes((0.945, 0.15, 0.011, 0.7))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.get_cmap("Blues")),
                      cax=cax)
    cb.set_label("행 정규화 비율(recall)", fontsize=10, labelpad=10)

    RESULT_E1.mkdir(parents=True, exist_ok=True)
    png = RESULT_E1 / "e1_confusion_quadrant.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {png}")


if __name__ == "__main__":
    main()
