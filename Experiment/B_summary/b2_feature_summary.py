# -*- coding: utf-8 -*-
"""Aggregate a10/a11 CSVs using their original l3_explain calculation."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.A_analysis.a10_features import l3_explain


def main():
    sources = {
        "full": ROOT / "Result/A_analysis/a10_features/roi_features_full.csv",
        "matched": ROOT / "Result/A_analysis/a11_features_matched/roi_features_matched.csv",
    }
    rows = []
    for cohort, path in sources.items():
        df = pd.read_csv(path)
        for feature in ("mu", "grad", "var", "p90"):
            for roi, metrics in l3_explain(df, feature).items():
                rows.append(dict(cohort=cohort, feature=feature, roi=roi, **metrics))
    out = ROOT / "Result/B_summary/b2_feature_summary/feature_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
