"""Combine ROI drift CIs with patient-bootstrap between-year feature AUC differences."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
B, SEED = 20000, 20261012
FEATURES = ['mean', 'median', 'entropy', 'uniformity', 'glrlm_SRE', 'glszm_SAE']
ROIS = ['RT', 'LT', 'RB', 'LB']


def max_auc_distribution(frame, roi, lower, patients, draws):
    sub = frame[frame[roi].isin([lower, lower + 1])]
    positive = sub[roi].eq(lower + 1).to_numpy()
    ids = pd.Categorical(sub.patient, categories=patients).codes
    values = np.full((len(draws), len(FEATURES)), np.nan)
    observed = []

    def auc(pos, neg):
        denominator = pos.sum(axis=1) * neg.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            score = (pos * (neg.cumsum(axis=1) - .5 * neg)).sum(axis=1) / denominator
        return np.maximum(score, 1 - score)

    for index, feature in enumerate(FEATURES):
        x = sub[f'{roi}_{feature}'].to_numpy()
        assert np.isfinite(x).all()
        groups = np.unique(x, return_inverse=True)[1]
        pos = np.zeros((len(patients), groups.max() + 1))
        neg = np.zeros_like(pos)
        np.add.at(pos, (ids[positive], groups[positive]), 1)
        np.add.at(neg, (ids[~positive], groups[~positive]), 1)
        observed.append(auc(pos.sum(axis=0, keepdims=True), neg.sum(axis=0, keepdims=True))[0])
        for start in range(0, len(draws), 500):
            draw = draws[start:start + 500]
            values[start:start + 500, index] = auc(draw @ pos, draw @ neg)
    valid = np.isfinite(values).any(axis=1)
    maximum = np.full(len(draws), np.nan)
    maximum[valid] = np.nanmax(values[valid], axis=1)
    return max(observed), maximum, FEATURES[int(np.argmax(observed))], int((~positive).sum()), int(positive.sum())


def main():
    data = pd.read_csv(ROOT / 'Result/Phase1/A_analysis/a2_label_image_features.csv')
    summary = pd.read_csv(ROOT / 'Result/Phase1/B_summary/b1_summary.csv')
    rng = np.random.default_rng(SEED)
    draws = {}
    for year in [2024, 2026]:
        patients = sorted(data.loc[data.year.eq(year), 'patient'].unique())
        draws[year] = patients, rng.multinomial(len(patients), np.ones(len(patients)) / len(patients), B)
    rows = []
    for roi in ROIS:
        def value(metric):
            series = summary.loc[summary.metric.eq(f'delta_corrected/{roi}/{metric}'), 'value']
            assert len(series) == 1
            return float(series.iloc[0])
        drift, drift_low, drift_high = value('delta_corr'), value('ci/0'), value('ci/1')
        for lower in range(4):
            results = {year: max_auc_distribution(data[data.year.eq(year)], roi, lower, *draws[year])
                       for year in [2024, 2026]}
            delta = results[2026][0] - results[2024][0]
            boot = results[2026][1] - results[2024][1]
            valid = boot[np.isfinite(boot)]
            lo, hi = np.quantile(valid, [.025, .975])
            roi_pass = drift_low > 0 or drift_high < 0
            boundary_pass = lo > 0 or hi < 0
            rows.append(dict(roi=roi, boundary=f'{lower}-{lower+1}',
                delta_corr=drift, delta_corr_ci_low=drift_low, delta_corr_ci_high=drift_high,
                auc_2024=results[2024][0], auc_2026=results[2026][0], delta_auc=delta,
                delta_auc_ci_low=lo, delta_auc_ci_high=hi,
                feature_2024=results[2024][2], feature_2026=results[2026][2],
                n_lower_2024=results[2024][3], n_upper_2024=results[2024][4],
                n_lower_2026=results[2026][3], n_upper_2026=results[2026][4],
                roi_ci_excludes_zero=roi_pass, boundary_ci_excludes_zero=boundary_pass,
                selected=roi_pass and boundary_pass, bootstrap=B, seed=SEED, valid=len(valid),
                cluster='patient within year', multiplicity='unadjusted',
                statistic='max directionless AUC across six features; reselected per bootstrap'))
    frame = pd.DataFrame(rows)
    out = ROOT / 'Result/Phase1/B_summary'
    frame.to_csv(out / 'b3_roi_boundary_priority.csv', index=False, encoding='utf-8-sig')
    lines = ['| ROI | 방향반전 보정값〔95% CI〕 | 0·1 ΔAUC〔95% CI〕 | 1·2 ΔAUC〔95% CI〕 | 2·3 ΔAUC〔95% CI〕 | 3·4 ΔAUC〔95% CI〕 |',
             '|---|---|---|---|---|---|']
    for roi in ROIS:
        group = frame[frame.roi.eq(roi)]
        first = group.iloc[0]
        cells = [roi, f'{first.delta_corr:+.3f}〔{first.delta_corr_ci_low:.3f}, {first.delta_corr_ci_high:.3f}〕']
        for r in group.itertuples():
            cells.append(f'{r.delta_auc:+.3f}〔{r.delta_auc_ci_low:.3f}, {r.delta_auc_ci_high:.3f}〕' + (' 선정' if r.selected else ''))
        lines.append('| ' + ' | '.join(cells) + ' |')
    (out / 'b3_roi_boundary_priority.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(frame[['roi','boundary','delta_auc','delta_auc_ci_low','delta_auc_ci_high','selected']].to_string(index=False))


if __name__ == '__main__':
    main()
