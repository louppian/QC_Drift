"""Connect Phase 1 direction reversal with observed rereading changes (descriptive)."""
import json
import pandas as pd
from common import ROOT, ROIS, populations, presentations, save


def main():
    rows = []
    for name, data in populations(presentations()).items():
        for roi in ROIS:
            changes = {}
            for year in [2024, 2026]:
                sub = data.loc[data.year.eq(year)]
                old, new = sub[f'old_{roi}'], sub[f'new_{roi}']
                diff = new - old
                changes[year] = diff.mean()
                rows.append(dict(record_type='reread_change', analysis=name, roi=roi, year=year,
                                 n=len(sub), patients=sub.pid.nunique(), old_mean=old.mean(), new_mean=new.mean(),
                                 change_sum=int(diff.sum()), mean_change=diff.mean(), up=int(diff.gt(0).sum()),
                                 same=int(diff.eq(0).sum()), down=int(diff.lt(0).sum()), weighting='equal presentation', unit='grade'))
            rows.append(dict(record_type='year_contrast', analysis=name, roi=roi,
                             metric='change_2024_minus_2026', estimate=changes[2024] - changes[2026],
                             weighting='equal presentation', unit='grade'))
    cross = pd.read_csv(ROOT / 'Result/Phase1/A_analysis/a4_cross.csv')
    for roi in ROIS:
        biases = {}
        for direction in ['fwd', 'rev']:
            sub = cross.loc[cross.direction.eq(direction) & cross.metric.eq(f'per_roi/{roi}/bias')]
            assert len(sub) == 5
            biases[direction] = sub.value.astype(float).mean()
            for r in sub.itertuples():
                rows.append(dict(record_type='phase1_run', roi=roi, arm=r.arm, metric='bias',
                                 direction=direction, estimate=float(r.value), source=r.source, unit='grade'))
        rows.append(dict(record_type='phase1_decomposition', roi=roi, fwd=biases['fwd'], rev=biases['rev'],
                         delta_obs=(biases['rev']-biases['fwd'])/2, gamma=(biases['rev']+biases['fwd'])/2, unit='grade'))
    summary = pd.read_csv(ROOT / 'Result/Phase1/B_summary/b1_summary.csv')
    for r in summary.loc[summary.metric.str.startswith('delta_corrected/')].itertuples():
        rows.append(dict(record_type='phase1_corrected', roi=r.roi, metric=r.metric,
                         estimate=json.loads(r.value), source=r.source, unit='grade'))
    save(rows, 'c2_relative_difference')


if __name__ == '__main__':
    main()
