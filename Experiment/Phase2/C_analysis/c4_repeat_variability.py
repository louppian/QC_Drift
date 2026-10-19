"""Assess criterion stability using repeat comparisons and full-presentation RB dispersion.
The 324 pairs and all bootstrap settings are unchanged. Full-presentation SD is descriptive.
No reading is ground truth.
"""
import numpy as np
import pandas as pd
from common import presentations, save
B, SEED = (20000, 20261014)

def main():
    j = presentations()
    f = j.drop_duplicates('uid')
    r = j[j.uid.duplicated()].merge(f[['uid', 'new_RT', 'new_LT', 'new_RB', 'new_LB']], on='uid', suffixes=('', '_first'), validate='many_to_one')
    assert len(r) == 324 and r.uid.nunique() == 287
    patients = r.groupby('pid', sort=True).year.first()
    draw = np.zeros((B, len(patients)))
    rng = np.random.default_rng(SEED)
    for year in [2024, 2026]:
        ix = np.flatnonzero(patients.to_numpy() == year)
        draw[:, ix] = rng.multinomial(len(ix), np.repeat(1 / len(ix), len(ix)), B)

    def estimate(values, mask):
        table = pd.DataFrame({'pid': r.pid.to_numpy(), 'num': values * mask, 'den': mask.astype(int)}).groupby('pid')[['num', 'den']].sum().reindex(patients.index)
        totals = draw @ table.to_numpy()
        return (float(table.num.sum() / table.den.sum()), totals[:, 0] / totals[:, 1], int(table.num.sum()), int(table.den.sum()))

    def ci(values):
        assert np.isfinite(values).all()
        return np.quantile(values, [0.025, 0.975])
    rows, counts, contrasts = ([], [], [])
    for roi in ['RT', 'LT', 'RB', 'LB']:
        old = r['old_' + roi].to_numpy()
        first = r['new_' + roi + '_first'].to_numpy()
        repeat = r['new_' + roi].to_numpy()
        measures = {'ACC': lambda a, b: (a == b).astype(float), 'MAE': lambda a, b: abs(a - b), 'boundary_disagreement': lambda a, b: ((a == 4) != (b == 4)).astype(float)}
        for metric, measure in measures.items():
            a = measure(old, first)
            b = measure(first, repeat)
            np.testing.assert_array_equal(b, measure(repeat, first))
            ref, ref_boot, rnum, rden = estimate(b, np.ones(len(r), dtype=bool))
            rl, rh = ci(ref_boot)
            estimates = {}
            for year in [2024, 2026, 'all']:
                mask = r.year.eq(year).to_numpy() if year != 'all' else np.ones(len(r), dtype=bool)
                point, boot, num, den = estimate(a, mask)
                lo, hi = ci(boot)
                diff = point - ref
                dl, dh = ci(boot - ref_boot)
                status = 'confirmed' if (dh < 0 if metric == 'ACC' else dl > 0) else 'inconclusive'
                rows.append(dict(roi=roi, metric=metric, year=year, past_numerator=num, past_denominator=den, past_estimate=point, past_ci_low=lo, past_ci_high=hi, repeat_numerator=rnum, repeat_denominator=rden, repeat_estimate=ref, repeat_ci_low=rl, repeat_ci_high=rh, difference=diff, difference_ci_low=dl, difference_ci_high=dh, exceeds_repeat=status, bootstrap=B, seed=SEED, reference='pooled all-year repeats', cluster='patient within year'))
                estimates[year] = (point, boot)
            diff = estimates[2026][0] - estimates[2024][0]
            lo, hi = ci(estimates[2026][1] - estimates[2024][1])
            contrasts.append(dict(roi=roi, metric=metric, contrast='2026 minus 2024', estimate=diff, ci_low=lo, ci_high=hi, bootstrap=B, seed=SEED))
        for year in [2024, 2026, 'all']:
            mask = r.year.eq(year).to_numpy() if year != 'all' else np.ones(len(r), dtype=bool)
            d = repeat[mask] - first[mask]
            counts.append(dict(roi=roi, year=year, pairs=int(mask.sum()), unique_images=r.loc[mask, 'uid'].nunique(), patients=r.loc[mask, 'pid'].nunique(), same=int((d == 0).sum()), up=int((d > 0).sum()), down=int((d < 0).sum())))
    # Retain the old report's full-presentation SD comparison alongside the
    # repeat-pair analysis. This uses all presentations, not the 324 pairs.
    dispersion = []
    for year in [2024, 2026]:
        sub = j.loc[j.year.eq(year)]
        change = sub.new_RB - sub.old_RB
        dispersion.append(dict(
            record_type='full_presentation_dispersion', analysis='presentation1155',
            roi='RB', metric='SD', year=year, n=len(sub), patients=sub.pid.nunique(),
            mean_change=change.mean(), change_sum=int(change.sum()),
            change_square_sum=int((change ** 2).sum()), estimate=change.std(ddof=1),
            ddof=1, weighting='equal presentation', unit='grade',
        ))
    save([dict(record_type='comparison', **r) for r in rows] + [dict(record_type='repeat_count', **r) for r in counts] + [dict(record_type='year_contrast', **r) for r in contrasts] + dispersion, 'c4_repeat_variability')
if __name__ == '__main__':
    main()
