"""Grade transitions followed by RB boundary estimates; one CSV for all populations."""
import argparse
from common import B, SEED, ROIS, calculate, interval, populations, presentations, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", choices=ROIS, default="RB")
    boundary_roi = ap.parse_args().roi
    rows = []
    conditions = [('H2024', 'T', 1), ('H2024', 'delta_year', 1), ('H2024', 'delta_cutpoint_2024', 1),
                  ('HAI', 'U', -1), ('HAI', 'delta_year', 1), ('HAI', 'delta_cutpoint_2026', -1)]
    for name, data in populations(presentations()).items():
        for roi in ROIS:
            for year in [2024, 2026]:
                sub = data.loc[data.year.eq(year)]
                for old in range(5):
                    grade = sub.loc[sub[f'old_{roi}'].eq(old)]
                    diff = grade[f'new_{roi}'] - old
                    rows.append(dict(record_type='grade_change', analysis=name, roi=roi, year=year, old_grade=old,
                                     n=len(grade), change_sum=int(diff.sum()), mean_change=diff.mean(),
                                     weighting='equal presentation'))
                    for new in range(5):
                        rows.append(dict(record_type='transition', analysis=name, roi=roi, year=year,
                                         old_grade=old, new_grade=new, n=int(grade[f'new_{roi}'].eq(new).sum()),
                                         weighting='equal presentation'))
        point, boot, counts = calculate(data, roi=boundary_roi)
        weighting = 'inverse inclusion probability' if name == 'original800' else 'equal presentation'
        for count in counts:
            rows.append(dict(record_type='boundary_count', analysis=name, roi=boundary_roi, weighting=weighting, **count))
        for term, value in point.items():
            rows.append(dict(record_type='boundary_estimate', analysis=name, roi=boundary_roi, parameter=term,
                             **interval(value, boot[term]), bootstrap=B, seed=SEED, n_presentations=len(data),
                             weighting=weighting, cluster='patient within year', unit='probability difference or rate'))
        # Preserve the old specified directional tests as interpretation rows, not as the analysis outline.
        for hypothesis, term, direction in conditions:
            stats = interval(point[term], boot[term])
            passed = stats['lower_one_sided'] > 0 if direction == 1 else stats['upper_one_sided'] < 0
            status = '충족' if passed else '반대 방향' if direction * point[term] < 0 else '불충분'
            rows.append(dict(record_type='hypothesis_condition', analysis=name, roi=boundary_roi, hypothesis=hypothesis,
                             parameter=term, direction=direction, status=status, **stats, bootstrap=B, seed=SEED,
                             n_presentations=len(data), weighting=weighting, cluster='patient within year'))
    save(rows, 'c3_grade_boundary' + ('' if boundary_roi == 'RB' else '_' + boundary_roi))


if __name__ == '__main__':
    main()
