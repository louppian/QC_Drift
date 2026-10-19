"""Compare T and the common-denominator AI components; export aggregate rows only."""
import numpy as np
import pandas as pd
from common import B, SEED, OUT, interval, populations, presentations, save


def main():
    original = populations(presentations())['original800']
    linkage = pd.read_csv(OUT / 'c5_ai_linkage.csv')
    ai = linkage.loc[linkage.record_type.eq('observation'), ['uid', 'ai_RB']]
    data = original.merge(ai, on='uid', how='left', validate='one_to_one')
    assert len(data) == 800 and data.loc[data.year.eq(2026), 'ai_RB'].notna().all()
    patients = data.groupby('pid', sort=True).year.first()
    pids = list(patients.index)
    draw = np.zeros((B, len(pids)))
    rng = np.random.default_rng(SEED)
    for year in [2024, 2026]:
        ix = np.flatnonzero(patients.to_numpy() == year)
        draw[:, ix] = rng.multinomial(len(ix), np.repeat(1 / len(ix), len(ix)), B)

    def rate(mask, event):
        sub = data.loc[mask].copy()
        sub['num'] = event.loc[mask].astype(float) * sub.w
        sub['den'] = sub.w
        mat = sub.groupby('pid')[['num', 'den']].sum().reindex(pids, fill_value=0).to_numpy()
        totals = draw @ mat
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = totals[:, 0] / totals[:, 1]
        return mat[:, 0].sum() / mat[:, 1].sum(), boot, dict(
            numerator=int(event.loc[mask].sum()), denominator=len(sub), patients=sub.pid.nunique())

    up, up_boot, up_count = rate(data.year.eq(2024) & data.old_RB.eq(3), data.new_RB.eq(4))
    down, down_boot, down_count = rate(data.year.eq(2024) & data.old_RB.eq(4), data.new_RB.le(3))
    old4 = data.year.eq(2026) & data.old_RB.eq(4)
    b, b_boot, b_count = rate(old4, data.ai_RB.eq(4) & data.new_RB.le(3))
    c, c_boot, c_count = rate(old4, data.ai_RB.le(3) & data.new_RB.le(3))
    common = dict(analysis='original800', roi='RB', bootstrap=B, seed=SEED,
                  cluster='patient within year', multiplicity='none; exploratory directional contrasts')
    rows = []
    terms = [('T', up-down, up_boot-down_boot), ('B', b, b_boot), ('C', c, c_boot),
             ('B_minus_C', b-c, b_boot-c_boot), ('C_minus_B', c-b, c_boot-b_boot)]
    for term, point, boot in terms:
        counts = b_count if term == 'B' else c_count if term == 'C' else {}
        rows.append(dict(record_type='hypothesis_estimate', parameter=term,
                         unit='probability difference' if term not in ['B', 'C'] else 'probability',
                         **common, **counts, **interval(point, boot)))
    for term, counts in [('up34_2024', up_count), ('down34_2024', down_count),
                         ('B', b_count), ('C', c_count)]:
        rows.append(dict(record_type='hypothesis_count', parameter=term, **common, **counts))
    for hypothesis, term in [('H1', 'T'), ('H2_pattern', 'B_minus_C'), ('H3_pattern', 'C_minus_B')]:
        _, point, boot = next(t for t in terms if t[0] == term)
        stats = interval(point, boot)
        status = '충족' if stats['lower_one_sided'] > 0 else '반대 방향' if point < 0 else '불충분'
        rows.append(dict(record_type='hypothesis_condition', hypothesis=hypothesis, parameter=term,
                         direction=1, status=status, **common, **stats))
    # Publish c5 aggregates without patient/image IDs, file paths, row indices or hashes.
    aggregate = linkage.loc[~linkage.record_type.isin(['observation', 'source'])].copy()
    aggregate = aggregate.drop(columns=['uid', 'pid', 'source_npz', 'source_index', 'sha256'], errors='ignore')
    assert not any(col in aggregate for col in ['uid', 'pid', 'source_npz', 'source_index', 'sha256'])
    result = pd.concat([aggregate, pd.DataFrame(rows)], ignore_index=True)
    save(result, 'c6_three_hypotheses')


if __name__ == '__main__':
    main()
