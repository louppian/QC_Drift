"""Descriptive audit of mini-sequence context; aggregate output only."""
import numpy as np
import pandas as pd
from common import OUT, presentations, save


def main():
    j = presentations()
    contents = j.sort_values(['ms_id', 'seq2']).groupby('ms_id').uid.apply(tuple)
    sizes = j.groupby('ms_id').size()
    assert len(sizes) == 165 and sizes.eq(7).all()
    first = j.drop_duplicates('uid')
    r = j[j.uid.duplicated()].merge(
        first[['uid', 'ms_id', 'new_RB']], on='uid',
        suffixes=('', '_first'), validate='many_to_one')
    assert len(r) == 324 and r.uid.nunique() == 287
    r['different_context'] = r.ms_id.map(contents) != r.ms_id_first.map(contents)
    rows = []

    def add(kind, metric, num, den, unit, year='all', **extra):
        rows.append(dict(record_type=kind, metric=metric, year=year, roi='RB',
                         numerator=int(num), denominator=int(den),
                         estimate=num / den if den else np.nan, unit=unit, **extra))

    add('sequence_composition', 'length7', sizes.eq(7).sum(), len(sizes), 'sequence')
    for year in ['all', 2024, 2026]:
        sub = r if year == 'all' else r[r.year.eq(year)]
        delta = sub.new_RB - sub.new_RB_first
        add('sequence_composition', 'different_context', sub.different_context.sum(),
            len(sub), 'pair', year)
        for metric, mask in [('same_grade', delta.eq(0)), ('higher_grade', delta.gt(0)),
                             ('lower_grade', delta.lt(0)),
                             ('grade4_changed', sub.new_RB.eq(4) != sub.new_RB_first.eq(4))]:
            add('repeat_grade', metric, mask.sum(), len(sub), 'pair', year,
                unique_images=sub.uid.nunique(), patients=sub.pid.nunique())

    a = pd.read_csv(OUT / 'c5_ai_linkage.csv')
    a = a[a.record_type.eq('observation')].copy()
    assert len(a) == 300 and a.uid.is_unique
    check = a.merge(first[['uid', 'new_RB']], on='uid', suffixes=('', '_first'), validate='one_to_one')
    assert len(check) == 300 and check.new_RB.eq(check.new_RB_first).all()
    target = a[a.ai_RB.le(3) & a.old_RB.eq(4) & a.new_RB.eq(3)]
    repeated = r[r.uid.isin(target.uid)]
    n_repeated = repeated.uid.nunique()
    add('target_repeat', 'repeated_images', n_repeated, len(target), 'image', 2026)
    add('target_repeat', 'single_presentation_images', len(target) - n_repeated,
        len(target), 'image', 2026)
    stable = repeated.groupby('uid').new_RB.apply(lambda x: x.eq(3).all())
    add('target_repeat', 'all_repeats_grade3', stable.sum(), len(stable), 'image', 2026)
    add('target_repeat', 'repeat_grade3', repeated.new_RB.eq(3).sum(), len(repeated), 'pair', 2026)
    lengths = []
    for path in a.source_npz.dropna().unique():
        with np.load(path, allow_pickle=True) as source:
            lengths.append(len(source['inference']))
    assert len(lengths) == 33
    for length, count in pd.Series(lengths).value_counts().sort_index().items():
        add('source_length', 'length_frequency', count, len(lengths), 'sequence', 2026,
            sequence_length=int(length))
    save(rows, 'c7_sequence_context')


if __name__ == '__main__':
    main()
