"""Compare aggregated Phase1 CSV with Result_old JSON/CSV; --run reads saved predictions only.
Run: python Experiment/compare.py [--old DIR] [--new DIR] [--atol 1e-6] [--rtol 1e-5]
Exit 0: equal metrics; 1: mismatch/missing; 2: invalid inputs.
"""
import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASE1 = {'E0', 'E1', 'E2', 'E3', 'E4', 'AppendixB', 'L'}
METADATA = {'mrm', 'npz', 'image_path', 'timestamp', 'elapsed', 'elapsed_seconds', 'tail_bias'}
KEY_COLUMNS = ('cohort', 'uid', 'patient_id', 'patient', 'pid', 'year', 'model', 'roi', 'ROI',
               'metric', 'feature', 'boundary', 'grade', 'mode', 'seed', 'fold', 'split', 'arm')

def numeric(value):
    if isinstance(value, bool) or value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def compare_values(old, new, at, issues, metadata, atol, rtol):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            loc = f'{at}/{key}'
            if key not in old or key not in new:
                issues.append((loc, old.get(key, '<missing>'), new.get(key, '<missing>'), 'missing key'))
            elif key in METADATA:
                if old[key] != new[key]: metadata.append(loc)
            else:
                compare_values(old[key], new[key], loc, issues, metadata, atol, rtol)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new): issues.append((at, len(old), len(new), 'length'))
        for i, (a, b) in enumerate(zip(old, new)):
            compare_values(a, b, f'{at}[{i}]', issues, metadata, atol, rtol)
    else:
        a, b = numeric(old), numeric(new)
        if a is not None and b is not None:
            equal = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
            if not equal: issues.append((at, old, new, f'abs_diff={abs(a-b):.12g}'))
        elif old != new:
            issues.append((at, old, new, 'value/type'))

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)): raise ValueError(f'duplicate columns: {path}')
        return fields, list(reader)

def align_csv(a, b):
    # Use the smallest shared identifier set that uniquely identifies rows.
    candidates = [k for k in KEY_COLUMNS if a and b and k in a[0] and k in b[0]]
    for size in range(1, len(candidates)+1):
        for keys in itertools.combinations(candidates, size):
            def indexed(rows): return {tuple(row[k] for k in keys): row for row in rows}
            aa, bb = indexed(a), indexed(b)
            if len(aa) == len(a) and len(bb) == len(b):
                return {repr(k):v for k,v in aa.items()}, {repr(k):v for k,v in bb.items()}, keys
    # No reliable identifier: preserve row order, never sort numeric metric values.
    return a, b, ('row_order',)

# Old baseline paths -> new task paths, without changing Result_old.
TASKS = {'E1':'a4_cross', 'E2':'a5_indomain_matched', 'E3':'a6_positive_control',
         'E4':'a7_negative_control', 'AppendixB':'a8_indomain_raw'}
SPECIAL = {
 'E0/bestval_summary.csv':'A_analysis/a9_recompute_bestval/bestval_summary.csv',
 'L/roi_features_full.csv':'A_analysis/a10_features/roi_features_full.csv',
 'L/L3_feature_summary.txt':'A_analysis/a10_features/L3_feature_summary.txt',
 'L/roi_features_matched.csv':'A_analysis/a11_features_matched/roi_features_matched.csv',
 'L/L3_matched_summary.txt':'A_analysis/a11_features_matched/L3_matched_summary.txt',
 'L/feature_summary.csv':'B_summary/b2_feature_summary/feature_summary.csv',
 'L/l1_structure.csv':'A_analysis/a1_structure.csv',
 'L/l2_features.csv':'A_analysis/a2_label_image_features.csv',
 'L/l2_auc.csv':'A_analysis/a2_label_image_auc.csv',
 'L/l4_reproducibility.csv':'A_analysis/a3_reproducibility.csv',
 'A1_verdict.json':'B_summary/b1_summary/A1_verdict.json',
 'E1/e1_summary.csv':'B_summary/b1_summary/e1_summary.csv',
 'E2/e2_summary.csv':'B_summary/b1_summary/e2_summary.csv',
 'AppendixB/appb_summary.csv':'B_summary/b1_summary/appb_summary.csv',
 'E1/e1_confusion_quadrant.png':'C_figures/c1_confusion.png',
}

def collect(base):
    files = {}
    reverse = {v:k for k,v in SPECIAL.items()}
    reverse_tasks = {v:k for k,v in TASKS.items()}
    for path in base.rglob('*'):
        if not path.is_file(): continue
        rel = path.relative_to(base).as_posix()
        parts = Path(rel).parts
        if rel in reverse:
            key = reverse[rel]
        elif len(parts) > 2 and parts[0] == 'A_analysis' and parts[1] in reverse_tasks:
            key = '/'.join([reverse_tasks[parts[1]], *parts[2:]])
        elif parts[0] in PHASE1 or rel == 'A1_verdict.json':
            key = rel
        elif parts[0] in {'A_analysis', 'B_summary', 'C_figures'} and path.suffix.lower() in {'.csv', '.json'}:
            raise ValueError(f'Unmapped result file: {path}')
        else:
            continue
        if key in files: raise ValueError(f'Duplicate old/new layout for {key} in {base}')
        files[key] = path
    return files

# Fields requiring training history or model-selection logs, not prediction NPZ.
ARM_UNAVAILABLE = {'n_train', 'final_epoch', 'best_val_epoch', 'best_val_mae', 'tail_bias', 'mrm'}
AGGREGATES = {
    'A_analysis/a4_cross.csv': 'E1',
    'A_analysis/a5_indomain_matched.csv': 'E2',
    'A_analysis/a6_positive_control.csv': 'E3',
    'A_analysis/a7_negative_control.csv': 'E4',
    'A_analysis/a8_indomain_raw.csv': 'AppendixB',
    'B_summary/b1_summary.csv': 'summary',
}


def new_documents(base):
    from Experiment.A_analysis.common import read_documents
    documents = {}
    for rel in AGGREGATES:
        path = base / rel
        if not path.exists(): continue
        for source, value in read_documents(path).items():
            if source in documents: raise ValueError(f'Duplicate source in aggregation CSVs: {source}')
            documents[source] = value
    for source, rel in SPECIAL.items():
        path = base / rel
        if path.is_file() and path.suffix.lower() == '.csv':
            if source in documents: raise ValueError(f'Duplicate source: {source}')
            _, documents[source] = read_csv(path)
    # New CSV/JSON files at stage roots must not silently disappear.
    known = set(AGGREGATES) | set(SPECIAL.values())
    for folder in ('A_analysis', 'B_summary'):
        for path in (base/folder).glob('*'):
            if path.is_file() and path.suffix.lower() in {'.csv', '.json'}:
                if path.relative_to(base).as_posix() not in known:
                    raise ValueError(f'Unmapped result file: {path}')
    return documents


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old', type=Path, default=ROOT/'Result_old')
    ap.add_argument('--new', type=Path, default=ROOT/'Result')
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--run', action='store_true', help='Aggregate existing checkpoint NPZ from a4 onward; no training/inference')
    args = ap.parse_args()
    if args.atol < 0 or args.rtol < 0: ap.error('tolerances must be nonnegative')
    if args.run:
        if args.old.resolve() != (ROOT/'Result_old').resolve() or args.new.resolve() != (ROOT/'Result').resolve():
            ap.error('--run uses repository Result_old and Result')
        from Experiment.A_analysis import common as A
        try:
            A.require_predictions(list(A.STAGES))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        tasks = [*(f'A_analysis/{stage}.py' for stage in A.STAGES),
                 'B_summary/b1_summary.py', 'C_figures/c1_confusion.py']
        for task in tasks:
            print(f'RUN: {task}', flush=True)
            completed = subprocess.run([sys.executable, str(ROOT/'Experiment'/task)], cwd=ROOT)
            if completed.returncode:
                print(f'STOP: {task} exited {completed.returncode}', file=sys.stderr)
                return completed.returncode
    if not args.old.is_dir() or not args.new.is_dir(): ap.error('both directories must exist')
    try:
        old = {key:path for key,path in collect(args.old).items()
               if path.suffix.lower() in {'.csv', '.json'} and path.name != 'history.json'}
        new = new_documents(args.new)
    except (ValueError, OSError, csv.Error) as exc:
        ap.error(str(exc))
    if not old: ap.error('no Phase1 baseline files')
    if not new:
        print('No new CSV outputs. Run: python Experiment/compare.py --run', file=sys.stderr)
        return 2
    print('EXCLUDED: history.json; per-run fields '+', '.join(sorted(ARM_UNAVAILABLE)))
    print('INPUT: aggregated CSV only; previous per-run Result JSON/NPZ are not read')
    counts = dict(passed=0, different=0, missing_new=0, new_only=0)
    for source in sorted(old.keys() | new.keys()):
        if source not in old or source not in new:
            status = 'missing_new' if source not in new else 'new_only'
            counts[status] += 1
            print(f'{status}: {source}')
            continue
        issues, metadata = [], []
        try:
            b = new[source]
            if source.endswith('.csv'):
                fields, a = read_csv(old[source])
                if not isinstance(b, list): raise ValueError('Expected CSV rows')
                if b and set(fields) != set(b[0]): issues.append((source, fields, list(b[0]), 'columns'))
                a,b,keys = align_csv(a,b)
            else:
                a = json.loads(old[source].read_text(encoding='utf-8-sig'))
                if source.endswith('/results.json'):
                    a = {k:v for k,v in a.items() if k not in ARM_UNAVAILABLE}
                    b = {k:v for k,v in b.items() if k not in ARM_UNAVAILABLE}
            compare_values(a,b,source,issues,metadata,args.atol,args.rtol)
        except (ValueError, TypeError, OSError, csv.Error) as exc:
            issues.append((source, '', '', str(exc)))
        counts['different' if issues else 'passed'] += 1
        print(f'{"DIFF" if issues else "PASS"}: {source}')
        for at,a,b,why in issues[:20]: print(f'  {at}: old={a!r} new={b!r} {why}')
        if len(issues)>20: print(f'  ... {len(issues)} differing items total')
    print(' '.join(f'{k}={v}' for k,v in counts.items())+f'; atol={args.atol}; rtol={args.rtol}')
    return int(any(counts[k] for k in ('different','missing_new','new_only')))


if __name__ == '__main__':
    raise SystemExit(main())
