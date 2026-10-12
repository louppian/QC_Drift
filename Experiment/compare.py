"""Compare Phase1 Result_old with regenerated Result; does not run experiments.
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
PHASE1 = {'E1', 'E2', 'E3', 'E4', 'AppendixB', 'L'}
METADATA = {'mrm', 'npz', 'image_path', 'timestamp', 'elapsed', 'elapsed_seconds', 'tail_bias'}
KEY_COLUMNS = ('uid', 'patient_id', 'patient', 'pid', 'year', 'model', 'roi', 'ROI',
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

def collect(base):
    return {p.relative_to(base).as_posix(): p for p in base.rglob('*')
            if p.is_file() and (p.relative_to(base).parts[0] in PHASE1 or p.name == 'A1_verdict.json')}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old', type=Path, default=ROOT/'Result_old')
    ap.add_argument('--new', type=Path, default=ROOT/'Result')
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--run', action='store_true', help='Run Phase1 from saved best-val weights, then compare')
    args = ap.parse_args()
    if args.run:
        if args.old.resolve() != (ROOT/'Result_old').resolve() or args.new.resolve() != (ROOT/'Result').resolve():
            ap.error('--run uses repository Result_old and Result')
        tasks = [
            'A/a1_structure.py', 'A/a2_label_image.py', 'A/a3_reproducibility.py',
            'A/a4_cross.py', 'A/a5_indomain_matched.py', 'A/a6_positive_control.py',
            'A/a7_negative_control.py', 'A/a8_indomain_raw.py',
            'B/b1_summary.py', 'C/c1_confusion.py',
        ]
        for task in tasks:
            print(f'RUN: {task}', flush=True)
            completed = subprocess.run([sys.executable, str(ROOT/'Experiment'/task)], cwd=ROOT)
            if completed.returncode:
                print(f'STOP: {task} exited {completed.returncode}', file=sys.stderr)
                return completed.returncode

    if args.atol < 0 or args.rtol < 0: ap.error('tolerances must be nonnegative')
    if not args.old.is_dir() or not args.new.is_dir(): ap.error('both directories must exist')
    old, new = collect(args.old), collect(args.new)
    if not new:
        print('No new Phase1 outputs. Run: python Experiment/compare.py --run', file=sys.stderr)
        return 2
    if not old: ap.error('no Phase1 baseline files')
    failures = 0
    for rel in sorted(old.keys() | new.keys()):
        if Path(rel).name == 'history.json':
            print(f'TRAINING_LOG_NOT_REPLAYED: {rel}')
            continue
        if Path(rel).suffix == '.npz':
            print(f'FRESH_PREDICTIONS: {rel}')
            continue
        if rel not in old or rel not in new:
            # Image presence is informative, not a numeric reproduction verdict.
            missing = 'missing_new' if rel not in new else 'new_only'
            print(f'{missing}: {rel}')
            if Path(rel).suffix.lower() in {'.csv', '.json'}: failures += 1
            continue
        if Path(rel).suffix.lower() not in {'.csv', '.json'}:
            print(f'NON_METRIC: {rel} bytes_equal={old[rel].read_bytes()==new[rel].read_bytes()}')
            continue
        issues, metadata = [], []
        try:
            if rel.endswith('.csv'):
                fa, a = read_csv(old[rel]); fb, b = read_csv(new[rel])
                if set(fa) != set(fb): issues.append((rel, fa, fb, 'columns'))
                a, b, keys = align_csv(a,b)
                print(f'CSV_KEY: {rel} {keys}')
            else:
                a = json.loads(old[rel].read_text(encoding='utf-8-sig'))
                b = json.loads(new[rel].read_text(encoding='utf-8-sig'))
            compare_values(a,b,rel,issues,metadata,args.atol,args.rtol)
        except (ValueError, OSError, csv.Error) as exc:
            issues.append((rel,'','',str(exc)))
        print(f'{"DIFF" if issues else "PASS"}: {rel}')
        for at, a, b, why in issues:
            print(f'  {at}: old={a!r} new={b!r} {why}')
        for at in metadata: print(f'  METADATA: {at}')
        failures += bool(issues)
    print(f'files_with_differences_or_missing={failures}; atol={args.atol}; rtol={args.rtol}')
    return int(failures > 0)

if __name__ == '__main__':
    raise SystemExit(main())
