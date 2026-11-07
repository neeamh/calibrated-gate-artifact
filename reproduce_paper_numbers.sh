#!/usr/bin/env bash
# Regenerate every number, table and figure in the paper from the saved results, then check that
# the outputs match the committed ones: .tex and .csv byte for byte, .json up to floating-point
# rounding in the last digits (different numerical-library builds can change the 16th digit).
#
#   PYTHON=python ./reproduce_paper_numbers.sh          # everything (about 12 minutes)
#   QUICK=1 PYTHON=python ./reproduce_paper_numbers.sh  # skip the 1,000 reshuffled splits (about 3 minutes)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export SOURCE_DATE_EPOCH=0 MPLBACKEND=Agg
reference="$(mktemp -d)"
mkdir -p "$reference/generated" "$reference/followup"
cp -r paper/generated/. "$reference/generated/"
cp -r results/followup_v042/analysis/. "$reference/followup/"

echo "== Original analysis (main_v5, 39-table set)"
for script in make_numbers make_figures extra_analyses mechanism_status make_extra_tables reanalysis make_reanalysis_tables; do
  echo "   $script"
  "$PY" "paper/scripts/$script.py" > /dev/null
done

echo "== Calibration validation by reshuffled splits"
if [ -n "${QUICK:-}" ]; then
  echo "   skipped (QUICK=1); the committed resplit_validation.json is used"
else
  "$PY" paper/scripts/resplit_validation.py > /dev/null
fi

echo "== v0.4.2 follow-up (primary analysis, 50 bases)"
"$PY" paper/scripts/extract_followup_tables.py > /dev/null
"$PY" paper/scripts/followup_reanalysis.py > /dev/null
"$PY" paper/scripts/make_followup.py > /dev/null

set +e
"$PY" - "$reference" <<'PYCHECK'
import filecmp, json, math, pathlib, sys
ref = pathlib.Path(sys.argv[1])
pairs = [(ref / 'generated', pathlib.Path('paper/generated')), (ref / 'followup', pathlib.Path('results/followup_v042/analysis'))]

def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b

status = 0
for old_dir, new_dir in pairs:
    for old in sorted(old_dir.iterdir()):
        if old.suffix not in ('.tex', '.json', '.csv'):
            continue
        new = new_dir / old.name
        if old.suffix == '.json':
            same = close(json.loads(old.read_text()), json.loads(new.read_text()))
        else:
            same = filecmp.cmp(old, new, shallow=False)
        print(f"{'identical ' if same else 'DIFFERENT '} {new}")
        status |= not same
print('All regenerated numbers match the paper.' if status == 0 else 'Some outputs differ; see above.')
sys.exit(status)
PYCHECK
status=$?
rm -rf "$reference"
exit "$status"
