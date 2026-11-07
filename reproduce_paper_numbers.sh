#!/usr/bin/env bash
# Regenerate every number, table and figure in the paper from the saved campaign results,
# then check that the text outputs are identical to the committed ones in paper/generated/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export SOURCE_DATE_EPOCH=0 MPLBACKEND=Agg
reference="$(mktemp -d)"
cp -r paper/generated/. "$reference/"
for script in make_numbers make_figures extra_analyses mechanism_status make_extra_tables reanalysis make_reanalysis_tables; do
  echo "== $script"
  "$PY" "paper/scripts/$script.py" > /dev/null
done
status=0
for f in "$reference"/*.tex "$reference"/*.json; do
  name="$(basename "$f")"
  if cmp -s "$f" "paper/generated/$name"; then echo "identical  $name"; else echo "DIFFERENT  $name"; status=1; fi
done
rm -rf "$reference"
if [ "$status" -eq 0 ]; then echo "All regenerated numbers match the paper."; else echo "Some outputs differ; see above."; fi
exit "$status"
