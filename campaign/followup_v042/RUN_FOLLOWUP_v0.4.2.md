# Running the v0.4.2 follow-up

Run everything from this folder (the one that contains `runs/`), in Terminal, with the same Python
environment that ran the main_v5 notebook (the `stb04` kernel). To find that Python:

    cat ~/Library/Jupyter/kernels/stb04/kernel.json

The first entry of `"argv"` is the Python path. Below it is written as `PY`; set it once:

    PY=/path/from/kernel.json/python

## 1. Verify inputs (seconds)

    $PY followup_v042.py --dry-run

Checks the library against the main_v5 source hashes, the package versions, the frozen calibration
model and every saved input of the 50 audit runs, then prints the plan. It evaluates nothing.

## 2. Reproduction check (about 3 minutes)

    $PY followup_v042.py --check

Rescores one honest audit run with this script's pipeline and compares it with the saved main_v5
profile. It must say `CHECK PASSED`. On this Mac, with the default `--threads 2`, the match should be
exact. If it says `CHECK FAILED`, stop and send the output.

## 3. Full run (about 8 to 9 hours with 6 workers)

    caffeinate -dimsu $PY followup_v042.py --workers 6

Keep the lid open and the charger connected. Progress lines appear as each base finishes and are also
written to `runs/followup_v042/progress.log`. The run is resumable: if it stops, run the same command
again and finished tables are skipped. Order of work: the 50 audit bases (19 tables each), then the 150
supplementary null runs.

## 4. Analyse (a minute; works on a partial run)

    $PY analyze_followup_v042.py

Writes `runs/followup_v042/analysis/summary.json` and two CSV files, and prints the headline numbers.
It analyses complete bases only, so it can be run while the experiment is still going.
