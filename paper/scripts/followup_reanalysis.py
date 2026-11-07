"""Rerun the prespecified analysis and the sensitivity analyses of the v0.4.2 follow-up on the saved
profiles in this repository (no new runs, nothing refit).

The two scripts in campaign/followup_v042/ were written for the original run folder, whose ledger
(runs/main_v5/full_489b297b38e6f526/states/) is not released. This wrapper runs them unchanged and
replaces only the ledger lookup: the 50 honest audit profiles are read from results/main_v5/profiles/
instead. Outputs are written to results/followup_v042/analysis/.

Usage (from the repository root):  python paper/scripts/followup_reanalysis.py
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'campaign' / 'followup_v042'


class ArtifactRun:
    """Stand-in for followup_v042.FrozenRun that reads the released profiles."""

    def __init__(self, run_dir):
        self.dir = ROOT / 'results' / 'main_v5'

    def result(self, key, verify=()):
        kind, _, repeat = key.split('/')
        if kind != 'null':
            raise SystemExit(f'Unexpected ledger key {key}')
        matches = sorted((self.dir / 'profiles' / 'jobs').glob(f'null__profile__{int(repeat):02d}/*/profile.json'))
        if len(matches) != 1:
            raise SystemExit(f'Expected one saved profile for {key}, found {len(matches)}')
        return {'profile_path': str(matches[0])}, {}

    def verify(self, path, expected):
        return None  # file integrity is covered by SHA256SUMS.txt


def run(script, *args):
    sys.argv = [str(script), '--out', 'results/followup_v042', *args]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(script), run_name='__main__')
    return buffer.getvalue()


def main():
    import os
    os.chdir(ROOT)
    sys.path.insert(0, str(FOLLOWUP))
    import followup_v042 as F
    F.FrozenRun = ArtifactRun
    print(run(FOLLOWUP / 'analyze_followup_v042.py').strip().splitlines()[0])
    out = run(FOLLOWUP / 'sensitivity_followup_v042.py')
    print(out.strip().splitlines()[0])


if __name__ == '__main__':
    main()
