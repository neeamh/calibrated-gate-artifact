"""v0.4.2 follow-up experiments on the frozen main_v5 campaign.

What this script runs (prespecified in PREREGISTRATION_v0.4.2.md):

  1. Base replication. Every one of the 50 held-out audit null runs becomes a base table. Its honest
     candidate (pseudo-P, the table the audit already scored) receives each single flaw of the main
     design (12 flaw-dose cells) and a copy control. Honest and flawed tables therefore share the fit
     runs' sampling procedure, role sizes and code, so the calibration guarantee covers the honest
     comparison by construction.
  2. Dense flaws. Three mild components (F: number_inpatient rotated in 50% of rows; U: outcome
     rotated in 10% of rows; P: one training row copied in) are applied alone and in every
     combination. F alone is identical to the single-flaw cell feature_shuffle@0.5 and is not rerun.
  3. Supplementary audit. Fresh real-versus-real null runs (repeat ids 100, 101, ...) drawn by the
     campaign's own procedure and scored with the frozen calibration model only.

Nothing here refits or changes the calibration model, the evaluators or the judges. Every table is
scored with the exact library that produced main_v5 (verified by source hashes) and the frozen model
(verified by SHA-256). The frozen v0.3 profiler is not run: it is not part of these comparisons.

Usage (from the folder that contains runs/):
  python followup_v042.py --dry-run          # verify every input and print the plan; evaluates nothing
  python followup_v042.py --check            # rescore one honest audit run and compare with the saved profile
  python followup_v042.py --workers 6        # full run; resumable, safe to stop and restart
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROTOCOL = 'v0.4.2-followup'
RUN_NAME = 'full_489b297b38e6f526'
# Modules imported by this script; each must match the source hash recorded by the main_v5 run.
USED_MODULES = ('__init__.py', 'config.py', 'core.py', 'data.py', 'results.py', 'reference.py',
                'cached_privacy.py', 'evaluation.py', 'calibration.py', 'judges.py', 'perturbations.py',
                'generators.py')
VERSION_PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'sdmetrics')

# ----------------------------------------------------------------------------- design (prespecified)
SINGLE = (('exact_copy', (.001, .01, .05)), ('feature_shuffle', (.1, .5, .9)),
          ('mcar', (.05, .15, .30)), ('subgroup_suppression', (.10, .50, .90)))
DENSE_COMPONENTS = {
    # name: (strategy family, dose, column override for feature_shuffle, seed stream, intended dimension)
    'F': ('feature_shuffle', .5, None, 'feature_shuffle', 'fidelity'),
    'U': ('feature_shuffle', .1, 'readmitted_30d', 'outcome_shuffle', 'utility'),
    'P': ('exact_copy', 'one_row', None, 'exact_copy', 'privacy'),
}
# F alone equals the single-flaw cell feature_shuffle@0.5 (same seed stream), so it is not rerun.
DENSE_CONDITIONS = ('U', 'P', 'FU', 'FP', 'UP', 'FUP')


def conditions() -> list[dict]:
    out = []
    for family, doses in SINGLE:
        for dose in doses:
            out.append({'name': f'single/{family}@{dose:g}', 'design': 'single',
                        'components': [(family, dose, None, family)]})
    out.append({'name': 'single/copy_control', 'design': 'copy_control', 'components': [('copy', 1., None, 'copy')]})
    for combo in DENSE_CONDITIONS:
        # Fixed application order F, U, P: the copied training row is written last and stays intact.
        comps = [DENSE_COMPONENTS[c][:4] for c in 'FUP' if c in combo]
        out.append({'name': f'dense/{combo}', 'design': 'dense', 'components': comps, 'dense_set': combo})
    return out


# ----------------------------------------------------------------------------- helpers
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    from syntrustbench_campaign.core import write_json  # same JSON cleaning as the campaign
    write_json(path, value)


class FrozenRun:
    """Read-only access to the main_v5 ledger. Paths are rebased onto the run's current location
    and every file this script reads is checked against the ledger's recorded SHA-256."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir).resolve()
        if self.dir.name != RUN_NAME:
            raise SystemExit(f'Expected the run folder {RUN_NAME}, got {self.dir.name}.')
        self._verified: set[str] = set()

    def state(self, key: str) -> dict:
        path = self.dir / 'states' / (key.replace('/', '__') + '.json')
        if not path.is_file():
            raise SystemExit(f'Missing ledger state {path}.')
        record = json.loads(path.read_text())
        if record.get('state') != 'completed':
            raise SystemExit(f'{key} is {record.get("state")}, not completed.')
        return record

    def rebase(self, value):
        if isinstance(value, dict):
            return {k: self.rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.rebase(v) for v in value]
        if isinstance(value, str) and value.startswith('/') and RUN_NAME in value:
            parts = Path(value).parts
            index = len(parts) - 1 - parts[::-1].index(RUN_NAME)
            return str(self.dir.joinpath(*parts[index + 1:]))
        return value

    def result(self, key: str, verify: tuple[str, ...] = ()) -> tuple[dict, dict]:
        """Rebased result plus {rebased path: recorded sha256}; verifies the listed rebased paths."""
        record = self.state(key)
        hashes = {self.rebase(p): h for p, h in record.get('artifact_hashes', {}).items()}
        for path in verify:
            self.verify(path, hashes.get(path))
        return self.rebase(record['result']), hashes

    def verify(self, path: str, expected: str | None) -> None:
        if path in self._verified:
            return
        if expected is None:
            raise SystemExit(f'No recorded hash for {path}.')
        if not Path(path).is_file():
            raise SystemExit(f'Missing frozen artifact {path}.')
        if sha256(path) != expected:
            raise SystemExit(f'Frozen artifact changed: {path}')
        self._verified.add(path)


def find_library(run: FrozenRun, override: str | None) -> Path:
    recorded = json.loads((run.dir / 'implementation_hashes.json').read_text())
    candidates = [Path(override)] if override else sorted((run.dir.parent / 'notebook_library').glob('*/syntrustbench_campaign'))
    for candidate in candidates:
        if all((candidate / m).is_file() and sha256(candidate / m) == recorded[m] for m in USED_MODULES):
            return candidate.resolve()
    raise SystemExit('No syntrustbench_campaign folder matches the main_v5 implementation hashes for '
                     f'{", ".join(USED_MODULES)}. Pass --library to the folder the campaign ran with.')


def configure_threads(threads: int) -> None:
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
        os.environ[name] = str(threads)


def init_worker(library_parent: str, threads: int) -> None:
    configure_threads(threads)
    if library_parent not in sys.path:
        sys.path.insert(0, library_parent)


def seed_globals(job_key: str) -> None:
    # Mirrors the campaign worker (global seeds from the job key); torch is not used by evaluation.
    import random
    import numpy as np
    from syntrustbench_campaign.config import seed_for
    seed = seed_for('job', job_key)
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------- scoring
def decisions(v04: dict, sdmetrics: dict, model: dict, cfg) -> dict:
    """Calibrated gate and matched-cost judges, exactly as the campaign's judge stage scores them."""
    from syntrustbench_campaign.calibration import CalibratedEvaluator
    from syntrustbench_campaign.config import DIMENSIONS
    from syntrustbench_campaign.judges import (GeometricJudge, HarmonicJudge, MeanJudge, SDMetricsJudge,
                                               dimension_scores, matched_acceptance)
    scores, missing = dimension_scores(v04, model, cfg.judges)
    by_alpha = {}
    for alpha in cfg.calibration.target_rates:
        calibrated = CalibratedEvaluator(model).apply(v04, alpha)
        matched = {'Gate': None if calibrated['gate'] == 'NotEvaluated' else calibrated['gate'] == 'Pass'}
        notes = {}
        for judge in (MeanJudge(), GeometricJudge(), HarmonicJudge(), SDMetricsJudge()):
            name = getattr(judge, 'name', 'SDMetrics')
            matched[name], notes[name] = matched_acceptance(judge, v04, sdmetrics, model, cfg, alpha)
        by_alpha[str(alpha)] = {
            'gate': calibrated['gate'], 'gate_family_pvalue': calibrated['gate_family_pvalue'],
            'dimensions': {d: calibrated['dimensions'][d]['status'] for d in DIMENSIONS},
            'accepted_matched': matched, 'matched_notes': notes}
    return {'by_alpha': by_alpha, 'dimension_scores': scores, 'score_missing': missing}


def build_candidate(condition: dict, base, train, cfg, base_id: int):
    """Apply the condition's components in order; each component has its own seed stream per base."""
    from dataclasses import replace
    from syntrustbench_campaign.config import seed_for
    from syntrustbench_campaign.perturbations import strategy
    table, audits = base.copy(deep=True), []
    for family, dose, column, stream in condition['components']:
        local = cfg if column is None else replace(cfg, injections=replace(cfg.injections, shuffle_feature=column))
        # One training row: ceil(dose * n) == 1 for any n >= 1.
        value = .5 / len(table) if dose == 'one_row' else float(dose)
        seed = seed_for(PROTOCOL, base_id, stream)
        table, audit = strategy(family, local, train).apply(table, value, seed)
        if audit['changed_cell_count'] < 1:
            raise RuntimeError(f'{condition["name"]}: component {family}@{dose} changed no cell.')
        audits.append({'family': family, 'dose': dose, 'column': column or (cfg.injections.shuffle_feature
                       if family == 'feature_shuffle' else None), 'seed': seed,
                       'changed_cell_count': audit['changed_cell_count'],
                       'realized_dose': audit.get('realized_dose')})
    return table, audits


def frame_sha(frame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, float_format='%.17g', lineterminator='\n').encode()).hexdigest()


# ----------------------------------------------------------------------------- tasks (run in workers)
def run_base(args: dict) -> dict:
    """All conditions for one audit base. Completed profiles are skipped, so restarts resume."""
    from threadpoolctl import threadpool_limits
    from syntrustbench_campaign.config import from_mapping
    from syntrustbench_campaign.data import read_analytic
    from syntrustbench_campaign.evaluation import evaluate_v04
    from syntrustbench_campaign.judges import quality_report

    run = FrozenRun(Path(args['run_dir']))
    cfg = from_mapping(json.loads((run.dir / 'configuration.json').read_text()))
    model = json.loads(Path(args['model_path']).read_text())
    base_id, out = args['base'], Path(args['out'])
    roles, data_hashes = run.result(f'null/data/{base_id:02d}')
    context, context_hashes = run.result(f'null/context/{base_id:02d}')
    for name in ('train', 'reference', 'positive', 'test', 'floor'):
        run.verify(roles['paths'][name], data_hashes.get(roles['paths'][name]))
    run.verify(context['context_path'], context_hashes.get(context['context_path']))
    base = read_analytic(roles['paths']['positive'], cfg)
    train = read_analytic(roles['paths']['train'], cfg)
    reference = read_analytic(roles['paths']['reference'], cfg)
    if len(base) != roles['n']:
        raise RuntimeError(f'Base {base_id}: candidate is not size matched.')
    done, failed = 0, []
    log_path = out / 'logs' / f'base_{base_id:02d}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for condition in args['conditions']:
        key = f'followup/base{base_id:02d}/{condition["name"]}'
        target = out / 'profiles' / f'base{base_id:02d}' / (condition['name'].replace('/', '__') + '.json')
        if target.is_file():
            existing = json.loads(target.read_text())
            if existing.get('protocol') == PROTOCOL and existing.get('model_sha256') == args['model_sha256']:
                done += 1
                continue
        started = time.perf_counter()
        try:
            with log_path.open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), \
                    threadpool_limits(limits=args['threads']):
                print(f'== {key} {time.strftime("%Y-%m-%d %H:%M:%S")}')
                seed_globals(key)
                candidate, audits = build_candidate(condition, base, train, cfg, base_id)
                if len(candidate) != roles['n'] or list(candidate.columns) != list(base.columns):
                    raise RuntimeError('Candidate shape or schema changed.')
                v04 = evaluate_v04(cfg, context['context_path'], roles, candidate, key)
                sdmetrics = quality_report(reference, candidate, cfg)
                scored = decisions(v04, sdmetrics, model, cfg)
            payload = {'protocol': PROTOCOL, 'model_sha256': args['model_sha256'],
                       'labels': {'table': key, 'kind': 'followup', 'design': condition['design'],
                                  'condition': condition['name'], 'dense_set': condition.get('dense_set'),
                                  'base': base_id, 'truth': 'bad', 'known_bad': True},
                       'components': audits, 'candidate_sha256': frame_sha(candidate),
                       'base_sha256': data_hashes[roles['paths']['positive']],
                       'v04': v04, 'sdmetrics': sdmetrics, 'decisions': scored,
                       'seconds': time.perf_counter() - started}
            atomic_json(target, payload)
            done += 1
        except Exception:
            failed.append(condition['name'])
            error = out / 'errors' / f'base{base_id:02d}__{condition["name"].replace("/", "__")}.txt'
            error.parent.mkdir(parents=True, exist_ok=True)
            error.write_text(traceback.format_exc())
    return {'task': f'base {base_id:02d}', 'done': done, 'failed': failed}


def run_supplementary(args: dict) -> dict:
    """One fresh real-versus-real null run, scored with the frozen model. Its roles and context are
    built in a temporary folder and deleted afterwards; only the profile is kept."""
    from threadpoolctl import threadpool_limits
    from syntrustbench_campaign.config import from_mapping
    from syntrustbench_campaign.data import RoleAssigner, read_analytic
    from syntrustbench_campaign.evaluation import build_context, evaluate_v04
    from syntrustbench_campaign.judges import quality_report

    run = FrozenRun(Path(args['run_dir']))
    cfg = from_mapping(json.loads((run.dir / 'configuration.json').read_text()))
    model = json.loads(Path(args['model_path']).read_text())
    repeat, out = args['repeat'], Path(args['out'])
    target = out / 'supplementary' / f'null_{repeat:03d}.json'
    if target.is_file():
        existing = json.loads(target.read_text())
        if existing.get('protocol') == PROTOCOL and existing.get('model_sha256') == args['model_sha256']:
            return {'task': f'null {repeat}', 'done': 1, 'failed': []}
    cohort, cohort_hashes = run.result('prepare/cohort')
    for path in list(cohort['paths'].values()) + [cohort['patient_audit']]:
        run.verify(path, cohort_hashes.get(path))
    key = f'null/{repeat:02d}'
    log_path = out / 'logs' / f'null_{repeat:03d}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f'null_{repeat:03d}_', dir=out / 'tmp'))
    started = time.perf_counter()
    try:
        with log_path.open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), \
                threadpool_limits(limits=args['threads']):
            seed_globals(f'null/data/{repeat:02d}')
            roles = RoleAssigner(cfg).null(cohort, repeat, scratch / 'roles')
            seed_globals(f'null/context/{repeat:02d}')
            (scratch / 'context').mkdir(parents=True, exist_ok=True)
            context = build_context(cfg, roles, scratch / 'context')
            seed_globals(f'null/profile/{repeat:02d}')
            candidate = read_analytic(roles['paths']['positive'], cfg)
            reference = read_analytic(roles['paths']['reference'], cfg)
            v04 = evaluate_v04(cfg, context['context_path'], roles, candidate, key)
            sdmetrics = quality_report(reference, candidate, cfg)
            scored = decisions(v04, sdmetrics, model, cfg)
        payload = {'protocol': PROTOCOL, 'model_sha256': args['model_sha256'],
                   'labels': {'table': key, 'kind': 'null', 'repeat': repeat, 'partition': 'supplementary_audit',
                              'truth': 'good', 'known_bad': False},
                   'roles': {k: v for k, v in roles.items() if k in ('n', 'key', 'role_counts', 'scope')},
                   'candidate_sha256': sha256(roles['paths']['positive']),
                   'v04': v04, 'sdmetrics': sdmetrics, 'decisions': scored,
                   'seconds': time.perf_counter() - started}
        atomic_json(target, payload)
        return {'task': f'null {repeat}', 'done': 1, 'failed': []}
    except Exception:
        error = out / 'errors' / f'null_{repeat:03d}.txt'
        error.parent.mkdir(parents=True, exist_ok=True)
        error.write_text(traceback.format_exc())
        return {'task': f'null {repeat}', 'done': 0, 'failed': [f'null_{repeat:03d}']}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ----------------------------------------------------------------------------- check against main_v5
def compare(saved, new, path='', tolerance=1e-9, problems=None):
    problems = [] if problems is None else problems
    if isinstance(saved, dict) and isinstance(new, dict):
        for k in sorted(set(saved) | set(new)):
            if k not in saved or k not in new:
                problems.append(f'{path}/{k}: present in only one profile')
            else:
                compare(saved[k], new[k], f'{path}/{k}', tolerance, problems)
    elif isinstance(saved, list) and isinstance(new, list):
        if len(saved) != len(new):
            problems.append(f'{path}: list length {len(saved)} vs {len(new)}')
        else:
            for i, (a, b) in enumerate(zip(saved, new)):
                compare(a, b, f'{path}[{i}]', tolerance, problems)
    elif isinstance(saved, (int, float)) and isinstance(new, (int, float)) and not isinstance(saved, bool):
        if abs(saved - new) > tolerance * max(1., abs(saved), abs(new)):
            problems.append(f'{path}: {saved!r} vs {new!r}')
    elif saved != new:
        problems.append(f'{path}: {saved!r} vs {new!r}')
    return problems


def check(run: FrozenRun, model: dict, cfg, base_id: int, threads: int) -> None:
    """Rescore one honest audit run with this script's pipeline and compare with main_v5."""
    from threadpoolctl import threadpool_limits
    from syntrustbench_campaign.core import clean_json
    from syntrustbench_campaign.data import read_analytic
    from syntrustbench_campaign.evaluation import evaluate_v04
    from syntrustbench_campaign.judges import quality_report

    roles, data_hashes = run.result(f'null/data/{base_id:02d}')
    context, context_hashes = run.result(f'null/context/{base_id:02d}')
    run.verify(context['context_path'], context_hashes.get(context['context_path']))
    profile_result, profile_hashes = run.result(f'null/profile/{base_id:02d}')
    run.verify(profile_result['profile_path'], profile_hashes.get(profile_result['profile_path']))
    saved = json.loads(Path(profile_result['profile_path']).read_text())
    for name in ('train', 'reference', 'positive', 'test', 'floor'):
        run.verify(roles['paths'][name], data_hashes.get(roles['paths'][name]))
    print(f'Rescoring honest audit run {base_id:02d} (about 2-3 minutes) ...', flush=True)
    with threadpool_limits(limits=threads):
        seed_globals(f'null/profile/{base_id:02d}')
        candidate = read_analytic(roles['paths']['positive'], cfg)
        reference = read_analytic(roles['paths']['reference'], cfg)
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            v04 = clean_json(evaluate_v04(cfg, context['context_path'], roles, candidate, saved['labels']['table']))
            sdmetrics = clean_json(quality_report(reference, candidate, cfg))
    exact = compare(saved['v04']['signals'], v04['signals'], 'v04.signals', tolerance=1e-12)
    exact += compare(saved['sdmetrics'].get('score'), sdmetrics.get('score'), 'sdmetrics.score', tolerance=1e-12)
    old_decisions = decisions(saved['v04'], saved['sdmetrics'], model, cfg)
    new_decisions = decisions(v04, sdmetrics, model, cfg)
    strip = lambda d: {a: {k: v for k, v in row.items() if k != 'matched_notes'} for a, row in d['by_alpha'].items()}
    verdicts = compare(strip(old_decisions), strip(new_decisions), 'decisions', tolerance=0.)
    verdicts = [v for v in verdicts if 'gate_family_pvalue' not in v]
    if verdicts:
        print('CHECK FAILED: decisions on the rescored honest run differ from main_v5:')
        for line in (verdicts + exact)[:25]:
            print('  ', line)
        raise SystemExit(1)
    if exact:
        largest = max(exact, key=len)
        print(f'CHECK PASSED WITH NUMERICAL DIFFERENCES: every gate, dimension and judge decision at three alpha '
              f'levels matches main_v5 for audit run {base_id:02d}, but {len(exact)} signal value(s) differ slightly '
              f'(for example {largest}).\nOn the machine that produced main_v5, with the same --threads, an exact '
              'match is expected; differences there mean the environment changed.')
        return
    print(f'CHECK PASSED: every signal, the SDMetrics score and every gate, dimension and judge decision at three '
          f'alpha levels reproduce main_v5 exactly for audit run {base_id:02d}.')


def collect(future, futures) -> dict:
    try:
        return future.result()
    except Exception:
        payload = futures[future]
        name = f'base {payload["base"]:02d}' if 'base' in payload else f'null {payload["repeat"]}'
        return {'task': name, 'done': 0, 'failed': ['worker crashed: ' + traceback.format_exc().splitlines()[-1]]}


def report(outcome: dict, finished: int, total: int, started: float, out: Path) -> None:
    elapsed = (time.time() - started) / 3600
    remaining = elapsed / finished * (total - finished)
    line = (f'[{finished}/{total}] {outcome["task"]}: {outcome["done"]} done, {len(outcome["failed"])} failed'
            f' | {elapsed:.1f} h elapsed, about {remaining:.1f} h left')
    print(line, flush=True)
    with (out / 'progress.log').open('a') as handle:
        handle.write(time.strftime('%Y-%m-%d %H:%M:%S ') + line + '\n')


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', default=f'runs/main_v5/{RUN_NAME}')
    parser.add_argument('--out', default='runs/followup_v042')
    parser.add_argument('--library', default=None, help='syntrustbench_campaign folder (auto-detected by hash)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2, help='threads per worker')
    parser.add_argument('--bases', type=int, default=50, help='number of audit bases, in the frozen audit order')
    parser.add_argument('--supplementary', type=int, default=150, help='fresh null runs to score (0 to skip)')
    parser.add_argument('--skip-injections', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='verify inputs and print the plan only')
    parser.add_argument('--check', action='store_true', help='rescore one honest audit run and compare, then stop')
    parser.add_argument('--allow-version-mismatch', action='store_true')
    parser.add_argument('--only', default=None, help=argparse.SUPPRESS)  # comma-separated condition names; testing only
    args = parser.parse_args()

    configure_threads(args.threads)
    run = FrozenRun(Path(args.run_dir))
    library = find_library(run, args.library)
    sys.path.insert(0, str(library.parent))
    from syntrustbench_campaign.config import from_mapping

    # Environment: the numerical pins the campaign recorded.
    pins = json.loads((run.dir / 'numerical_environment.json').read_text())
    versions = {p: importlib.metadata.version(p) for p in VERSION_PACKAGES}
    mismatched = {p: (pins[p], versions[p]) for p in VERSION_PACKAGES if pins.get(p) and versions[p].split('+')[0] != pins[p]}
    if mismatched and not args.allow_version_mismatch:
        raise SystemExit(f'Package versions differ from main_v5 (expected, installed): {mismatched}')

    # Configuration and frozen model.
    cfg = from_mapping(json.loads((run.dir / 'configuration.json').read_text()))
    if not RUN_NAME.endswith(cfg.fingerprint[:16]):
        raise SystemExit('configuration.json does not reproduce the run fingerprint.')
    freeze, freeze_hashes = run.result('calibrate/model')
    if run.state('calibrate/model')['fingerprint'] != cfg.fingerprint:
        raise SystemExit('Calibration model belongs to a different campaign fingerprint.')
    run.verify(freeze['model_path'], freeze_hashes.get(freeze['model_path']))
    if sha256(freeze['model_path']) != freeze['sha256']:
        raise SystemExit('Frozen calibration model hash mismatch.')
    model = json.loads(Path(freeze['model_path']).read_text())
    audit_ids = list(model['audit_ids'])
    if len(audit_ids) != 50 or set(audit_ids) & set(model['fit_ids']):
        raise SystemExit('Unexpected fit/audit assignment in the frozen model.')

    out = Path(args.out).resolve()
    if args.check:
        check(run, model, cfg, audit_ids[0], args.threads)
        return

    bases = audit_ids[:max(0, min(args.bases, 50))] if not args.skip_injections else []
    plan = conditions()
    if args.only:
        wanted = set(args.only.split(','))
        plan = [c for c in plan if c['name'] in wanted]
        if len(plan) != len(wanted):
            raise SystemExit(f'Unknown condition in --only: {sorted(wanted - {c["name"] for c in plan})}')
    repeats = list(range(100, 100 + max(0, args.supplementary)))
    # Verify every frozen input before any work starts.
    for base_id in bases:
        roles, data_hashes = run.result(f'null/data/{base_id:02d}')
        context, context_hashes = run.result(f'null/context/{base_id:02d}')
        for name in ('train', 'reference', 'positive', 'test', 'floor'):
            run.verify(roles['paths'][name], data_hashes.get(roles['paths'][name]))
        run.verify(context['context_path'], context_hashes.get(context['context_path']))
    if repeats:
        cohort, cohort_hashes = run.result('prepare/cohort')
        for path in list(cohort['paths'].values()) + [cohort['patient_audit']]:
            run.verify(path, cohort_hashes.get(path))

    script_sha = sha256(__file__)
    manifest_path = out / 'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get('script_sha256') != script_sha or previous.get('model_sha256') != freeze['sha256']:
            raise SystemExit(f'{out} was started with a different script or model; use a new --out folder.')
    tables = len(bases) * len(plan)
    print(f'Plan: {len(bases)} audit bases x {len(plan)} conditions = {tables} flawed tables; '
          f'{len(repeats)} supplementary null runs. Workers {args.workers} x {args.threads} threads.')
    print(f'Library: {library}\nFrozen model: {freeze["model_path"]} ({freeze["sha256"][:12]})\nOutput: {out}')
    hours = (tables * 2.6 + len(repeats) * 3.5) / 60 / max(1, args.workers)
    print(f'Rough duration: {hours:.1f} hours (from main_v5 timings; varies with the machine).')
    if args.dry_run:
        print('Dry run: every frozen input verified; nothing was evaluated.')
        return

    (out / 'tmp').mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, {
        'protocol': PROTOCOL, 'script_sha256': script_sha, 'model_sha256': freeze['sha256'],
        'model_path': freeze['model_path'], 'run_dir': str(run.dir), 'library': str(library),
        'configuration_fingerprint': cfg.fingerprint, 'audit_order': audit_ids, 'bases': bases,
        'conditions': plan, 'supplementary_repeats': repeats, 'versions': versions,
        'python': platform.python_version(), 'platform': platform.platform(),
        'workers': args.workers, 'threads': args.threads, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})

    common = {'run_dir': str(run.dir), 'out': str(out), 'model_path': freeze['model_path'],
              'model_sha256': freeze['sha256'], 'threads': args.threads}
    tasks = [(run_base, {**common, 'base': b, 'conditions': plan}) for b in bases]
    tasks += [(run_supplementary, {**common, 'repeat': r}) for r in repeats]
    started, finished, failures = time.time(), 0, []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=init_worker,
                             initargs=(str(library.parent), args.threads)) as pool:
        futures = {pool.submit(function, payload): payload for function, payload in tasks}
        try:
            completed = as_completed(futures)
            for future in completed:
                finished += 1
                outcome = collect(future, futures)
                failures += [f'{outcome["task"]}: {f}' for f in outcome['failed']]
                report(outcome, finished, len(tasks), started, out)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print('\nStopped. Finished tables are saved; run the same command again to resume.', flush=True)
            raise SystemExit(130)
    shutil.rmtree(out / 'tmp', ignore_errors=True)
    atomic_json(out / 'completion.json', {'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                          'failures': failures, 'tasks': len(tasks)})
    if failures:
        print(f'{len(failures)} item(s) failed; tracebacks are in {out / "errors"}. Rerun the same command to retry them.')
        raise SystemExit(1)
    print('All follow-up tables completed.')


if __name__ == '__main__':
    main()
