"""Resumable orchestration of every specified experiment stage."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ExperimentConfig, STAGES, DIMENSIONS, seed_for
from .core import JobLedger, JobRunner, UnavailableAnalysis, file_hash, read_json, save_frame, utc, write_json
from .data import CohortBuilder, RoleAssigner, read_analytic, split_three, stratified_indices
from .evaluation import OfficialProfiler, build_context, evaluate_v04
from .generators import generator, seed_all
from .perturbations import strategy
from .calibration import NullCalibrator, CalibratedEvaluator
from .judges import quality_report
from .reporting import Analysis, Reporter, collect_profiles, judge_artifact


def worker(cfg: ExperimentConfig, key: str, kind: str, args: dict,
           required: tuple[str, ...], watched: tuple[str, ...]) -> dict | None:
    """Importable spawn-safe worker. Writes only its job attempt/state plus append-only events."""
    ledger = JobLedger(cfg.root, cfg.fingerprint)
    runner = JobRunner(ledger, cfg.execution.retry_failed)

    def operation(folder):
        for dep in required: ledger.result(dep)
        os.environ['LOKY_MAX_CPU_COUNT'] = str(cfg.execution.threads_per_worker)
        seed_all(seed_for('job', key))
        import torch
        torch.set_num_threads(cfg.execution.threads_per_worker)
        with threadpool_limits(limits=cfg.execution.threads_per_worker):
            return Operations(cfg, ledger).execute(kind, args, folder)

    inputs = [ledger.state_path(k) for k in dict.fromkeys(required + watched)]
    inputs += [Path(p) for p in args.get('input_files', [])]
    return runner.run(key, operation, inputs)


class Operations:
    def __init__(self, cfg: ExperimentConfig, ledger: JobLedger):
        self.cfg, self.ledger = cfg, ledger

    def execute(self, kind: str, args: dict, folder: Path) -> dict:
        methods = {'cohort': self.cohort, 'roles': self.roles, 'null_roles': self.null_roles,
                   'context': self.context, 'generation': self.generation, 'control': self.control,
                   'injection': self.injection, 'profile': self.profile, 'alias': self.alias,
                   'calibrate': self.calibrate, 'judges': self.judges, 'analysis': self.analysis,
                   'report': self.report, 'smoke': self.smoke}
        return methods[kind](args, folder)

    def cohort(self, args, folder):
        return CohortBuilder(self.cfg).build(folder)

    def roles(self, args, folder):
        return RoleAssigner(self.cfg).main(self.ledger.result('prepare/cohort'), folder)

    def null_roles(self, args, folder):
        return RoleAssigner(self.cfg).null(self.ledger.result('prepare/cohort'), args['repeat'], folder)

    def context(self, args, folder):
        return build_context(self.cfg, self.ledger.result(args['roles_key']), folder)

    def generation(self, args, folder):
        roles = self.ledger.result('prepare/roles')
        train = read_analytic(roles['paths']['train'], self.cfg)
        model = generator(args['generator'], self.cfg, args['seed'])
        model.artifact_dir = folder
        full = model.fit(train).sample(len(train), seed_for(args['generator'], args['seed'], 'sampling'))
        full_path = save_frame(full, folder / 'synthetic_full.csv')
        # Use the same sampled candidate for all judges and official/v0.4 evaluations.
        rows = np.random.default_rng(seed_for(args['generator'], args['seed'], 'size_match')).choice(len(full), roles['n'], replace=False)
        candidate = save_frame(full.iloc[rows].reset_index(drop=True), folder / 'synthetic_matched.csv')
        read_analytic(candidate, self.cfg)
        model.save(folder / 'generator.pkl')
        write_json(folder / 'generator_provenance.json', model.provenance)
        return {'candidate_path': candidate, 'full_path': full_path, 'rows': len(full), 'matched_rows': roles['n'],
                'matched_positions': rows.tolist(), 'generator': args['generator'], 'seed': args['seed'],
                'train_sha256': file_hash(roles['paths']['train']), 'timing': model.timing,
                'generator_provenance': model.provenance}

    def control(self, args, folder):
        roles = self.ledger.result('prepare/roles')
        train = read_analytic(roles['paths']['train'], self.cfg)
        base = read_analytic(roles['paths']['positive' if args['control'] == 'positive' else 'floor'], self.cfg)
        seed = seed_for('control', args['control'], args['seed'])
        if args['control'] == 'positive':
            candidate = base.sample(n=roles['n'], random_state=seed).reset_index(drop=True)
            audit = {'seed': seed, 'reuses_fixed_P_pool': True, 'rows': len(candidate)}
        else:
            candidate, audit = strategy(args['control'], self.cfg, train).apply(base, 1., seed)
        return {'candidate_path': save_frame(candidate, folder / 'candidate.csv'), 'audit': audit}

    def injection(self, args, folder):
        baseline = self.ledger.result(args['baseline_data'])
        roles = self.ledger.result('prepare/roles')
        base = read_analytic(baseline['candidate_path'], self.cfg)
        train = read_analytic(roles['paths']['train'], self.cfg)
        seed = seed_for('injection', args['baseline'], args['family'])
        candidate, audit = strategy(args['family'], self.cfg, train).apply(base, args['dose'], seed)
        path = folder / 'candidate.csv'
        if args['dose'] == 0:
            shutil.copyfile(baseline['candidate_path'], path)
            if file_hash(path) != file_hash(baseline['candidate_path']):
                raise RuntimeError('Dose-zero bytes changed.')
        else:
            save_frame(candidate, path)
        previous = args.get('previous_data')
        if previous:
            prior = self.ledger.result(previous)['audit']
            before, after = {tuple(x) for x in prior['changed_cells']}, {tuple(x) for x in audit['changed_cells']}
            if not before <= after:
                raise RuntimeError('Injection changes are not nested.')
        write_json(folder / 'injection_audit.json', audit)
        return {'candidate_path': str(path), 'audit': audit, 'baseline': args['baseline'],
                'family': args['family'], 'dose': args['dose'], 'seed': seed,
                'known_bad': args['dose'] > 0 and audit['changed_cell_count'] > 0}

    def profile(self, args, folder):
        roles = self.ledger.result(args['roles_key'])
        context = self.ledger.result(args['context_key'])
        labels = dict(args['labels'])
        if args.get('data_key'):
            data = self.ledger.result(args['data_key'])
            candidate_path = data['candidate_path']
            if labels['kind'] == 'injection': labels['known_bad'] = data['known_bad']
        else:
            candidate_path = roles['paths']['positive']
        candidate = read_analytic(candidate_path, self.cfg)
        if len(candidate) != roles['n']:
            raise ValueError('Candidate is not size matched to reference.')
        started = time.perf_counter()
        official = OfficialProfiler(self.cfg).evaluate(roles, candidate, folder / 'official_v03')
        official_seconds = time.perf_counter()-started
        started = time.perf_counter()
        v04 = evaluate_v04(self.cfg, context['context_path'], roles, candidate, labels['table'])
        v04_seconds = time.perf_counter()-started
        reference = read_analytic(roles['paths']['reference'], self.cfg)
        external = quality_report(reference, candidate, self.cfg)
        payload = {'labels': labels, 'official': official, 'v04': v04, 'sdmetrics': external,
                   'official_folder': str(folder / 'official_v03'), 'availability': 'Available',
                   'input_hashes': {k: file_hash(p) for k, p in roles['paths'].items()} | {'candidate': file_hash(candidate_path)},
                   'rows': {'candidate': len(candidate), **roles['role_counts']},
                   'timing': {'official_seconds': official_seconds, 'v04_seconds': v04_seconds}}
        if args.get('data_key') and 'generator_provenance' in data:
            payload['generator_provenance'] = data['generator_provenance']
        if args.get('calibration_key'):
            calibration = self.ledger.result(args['calibration_key'])
            model = read_json(calibration['model_path'])
            if file_hash(calibration['model_path']) != calibration['sha256']:
                raise RuntimeError('Frozen calibration changed.')
            payload['calibration_sha256'] = calibration['sha256']
            payload['calibrated'] = {str(a): CalibratedEvaluator(model).apply(v04, a) for a in self.cfg.calibration.target_rates}
        return {'profile_path': write_json(folder / 'profile.json', payload)}

    def alias(self, args, folder):
        data = self.ledger.result(args['data_key'])
        original = self.ledger.result(args['baseline_profile'])
        profile = read_json(original['profile_path'])
        profile['labels'] = args['labels']
        profile['dose_zero_reuses_profile'] = {'path': original['profile_path'], 'sha256': file_hash(original['profile_path'])}
        return {'profile_path': write_json(folder / 'profile.json', profile)}

    def calibrate(self, args, folder):
        calibrator = NullCalibrator(self.cfg.calibration)
        fit, audit = calibrator.assignment(self.cfg.execution.smoke_mode)
        full = {r: read_json(self.ledger.result(f'null/profile/{r:02d}')['profile_path']) for r in fit}
        # Fit-run native SDMetrics scores calibrate the SDMetrics judge to the same honest-data rate.
        model = calibrator.fit({r: p['v04'] for r, p in full.items()}, self.cfg.execution.smoke_mode,
                               sdmetrics_scores={r: p['sdmetrics'].get('score') for r, p in full.items()})
        path = write_json(folder / 'empirical_maxT.json', model)
        save_frame(pd.DataFrame(model['raw_cutoffs']), folder / 'raw_cutoffs.csv')
        save_frame(pd.DataFrame(model['excluded']), folder / 'unavailable_calibration_metrics.csv')
        record = {'model_path': path, 'sha256': file_hash(path), 'frozen_at': utc(),
                  'fit_ids': fit, 'audit_ids': audit, 'blocked_dimensions': model['blocked_dimensions']}
        write_json(folder / 'freeze_record.json', record)
        return record

    def judges(self, args, folder):
        # Planned tables plus held-out honest audit runs: every judge is scored on both.
        profiles = collect_profiles(self.ledger, args['descriptors'] + args.get('audit_descriptors', []))
        model = read_json(self.ledger.result('calibrate/model')['model_path'])
        return judge_artifact(self.cfg, profiles, model, folder)

    def analysis(self, args, folder):
        profiles = collect_profiles(self.ledger, args['descriptors'])
        nulls = collect_profiles(self.ledger, args['null_descriptors'])
        model = read_json(self.ledger.result('calibrate/model')['model_path'])
        judges = read_json(self.ledger.result('judges/all')['judge_path'])
        return Analysis(self.cfg, profiles, nulls, model, judges).build(self.ledger.result('prepare/cohort'), folder,
                                                                      self.ledger.result('prepare/roles'))

    def report(self, args, folder):
        return Reporter(self.cfg).build(self.ledger.result('analysis/tables'), folder)

    def smoke(self, args, folder):
        from .verification import run_checks
        checks = run_checks(folder)
        if self.cfg.execution.smoke_mode:
            return {'checks': checks, 'scope': 'Small-campaign stage; remaining stages exercise all adapters.'}
        # Build a separate smoke campaign entirely from T+C. Never read main E.
        cohort = self.ledger.result('prepare/cohort')
        role = RoleAssigner(self.cfg).null(cohort, 'smoke_preflight', folder / 'development_roles')
        frames = {'real_train': read_analytic(role['paths']['train'], self.cfg),
                  'calibration_pool': pd.concat([read_analytic(role['paths'][k], self.cfg) for k in ('reference','positive')], ignore_index=True),
                  'real_test': read_analytic(role['paths']['test'], self.cfg)}
        audit = pd.read_csv(role['patient_audit'])
        ids = {'real_train': audit[audit.role == 'train'].patient_id_sha256.tolist(),
               'calibration_pool': sum([audit[audit.role == k].patient_id_sha256.tolist() for k in ('reference','positive')], []),
               'real_test': audit[audit.role == 'test'].patient_id_sha256.tolist()}
        small = folder / 'frozen_smoke'; small.mkdir()
        paths, patient_rows = {}, []
        for name, frame in frames.items():
            positions = stratified_indices(frame[self.cfg.data.target], min(len(frame), self.cfg.execution.smoke_rows), seed_for('smoke_cap',name))
            paths[name] = save_frame(frame.iloc[positions], small / f'{name}.csv')
            patient_rows.extend({'role': name, 'analytic_row': i, 'patient_id_sha256': ids[name][p]} for i,p in enumerate(positions))
        patient_path = save_frame(pd.DataFrame(patient_rows), small / 'patient_role_audit.csv')
        manifest_path = write_json(small / 'split_manifest.json', {'split_hashes': {n: file_hash(p) for n,p in paths.items()},
            'patient_audit': patient_path, 'patient_audit_sha256': file_hash(patient_path)})
        mini = replace(self.cfg, data=replace(self.cfg.data, frozen_dir=str(small), frozen_manifest=manifest_path),
                       execution=replace(self.cfg.execution, smoke_mode=True, dry_run=False, workers=1,
                                         stages=STAGES, output_root=str(folder / 'smoke_campaign')))
        from .environment import Environment
        mini = replace(mini, execution=replace(mini.execution, install_dependencies=False))
        Environment(mini).setup()
        campaign = Campaign(mini)
        for stage in STAGES: campaign.run_stage(stage)
        audit = campaign.completion_audit()
        optional_unavailable = [j['key'] for jobs in campaign.plan.values() for j in jobs
                                if j['key'].startswith('clean/TabDDPM/')
                                and campaign.ledger.state(j['key'])['state'] == 'unavailable']
        if audit['failed'] or audit['unavailable'] > len(optional_unavailable) or audit['planned']:
            raise UnavailableAnalysis('Smoke campaign incomplete; inspect its completion audit before full work.')
        return {'checks': checks, 'smoke_campaign': str(mini.root), 'optional_unavailable': optional_unavailable,
                'scope': 'All stage paths on development-only data; not paper evidence.'}


class Campaign:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.ledger = JobLedger(cfg.root, cfg.fingerprint)
        self.descriptors, self.null_descriptors, self.plan = self._plan()
        expected = {stage: [j['key'] for j in jobs] for stage, jobs in self.plan.items()}
        path = cfg.root / 'expected_jobs.json'
        if path.exists() and read_json(path) != expected:
            raise RuntimeError('Expected job plan changed; choose a new campaign directory.')
        write_json(path, expected)
        for jobs in self.plan.values():
            for j in jobs:
                if not self.ledger.state_path(j['key']).exists(): self.ledger.set(j['key'], 'planned')

    def _plan(self) -> tuple[list, list, dict]:
        cfg = self.cfg
        plan = {s: [] for s in STAGES}
        descriptors, null_descriptors = [], []

        def add(stage, key, kind, args=None, required=(), watched=()):
            plan[stage].append({'key': key, 'kind': kind, 'args': args or {},
                                'required': tuple(required), 'watched': tuple(watched)})

        def profile(stage, key, labels, data_key=None, roles='prepare/roles', context='clean/context', calibrated=True):
            required = [roles, context] + ([data_key] if data_key else []) + (['calibrate/model'] if calibrated else [])
            add(stage, key, 'profile', {'labels': labels, 'data_key': data_key, 'roles_key': roles,
                                      'context_key': context, 'calibration_key': 'calibrate/model' if calibrated else None}, required)
            descriptor = {'profile_key': key, 'labels': labels}
            (descriptors if calibrated else null_descriptors).append(descriptor)

        if cfg.data.frozen_dir:
            inputs = [cfg.data.frozen_manifest] + [str(Path(cfg.data.frozen_dir)/f'{n}.csv') for n in ('real_train','calibration_pool','real_test')]
            manifest = read_json(cfg.data.frozen_manifest) if Path(cfg.data.frozen_manifest).exists() else {}
            audit = manifest.get('patient_audit', str(Path(cfg.data.frozen_dir).resolve() / 'patient_role_audit.csv'))
            if not Path(audit).is_absolute(): audit = str(Path(cfg.data.frozen_manifest).resolve().parent / audit)
            inputs.append(audit)
        else: inputs = [cfg.data.raw_csv]
        add('prepare', 'prepare/cohort', 'cohort', {'input_files': inputs})
        add('prepare', 'prepare/roles', 'roles', required=['prepare/cohort'])
        add('smoke', 'smoke/all_paths', 'smoke', required=['prepare/cohort'])
        fit, audit = NullCalibrator(cfg.calibration).assignment(cfg.execution.smoke_mode)
        for r in sorted(fit+audit):
            role, context = f'null/data/{r:02d}', f'null/context/{r:02d}'
            add('null', role, 'null_roles', {'repeat': r}, ['prepare/cohort','smoke/all_paths'])
            add('null', context, 'context', {'roles_key': role}, [role])
            labels = {'table': f'null/{r:02d}', 'kind': 'null', 'repeat': r,
                      'partition': 'fit' if r in fit else 'audit', 'truth': 'good', 'known_bad': False}
            profile('null', f'null/profile/{r:02d}', labels, roles=role, context=context, calibrated=False)
        add('calibrate', 'calibrate/model', 'calibrate', required=[f'null/profile/{r:02d}' for r in fit])
        add('clean', 'clean/context', 'context', {'roles_key':'prepare/roles'}, ['prepare/roles','calibrate/model'])
        seeds = cfg.generators.seeds[:1] if cfg.execution.smoke_mode else cfg.generators.seeds
        reps = cfg.positive_replicates[:1] if cfg.execution.smoke_mode else cfg.positive_replicates
        baseline_data, baseline_profiles = {}, {}
        for g in cfg.generators.enabled_names:
            for s in seeds:
                key = f'clean/{g}/{s}'; data_key = key+'/data'
                labels = {'table': key, 'kind': 'clean', 'generator': g, 'seed': s, 'truth':'unknown', 'known_bad':False}
                add('clean', data_key, 'generation', {'generator':g,'seed':s}, ['prepare/roles','calibrate/model'])
                profile('clean', key+'/profile', labels, data_key)
                if g == 'GaussianCopula': baseline_data[key], baseline_profiles[key] = data_key, key+'/profile'
        for control in ('positive','copy','negative'):
            for s in reps:
                key = f'control/{control}/{s}'; data_key = key+'/data'
                labels = {'table':key, 'kind':'control', 'control':control, 'seed':s,
                          'truth':'good' if control == 'positive' else 'bad', 'known_bad':control != 'positive'}
                add('controls', data_key, 'control', {'control':control, 'seed':s}, ['prepare/roles','calibrate/model'])
                profile('controls', key+'/profile', labels, data_key)
                if control == 'positive': baseline_data[key], baseline_profiles[key] = data_key, key+'/profile'
        for baseline, data_key in baseline_data.items():
            for family, all_doses in cfg.injections.doses:
                doses = all_doses[-1:] if cfg.execution.smoke_mode else all_doses
                previous = None
                for dose in (0.,)+doses:
                    key = f'injection/{baseline}/{family}/{dose:g}'
                    labels = {'table': key, 'kind':'injection', 'baseline':baseline,
                              'baseline_kind':'positive' if baseline.startswith('control') else 'GaussianCopula',
                              'family':family, 'dose':dose, 'truth':'bad' if dose else 'baseline', 'known_bad':dose > 0}
                    required = [data_key,'prepare/roles','calibrate/model'] + ([previous] if previous else [])
                    add('injections', key+'/data', 'injection', {'baseline_data':data_key, 'baseline':baseline,
                        'family':family, 'dose':dose, 'previous_data':previous}, required)
                    if dose == 0:
                        add('injections', key+'/profile', 'alias', {'labels':labels,'data_key':key+'/data',
                            'baseline_profile':baseline_profiles[baseline]}, [key+'/data',baseline_profiles[baseline]])
                        descriptors.append({'profile_key':key+'/profile','labels':labels})
                    else: profile('injections', key+'/profile', labels, key+'/data')
                    previous = key+'/data'
        watched = [d['profile_key'] for d in descriptors]
        audit_descriptors = [d for d in null_descriptors if d['labels']['partition'] == 'audit']
        add('judges', 'judges/all', 'judges', {'descriptors':descriptors, 'audit_descriptors':audit_descriptors},
            ['calibrate/model'], watched + [d['profile_key'] for d in audit_descriptors])
        add('analysis', 'analysis/tables', 'analysis', {'descriptors':descriptors, 'null_descriptors':null_descriptors},
            ['judges/all','prepare/cohort','prepare/roles','calibrate/model'], watched+[d['profile_key'] for d in null_descriptors])
        add('report', 'report/artifacts', 'report', required=['analysis/tables'])
        return descriptors, null_descriptors, plan

    def _execute(self, job):
        return worker(self.cfg, job['key'], job['kind'], job['args'], job['required'], job['watched'])

    def run_stage(self, stage: str) -> dict:
        if stage not in STAGES: raise ValueError(stage)
        if self.cfg.execution.dry_run or stage not in self.cfg.execution.stages:
            return {'stage':stage, 'state':'planned', 'jobs':len(self.plan[stage]), 'reason':'dry_run or disabled'}
        self.ledger.set('stage/'+stage, 'running')
        try:
            # Dependency-bearing data/context jobs run first. Only independent profiles fan out.
            profiles = [j for j in self.plan[stage] if j['kind'] == 'profile']
            other = [j for j in self.plan[stage] if j['kind'] != 'profile']
            for job in other: self._execute(job)
            if self.cfg.execution.workers > 1 and len(profiles) > 1:
                with ProcessPoolExecutor(max_workers=self.cfg.execution.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = [pool.submit(worker, self.cfg, j['key'], j['kind'], j['args'], j['required'], j['watched']) for j in profiles]
                    for future in futures: future.result()
            else:
                for job in profiles: self._execute(job)
            incomplete = [j['key'] for j in self.plan[stage] if self.ledger.state(j['key'])['state'] != 'completed']
            state = 'unavailable' if incomplete else 'completed'
            self.ledger.set('stage/'+stage, state, incomplete_jobs=incomplete)
            return {'stage':stage, 'state':state, 'jobs':len(self.plan[stage]), 'incomplete':len(incomplete)}
        except BaseException:
            self.ledger.set('stage/'+stage, 'failed', traceback=traceback.format_exc())
            self.ledger.append('failures.jsonl', {'stage':stage, 'traceback':traceback.format_exc()})
            raise

    def completion_audit(self) -> dict:
        counts = {s:0 for s in ('planned','running','completed','failed','unavailable')}
        stages, missing, unavailable_analyses = {}, [], []
        for stage, jobs in self.plan.items():
            summary = {s:0 for s in counts}
            for j in jobs:
                record = self.ledger.state(j['key']); state = record['state']
                if state == 'completed':
                    result = self.ledger.result(j['key'])
                    if j['kind'] in ('profile','alias'):
                        profile = read_json(result['profile_path'])
                        for dimension, value in profile['v04']['dimensions'].items():
                            if value['status'] == 'NotEvaluated':
                                unavailable_analyses.append({'job':j['key'], 'dimension':dimension, 'metric':None,
                                                             'reason':'Dimension has missing required evidence.'})
                            for name, metric in value['metrics'].items():
                                if metric.get('status') == 'NotEvaluated':
                                    unavailable_analyses.append({'job':j['key'], 'dimension':dimension, 'metric':name,
                                                                 'required':metric.get('required'), 'reason':metric.get('note','')})
                            unavailable_analyses.extend({'job':j['key'], 'dimension':dimension, 'metric':name,
                                                        'required':True, 'reason':'Reference wrapper found missing evidence.'}
                                                       for name in value.get('missing_required_evidence', []))
                summary[state] += 1; counts[state] += 1
                if state != 'completed': missing.append({'key':j['key'], 'state':state, 'reason':record.get('reason','')})
            stages[stage] = {'expected':len(jobs), **summary}
        audit = {'time':utc(), 'scientific_hash':self.cfg.fingerprint, 'smoke_only':self.cfg.execution.smoke_mode,
                 'stages':stages, **counts, 'incomplete':missing, 'unavailable_analyses':unavailable_analyses,
                 'interpretation':'Execution coverage is not scientific acceptance; inspect NotEvaluated metrics inside completed profiles.'}
        write_json(self.cfg.root / 'audits' / (utc().replace(':','-')+'.json'), audit)
        return audit


def self_test() -> None:
    assert STAGES[0] == 'prepare' and STAGES[-1] == 'report'
