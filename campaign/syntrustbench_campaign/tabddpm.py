"""Optional Synthcity adapter: pinned subprocess environment, explicit unavailability."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pandas as pd

from .config import ExperimentConfig
from .core import UnavailableAnalysis, file_hash, read_json, save_frame, write_json
from .generators import Generator


class TabDDPMEnvironment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.requirements = Path(__file__).with_name('tabddpm-requirements.txt')
        self.worker = Path(__file__).with_name('tabddpm_worker.py')
        self.digest = file_hash(self.requirements)
        self.root = Path(cfg.execution.output_root).resolve()/'optional_environments'/('tabddpm-'+self.digest[:16])
        self.python = self.root/'venv/bin/python'

    def run(self, operation: str, request: dict, folder: Path) -> dict:
        req = write_json(folder/f'{operation}_request.json',request)
        response = folder/f'{operation}_response.json'
        env = dict(os.environ, PYTHONHASHSEED=str(request.get('seed',0)),
                   OMP_NUM_THREADS=str(self.cfg.execution.threads_per_worker),
                   OPENBLAS_NUM_THREADS=str(self.cfg.execution.threads_per_worker))
        env.pop('PYTHONPATH',None)
        with (folder/f'{operation}_subprocess.log').open('w') as log:
            subprocess.run([str(self.python),'-I',str(self.worker),operation,req,str(response)],
                cwd=str(folder),env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                timeout=self.cfg.generators.tabddpm_timeout_seconds)
        return read_json(response)

    def ensure(self) -> dict:
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'build.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            failure = self.root/'unavailable.json'
            if failure.exists() and not self.cfg.execution.retry_failed:
                raise UnavailableAnalysis(read_json(failure)['reason'])
            attempt = self.root/('check-'+uuid.uuid4().hex)
            attempt.mkdir()
            try:
                if not (self.root/'environment.json').exists():
                    if not self.cfg.generators.tabddpm_build_environment:
                        raise UnavailableAnalysis('TabDDPM environment absent and automatic build disabled.')
                    with (attempt/'build.log').open('w') as log:
                        if not self.python.exists():
                            subprocess.run([self.cfg.generators.tabddpm_python,'-m','venv',str(self.root/'venv')],
                                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                        subprocess.run([str(self.python),'-c',
                            'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 required"'],
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
                        # Every resolved dependency is version-pinned with allowed wheel/sdist hashes.
                        subprocess.run([str(self.python),'-m','pip','install','--require-hashes','--no-deps',
                            '-r',str(self.requirements)],stdout=log,stderr=subprocess.STDOUT,
                            check=True,timeout=self.cfg.generators.tabddpm_timeout_seconds)
                        subprocess.run([str(self.python),'-m','pip','check'],
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
                current = self.run('probe',{},attempt)
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                reason = f'TabDDPM Python 3.10 environment unavailable: {type(exc).__name__}: {exc}; logs: {attempt}'
                logs = sorted(attempt.glob('*.log'),key=lambda p:p.stat().st_mtime)
                if logs:
                    reason += '\nDiagnostic tail: ' + logs[-1].read_text(errors='replace')[-2000:]
                write_json(failure,{'status':'Unavailable','reason':reason})
                raise UnavailableAnalysis(reason) from exc
            snapshot = self.root/'environment.json'
            if snapshot.exists() and read_json(snapshot) != current:
                raise UnavailableAnalysis('TabDDPM environment/source drift; use a new isolated environment.')
            write_json(snapshot,current)
            if failure.exists():
                failure.unlink()
            return {'requirements_sha256':self.digest,'worker_sha256':file_hash(self.worker),
                    'manifest':current,'manifest_sha256':file_hash(snapshot)}


class TabDDPMGenerator(Generator):
    def synthesizer_class(self):
        return 'synthcity.plugins.generic.plugin_ddpm.TabDDPMPlugin (isolated)'

    def fit(self, train: pd.DataFrame) -> TabDDPMGenerator:
        self.environment = TabDDPMEnvironment(self.cfg)
        self.folder = Path(self.artifact_dir or self.cfg.root/('manual_tabddpm-'+uuid.uuid4().hex)).resolve()
        self.folder.mkdir(parents=True,exist_ok=True)
        try:
            provenance = self.environment.ensure()
        except UnavailableAnalysis:
            logs = {str(p):file_hash(p) for p in sorted(self.environment.root.rglob('*.log'))}
            write_json(self.folder/'optional_environment_unavailable.json',
                       {'requirements_sha256':self.environment.digest, 'build_log_hashes':logs})
            raise
        input_path = save_frame(train,self.folder/'tabddpm_train.csv')
        self.schema = list(train.columns)
        self.model_path = self.folder/'tabddpm_model.pkl'
        small = self.smoke or self.cfg.execution.smoke_mode
        self.request = {'input_csv':input_path,'model':str(self.model_path),
            'workspace':str(self.folder/'synthcity_workspace'),
            'categorical':list(self.cfg.data.categorical),'continuous':list(self.cfg.data.continuous),
            'columns':self.schema,'target':self.cfg.data.target,
            'threads':self.cfg.execution.threads_per_worker,
            'seed':self.seed,'epochs':1 if small else self.cfg.generators.tabddpm_epochs,
            'timesteps':10 if small else self.cfg.generators.tabddpm_timesteps}
        self.timing = self.environment.run('fit',self.request,self.folder)
        self.timing['parameters'] = {k:self.request[k] for k in ('epochs','timesteps','threads')}
        self.provenance.update(isolated_environment=provenance,
            preprocessing='Categorical missing token; numeric missing -> 0; no changes to other generators.')
        self.model = str(self.model_path)
        return self

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError('Fit before sampling.')
        output = self.folder/'tabddpm_sample.csv'
        self.timing.update(self.environment.run('sample',dict(self.request,n=n,seed=seed,output_csv=str(output)),self.folder))
        frame = pd.read_csv(output,keep_default_na=False,na_values=[''],dtype={c:'object' for c in self.cfg.data.categorical})
        if len(frame)!=n or list(frame.columns)!=self.schema:
            raise ValueError('TabDDPM CSV exchange violated its schema contract.')
        return frame

    def save(self, path: Path) -> None:
        shutil.copyfile(self.model_path,path)


def self_test() -> None:
    cfg = ExperimentConfig()
    assert not cfg.generators.enable_tabddpm and not cfg.generators.enable_dpmst
