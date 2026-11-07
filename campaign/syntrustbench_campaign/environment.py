"""Pinned numerical environment and frozen official checkout verification."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import COMMIT, ExperimentConfig
from .core import file_hash, read_json, utc, write_json

PINS = (('numpy', '2.3.4'), ('pandas', '2.3.3'), ('scipy', '1.16.3'),
        ('scikit-learn', '1.7.2'), ('torch', '2.7.1'), ('sdv', '1.37.4'),
        ('ctgan', '0.13.0'), ('rdt', '2.1.0'), ('copulas', '0.15.0'),
        ('sdmetrics', '0.33.0'), ('pyyaml', '6.0.2'), ('matplotlib', '3.10.3'),
        ('smartnoise-synth', '1.0.8'), ('mbi', '1.1.0'), ('jax', '0.6.2'),
        ('jaxlib', '0.6.2'), ('optax', '0.2.5'), ('chex', '0.1.90'),
        ('opendp', '0.14.2'), ('smartnoise-sql', '1.0.10'),
        ('networkx', '3.4.2'), ('disjoint-set', '0.9.0'), ('cloudpickle', '3.1.2'))

def _withheld() -> str:
    # The public repository of the frozen v0.3 release is withheld during double-blind review.
    # Set FROZEN_V030_REPOSITORY to its URL to rerun the full campaign; the analysis of saved
    # results uses third_party/syntrustbench_v030 and needs no network access.
    raise RuntimeError('Set FROZEN_V030_REPOSITORY to the public URL of the frozen v0.3 release.')


class Environment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def setup(self) -> dict:
        if sys.version_info < (3, 11):
            raise RuntimeError('Python >=3.11 required.')
        checkout = Path(self.cfg.execution.checkout_dir).resolve()
        if self.cfg.execution.install_dependencies:
            subprocess.run([sys.executable, '-m', 'pip', 'install', *[f'{k}=={v}' for k, v in PINS]], check=True)
            if not checkout.exists():
                checkout.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['git', 'clone', os.environ.get('FROZEN_V030_REPOSITORY') or _withheld(), str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', COMMIT], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(checkout)], check=True)
            importlib.invalidate_caches()
        if not (checkout / '.git').exists():
            raise RuntimeError('Frozen checkout missing. Enable install_dependencies in the configuration cell.')
        head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if head != COMMIT:
            raise RuntimeError('Frozen checkout commit mismatch.')
        if subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise RuntimeError('Tracked files changed in the frozen checkout.')
        import syntrustbench
        installed = Path(syntrustbench.__file__).resolve().parent
        hashes = {}
        for path in sorted((checkout / 'src/syntrustbench').rglob('*.py')):
            relative = path.relative_to(checkout / 'src/syntrustbench')
            target = installed / relative
            if not target.exists() or file_hash(target) != file_hash(path):
                raise RuntimeError(f'Installed frozen source differs: {relative}; restart kernel after install.')
            hashes[str(relative)] = file_hash(path)
        versions = {name: importlib.metadata.version(name) for name, _ in PINS}
        if {k:v.split('+',1)[0] for k,v in versions.items()} != dict(PINS):
            raise RuntimeError(f'Numerical package pins differ: {versions}')
        root = self.cfg.root
        root.mkdir(parents=True, exist_ok=True)
        lock = root / 'numerical_environment.json'
        if lock.exists() and read_json(lock) != versions:
            raise RuntimeError('Numerical environment changed; refusing campaign resume.')
        write_json(lock, versions)
        source = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.iterdir())
                  if p.is_file() and p.suffix in ('.py','.txt')}
        code_lock = root / 'implementation_hashes.json'
        if code_lock.exists() and read_json(code_lock) != source:
            raise RuntimeError('Campaign implementation changed; use a new output_root.')
        write_json(code_lock, source)
        import torch
        manifest = {'time': utc(), 'platform': platform.platform(), 'python': platform.python_version(),
                    'cpu_count': os.cpu_count(), 'cuda_available': torch.cuda.is_available(),
                    'device': 'CUDA' if self.cfg.generators.allow_cuda and torch.cuda.is_available() else 'CPU',
                    'packages': versions, 'git_commit': head, 'installed_source_hashes': hashes,
                    'scientific_hash': self.cfg.fingerprint, 'configuration': asdict(self.cfg),
                    'implementation_hashes': source}
        write_json(root / 'sessions' / (utc().replace(':', '-') + '.json'), manifest)
        write_json(root / 'configuration.json', asdict(self.cfg))
        return manifest


def self_test() -> None:
    assert dict(PINS)['sdv'] == '1.37.4' and len(dict(PINS)) == len(PINS)
