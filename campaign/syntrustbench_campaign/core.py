"""Atomic artifact handling and a transactional, hash-verifying job ledger."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd


class UnavailableAnalysis(RuntimeError):
    """Expected missing evidence, never a numerical/programming-error substitute."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str | Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as handle:
        json.dump(clean_json(value), handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    return str(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_frame(frame: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')
    return str(path)


def available_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


class Gate:
    """Non-compensatory precedence, including at the overall gate."""

    @staticmethod
    def aggregate(statuses) -> str:
        values = list(statuses)
        if not values:
            return 'NotEvaluated'
        if not set(values) <= {'Pass', 'Conditional', 'Fail', 'NotEvaluated'}:
            raise ValueError('Unknown evidence status.')
        return next(s for s in ('Fail', 'NotEvaluated', 'Conditional', 'Pass') if s in values)


class JobLedger:
    def __init__(self, root: Path, fingerprint: str):
        self.root, self.fingerprint = Path(root), fingerprint
        (self.root / 'states').mkdir(parents=True, exist_ok=True)

    def state_path(self, key: str) -> Path:
        return self.root / 'states' / (key.replace('/', '__') + '.json')

    def state(self, key: str) -> dict:
        path = self.state_path(key)
        return read_json(path) if path.exists() else {'key': key, 'state': 'planned'}

    def append(self, filename: str, record: dict) -> None:
        data = (json.dumps(clean_json({'time': utc(), **record}), allow_nan=False) + '\n').encode()
        descriptor = os.open(self.root / filename, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)

    def set(self, key: str, state: str, **details) -> dict:
        record = {'key': key, 'state': state, 'fingerprint': self.fingerprint,
                  'updated_at': utc(), **details}
        write_json(self.state_path(key), record)
        self.append('events.jsonl', record)
        return record

    def result(self, key: str) -> dict:
        record = self.state(key)
        if record['state'] != 'completed':
            raise UnavailableAnalysis(f'{key}: {record["state"]}; {record.get("reason", "completed dependency required")}')
        if record['fingerprint'] != self.fingerprint:
            raise RuntimeError(f'Campaign fingerprint changed: {key}')
        for path, expected in record['artifact_hashes'].items():
            if not Path(path).is_file() or file_hash(path) != expected:
                raise RuntimeError(f'Artifact changed or missing: {path}')
        for path, expected in record.get('input_hashes', {}).items():
            if not Path(path).is_file() or file_hash(path) != expected:
                raise RuntimeError(f'Input changed or missing: {path}')
        return record['result']


class JobRunner:
    def __init__(self, ledger: JobLedger, retry_failed: bool = False):
        self.ledger, self.retry_failed = ledger, retry_failed

    def run(self, key: str, operation: Callable[[Path], dict], inputs=()) -> dict | None:
        """Each job owns an attempt directory; retries never overwrite old evidence."""
        lock_path = self.ledger.state_path(key).with_suffix('.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = self.ledger.state(key)
            hashes = {str(Path(p).resolve()): file_hash(p) for p in inputs}
            if previous['state'] == 'unavailable' and not self.retry_failed:
                if previous.get('input_hashes', {}) != hashes:
                    raise RuntimeError(f'Unavailable job inputs changed: {key}; enable retry_failed to retry.')
                for path, expected in previous.get('artifact_hashes', {}).items():
                    if not Path(path).is_file() or file_hash(path) != expected:
                        raise RuntimeError(f'Unavailable artifact changed or missing: {path}')
                return None
            if previous['state'] == 'completed':
                if previous.get('input_hashes', {}) != hashes:
                    changed = set(hashes) | set(previous.get('input_hashes', {}))
                    changed = [p for p in changed if hashes.get(p) != previous.get('input_hashes', {}).get(p)]
                    refresh = (self.retry_failed and key in ('judges/all','analysis/tables','report/artifacts')
                               and all(Path(p).parent == self.ledger.root/'states' for p in changed))
                    if not refresh:
                        raise RuntimeError(f'Job inputs changed: {key}; use a new campaign.')
                    for path, expected in previous['artifact_hashes'].items():
                        if not Path(path).is_file() or file_hash(path) != expected:
                            raise RuntimeError(f'Artifact changed or missing: {path}')
                else:
                    return self.ledger.result(key)
            if previous['state'] in ('failed', 'running') and not self.retry_failed:
                raise RuntimeError(f'{key} was {previous["state"]}; inspect ledger before enabling retry_failed.')
            attempt = self.ledger.root / 'jobs' / key.replace('/', '__') / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            self.ledger.set(key, 'running', attempt=str(attempt), input_hashes=hashes)
            started = time.perf_counter()
            try:
                with (attempt / 'job.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = operation(attempt)
                write_json(attempt / 'result.json', result)
                artifacts = {str(p): file_hash(p) for p in sorted(attempt.rglob('*')) if p.is_file()}
                self.ledger.set(key, 'completed', result=result, artifact_hashes=artifacts,
                                input_hashes=hashes, seconds=time.perf_counter() - started)
                return result
            except UnavailableAnalysis as exc:
                write_json(attempt / 'unavailable.json', {'status': 'Unavailable', 'reason': str(exc)})
                artifacts = {str(p): file_hash(p) for p in sorted(attempt.rglob('*')) if p.is_file()}
                self.ledger.set(key, 'unavailable', reason=str(exc), attempt=str(attempt), input_hashes=hashes,
                                artifact_hashes=artifacts, seconds=time.perf_counter()-started)
                return None
            except BaseException:
                failure = traceback.format_exc()
                (attempt / 'failure_traceback.txt').write_text(failure)
                self.ledger.set(key, 'failed', attempt=str(attempt), input_hashes=hashes,
                                traceback_path=str(attempt / 'failure_traceback.txt'))
                self.ledger.append('failures.jsonl', {'key': key, 'traceback': failure})
                raise


def self_test() -> None:
    assert Gate.aggregate(['Pass', 'NotEvaluated']) == 'NotEvaluated'
    assert Gate.aggregate(['Fail', 'NotEvaluated']) == 'Fail'
    assert Gate.aggregate([]) == 'NotEvaluated'
