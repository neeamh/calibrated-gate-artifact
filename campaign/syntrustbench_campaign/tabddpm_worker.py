"""Isolated Python 3.10 entry point. No imports from the main SDV environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


def provenance() -> dict:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('TabDDPM requires isolated Python 3.10.')
    import synthcity
    from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin
    assert TabDDPMPlugin.name() == 'ddpm'
    package = Path(synthcity.__file__).parent
    return {'python':platform.python_version(),
            'packages':{d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()},
            'synthcity_source_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(package.rglob('*.py'))}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('probe','fit','sample'))
    parser.add_argument('request')
    parser.add_argument('response')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if args.operation == 'probe':
        result = provenance()
    else:
        import numpy as np
        import pandas as pd
        import torch
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
        from synthcity.utils.serialization import load_from_file, save_to_file
        from synthcity.utils.reproducibility import enable_reproducible_results
        torch.set_num_threads(request['threads'])
        enable_reproducible_results(request['seed'])
        started = time.perf_counter()
        if args.operation == 'fit':
            data = pd.read_csv(request['input_csv'],keep_default_na=False,na_values=[''],
                               dtype={c:'object' for c in request['categorical']})
            # Optional plugin cannot consume missing numeric/categorical values.
            # Fixed imputation, unrelated to DP and confined to its own training copy.
            for c in request['categorical']:
                data[c] = data[c].fillna('__MISSING__').astype(str)
            for c in request['continuous']:
                data[c] = pd.to_numeric(data[c],errors='raise').fillna(0.)
            model = Plugins(categories=['generic']).get('ddpm', n_iter=request['epochs'],
                num_timesteps=request['timesteps'], batch_size=min(256,len(data)),
                random_state=request['seed'], device=torch.device('cpu'),
                is_classification=True, workspace=Path(request['workspace']))
            model.fit(GenericDataLoader(data,target_column=request['target']))
            save_to_file(request['model'],model)
            result = {'fit_seconds':time.perf_counter()-started,'device':'CPU'}
        else:
            model = load_from_file(request['model'])
            output = model.generate(request['n'],random_state=request['seed']).dataframe()
            if len(output)!=request['n'] or set(output.columns)!=set(request['columns']):
                raise ValueError('TabDDPM returned wrong schema or row count.')
            for c in request['categorical']:
                output[c] = output[c].replace('__MISSING__',np.nan)
            output[request['columns']].to_csv(request['output_csv'],index=False)
            result = {'sample_seconds':time.perf_counter()-started,'sample_seed':request['seed']}
    Path(args.response).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
