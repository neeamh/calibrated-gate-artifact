"""Per-dimension calibrated pass counts for the mechanism table (alpha 0.05), from saved evidence."""
import json, glob, sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = _ROOT / 'paper' / 'generated'
DATA = _ROOT / 'results' / 'main_v5' / 'analysis_tables'
PROFILES = _ROOT / 'results' / 'main_v5' / 'profiles' / 'jobs'
sys.path.insert(0, str(_ROOT / 'campaign'))
from syntrustbench_campaign.config import GATE_DIMENSIONS
from syntrustbench_campaign.calibration import CalibratedEvaluator
P = PROFILES
model = json.load(open(glob.glob(str(P / 'calibrate__model/*/empirical_maxT.json'))[0]))
ev = CalibratedEvaluator(model)
audit = set(model['audit_ids'])
out = {}
for f in glob.glob(str(P / '*/*/profile.json')):
    p = json.load(open(f)); lab = p['labels']
    if lab['kind'] == 'null' and lab['repeat'] in audit: key = 'honest|audit'
    elif lab['kind'] == 'control' and lab.get('control') == 'positive': key = 'positive|control'
    elif lab['kind'] == 'control' and lab.get('control') == 'copy': key = 'copy|all'
    elif lab['kind'] == 'injection' and lab.get('baseline_kind') == 'positive' and lab.get('dose', 0) > 0: key = f"{lab['family']}|{lab['dose']:g}"
    else: continue
    r = ev.apply(p['v04'], .05)['dimensions']
    o = out.setdefault(key, {'n': 0, **{d: 0 for d in GATE_DIMENSIONS}})
    o['n'] += 1
    for d in GATE_DIMENSIONS: o[d] += r[d]['status'] == 'Pass'
json.dump(out, open(HERE / 'mechanism_status.json', 'w'), indent=1)
for k in sorted(out): print(k, out[k])
