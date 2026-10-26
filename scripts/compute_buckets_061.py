# Bucket every castor Slurm job into one psv per model set (the compute table).
# Usage: python scripts/compute_buckets_061.py <dir with raw/all_jobs_raw.psv, raw/logs_map.txt>
# Reads /work/harrychang08/castor/runs/008/driver/submissions.json. Output: runs/061/compute/.
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

S = sys.argv[1]
rows = [ln.split('|') for ln in Path(f'{S}/raw/all_jobs_raw.psv').read_text().splitlines()]
byid = {r[0]: r for r in rows}
logids = {ln.split('|')[0] for ln in Path(f'{S}/raw/logs_map.txt').read_text().splitlines()}
FIRST_CASTOR = 442347
castor = [r for r in rows if re.match(r'\d{3}-', r[1]) or r[0] in logids]
ledger = json.loads(Path('/work/harrychang08/castor/runs/008/driver/submissions.json').read_text())
grid_sft = {j for k, v in ledger.items() if k.endswith('|sft') for j in v}
grid_gj = {j for k, v in ledger.items() if not k.endswith('|sft') for j in v}
grid_refill = {r[0] for r in rows if r[1] == '008-refill' and int(r[0]) >= 517485}  # refill jobs from 2026-10-08T11:00
ladder = {ln.split('|')[0] for ln in Path('/home/harrychang08/castor/runs/016/sacct_jobs.psv').read_text().splitlines()}
def gpus(t):
    m = re.search(r'gres/gpu=(\d+)', t); return int(m.group(1)) if m else 0
def h(r): return int(r[3]) * gpus(r[4]) / 3600
def bucket(r):
    i, n = r[0], r[1]
    if re.fullmatch(r'011-i6198-d6198-seg\d|003-s2-i(6198-d4|4-d6198)-seg\d', n): return 'pretrain_reruns'
    if n.startswith('035-r2-seg'): return 'held_035'
    if n.startswith('006-train-'): return 'first_setup_twelve'
    if n.startswith('006-genjudge-'): return 'genjudge_first_setup_twelve'
    if n.startswith('009-train-'): return 'partial_reruns_posttrain'
    if n.startswith('009-gj-'): return 'genjudge_partial_reruns'
    if n.startswith('011-train-dclm-baseline-1B-replicate'): return 'same_seed_rerun_posttrain'
    if n.startswith('011-gj-dclm-baseline-1B-replicate'): return 'genjudge_same_seed_rerun'
    if i in grid_sft: return 'recipe_grid_train'
    if i in grid_gj: return 'genjudge_recipe_grid'
    if i in grid_refill: return 'recipe_grid_refill'
    if i in ladder: return 'ladder_train' if '-train-' in n else 'genjudge_ladder'
    if re.match(r'00[45]-|010-|012-', n) or n in ('024-train-A','024-train-B','024-gj-A','024-gj-B'): return 'checks_appC'
    if n.startswith(('024-', '025-')): return 'tests_appA'
    if re.match(r'009-eval|009-stage2-after|011-after|009-smoke-convert', n): return 'pretrain_rerun_evals'
    if n.startswith('003-'): return 'pretrain_checks_003'
    if re.search(r'smoke|mock|probe|^008-refill', n) and not n.startswith('020-'): return 'smokes_other'
    return 'other_analysis'
B = OrderedDict()
for r in castor:
    B.setdefault(bucket(r), []).append(r)
summary = {}
for b, rs in B.items():
    rs.sort(key=lambda r: int(r[0]))
    Path(f'{S}/{b}.psv').write_text(''.join('|'.join(r) + '\n' for r in rs))
    tot = sum(h(r) for r in rs); nc = sum(h(r) for r in rs if r[2] != 'COMPLETED')
    st = defaultdict(int)
    for r in rs: st[r[2].split()[0]] += 1
    summary[b] = {'n_jobs': len(rs), 'h200_h': round(tot, 4), 'h200_h_not_completed': round(nc, 4), 'states': dict(st),
                  'ids': [r[0] for r in rs]}
summary['_all_castor'] = {'n_jobs': len(castor), 'h200_h': round(sum(h(r) for r in castor), 4),
                          'h200_h_not_completed': round(sum(h(r) for r in castor if r[2] != 'COMPLETED'), 4)}
noncastor = [r for r in rows if r not in castor]
summary['_excluded_noncastor'] = {'n_jobs': len(noncastor), 'h200_h': round(sum(h(r) for r in noncastor), 4),
    'ids_ge_first_castor': [r[0]+':'+r[1] for r in noncastor if int(r[0]) >= FIRST_CASTOR]}
Path(f'{S}/summary.json').write_text(json.dumps(summary, indent=1))
for b, s in summary.items():
    print(b, {k: v for k, v in s.items() if k not in ('ids',)} if b != '_excluded_noncastor' else (s['n_jobs'], s['h200_h'], s['ids_ge_first_castor']))
