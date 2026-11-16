"""Rerun the whole study on REAL Qwen scores, in research_v10/real_qwen/.
The stand-in (student) results in research_v10/ stay untouched for comparison.

  python rerun_real_qwen.py            (from research_v10/, after the lab run is copied back)

Steps: base table -> merge real Qwen -> chunks (full, 84/128/256/512-day caps, per-writing) ->
5 repeats of C, O, B on full history, B at each cap, B deciding on every writing -> agg.json, horizons.json.
Works on Windows and Linux. Uses all CPU cores but one."""
import os, sys, shutil, subprocess, glob
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(HERE, 'real_qwen'); os.makedirs(WORK, exist_ok=True)
for f in glob.glob(os.path.join(HERE, '*.py')):
    if not f.endswith('rerun_real_qwen.py'): shutil.copy(f, WORK)
ENV = dict(os.environ, AIXPE_ROOT=ROOT, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
PY_ = sys.executable
CACHE = os.path.join(ROOT, 'private', 'qwen_rest', 'cohort_rest_PRIVATE.sqlite')
MERGE = os.path.join(ROOT, 'private', 'qwen_rest', 'merge_real_qwen.py')

def run(args, extra=None, log=None):
    env = dict(ENV, **(extra or {}))
    out = open(os.path.join(WORK, log), 'w') if log else None
    r = subprocess.run([PY_] + args, cwd=WORK, env=env, stdout=out or None, stderr=subprocess.STDOUT if out else None)
    if r.returncode: raise SystemExit(f'failed: {" ".join(args)} (see {log})')
    return log

assert os.path.exists(CACHE), f'copy the lab cache to {CACHE} first'
print('1/5 base table'); run(['base.py'])
print('2/5 merge real Qwen'); run([MERGE, CACHE, WORK])
print('3/5 chunks'); run(['chunks.py'])
for cap in (84, 128, 256, 512): run(['chunks.py'], {'CAPDAY': str(cap)})
run(['chunks.py'], {'WRITING': '1'})          # one decision row per writing
jobs = [(['seq_eval.py', v, f'{v}_r{r}', '-', str(r)], None, f'log_{v}_r{r}.txt') for r in range(5) for v in ('C', 'O', 'B')]
jobs += [(['seq_eval.py', 'B', f'cap{c}_B_r{r}', '-', str(r)], {'CHUNKPREF': f'chunkscap{c}'}, f'log_cap{c}_B_r{r}.txt')
         for c in (84, 128, 256, 512) for r in range(5)]
jobs += [(['seq_eval.py', 'B', f'W_B_r{r}', '-', str(r)], {'DECIDE': 'writing'}, f'log_W_B_r{r}.txt') for r in range(5)]
n = max(1, (os.cpu_count() or 2) - 1)
print(f'4/5 {len(jobs)} evaluation runs on {n} cores')
with ThreadPoolExecutor(n) as ex:
    for i, log in enumerate(ex.map(lambda j: run(*j), jobs), 1):
        print(f'   {i}/{len(jobs)} {log}', flush=True)
print('5/5 summaries'); run(['agg.py', 'C', 'O', 'B', 'W_B'], log='agg.txt'); run(['horizons.py'], log='horizons.txt')
print('DONE. Results in', WORK, ': agg.json, horizons.json, agg.txt, horizons.txt')
