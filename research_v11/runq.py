"""Run v11 jobs a few at a time, at below-normal priority (keeps the laptop usable).

  python research_v11/runq.py <max_parallel> <job> [<job> ...]
  job = "<candidate>@<rep>[@BM]"     e.g.  "MPN32@0"  "B@3@BM"  "ens:B+EMO|TXT32@4"
Logs go to private/v11/logs/<candidate>_<policy>[_BM]_r<rep>.txt. Skips jobs whose result file exists.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'private', 'v11'); LOGS = os.path.join(OUT, 'logs'); os.makedirs(LOGS, exist_ok=True)
BELOW_NORMAL = 0x00004000
POLICY = 'prev_adj'


def tag(c):
    return c.replace(':', '_').replace('|', '-').replace('+', '_')


def main():
    n = int(sys.argv[1]); jobs = sys.argv[2:]; running = []
    env0 = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    while jobs or running:
        running = [p for p in running if p.poll() is None]
        while jobs and len(running) < n:
            parts = jobs.pop(0).split('@'); c, r = parts[0], parts[1]; bm = len(parts) > 2 and parts[2] == 'BM'
            name = f'{tag(c)}_{POLICY}{"_BM" if bm else ""}_r{r}'
            if os.path.exists(os.path.join(OUT, f'res_{name}.json')):
                continue
            env = dict(env0, BEFOREDISC='matched') if bm else env0
            log = open(os.path.join(LOGS, f'{name}.txt'), 'w')
            running.append(subprocess.Popen([sys.executable, os.path.join(HERE, 'v11.py'), c, r, POLICY], stdout=log,
                                            stderr=subprocess.STDOUT, env=env, creationflags=BELOW_NORMAL))
            print('started', name, flush=True)
        time.sleep(3)
    print('queue done', flush=True)


if __name__ == '__main__':
    main()
