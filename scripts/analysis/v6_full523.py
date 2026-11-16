"""Re-run the chosen v6 candidates under 5x5 on all 523 users (v5 protocol, fold-rank
metric) for comparison with v1-v5. Not used for selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_research as R
R.DEV_ONLY = False
root, out, day = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
_orig = R.load_users
R.load_users = lambda o, r=None, dev_only=False: _orig(o, r, dev_only=False)
if not (out / 'folds' / f'text_scores_day{day}.json').exists():
    R.textfit(out, day, root)
T = R.Trials(out, day, root)
T3 = ('word', 'char', 'nb_word')
T.run(f'F{day}.01 v1_static', ('v1__',))
T.run(f'F{day}.02 text3 rankmean', text_models=T3, combiner='rankmean')
T.run(f'F{day}.03 text3 + v1 rankmean', text_models=T3, combiner='rankmean_models', block_groups=[('v1__',)])
T.run(f'F{day}.04 text word alone', text_models=('word',), combiner='score')
