"""v6 round 4: the leading candidates at days 14 and 84 (dev). Logged."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_research as R
T3 = ('word', 'char', 'nb_word')
for day in (14, 84):
    T = R.Trials(Path(sys.argv[2]), day, Path(sys.argv[1]))
    T.run(f'R4.{day}.01 v1_static', ('v1__',))
    T.run(f'R4.{day}.02 text3 rankmean', text_models=T3, combiner='rankmean')
    T.run(f'R4.{day}.03 text3 + v1 rankmean', text_models=T3, combiner='rankmean_models', block_groups=[('v1__',)])
    T.run(f'R4.{day}.04 text3 + vectors + v1 rankmean', text_models=T3, combiner='rankmean_models', block_groups=[('vec__',), ('v1__',)])
