"""v6 round 3: rank-mean ensembles of separately fitted model families. Day 28, 418 dev users."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_research as R
T = R.Trials(Path(sys.argv[2]), 28, Path(sys.argv[1]))
T3 = ('word', 'char', 'nb_word')
T.run('R3.01 rank: text3 + vectors', text_models=T3, combiner='rankmean_models', block_groups=[('vec__',)])
T.run('R3.02 rank: text3 + vectors + psy', text_models=T3, combiner='rankmean_models', block_groups=[('vec__',), ('psy__',)])
T.run('R3.03 rank: text3 + vectors + v1', text_models=T3, combiner='rankmean_models', block_groups=[('vec__',), ('v1__',)])
T.run('R3.04 rank: text3 + vectors + psy + v1', text_models=T3, combiner='rankmean_models', block_groups=[('vec__',), ('psy__',), ('v1__',)])
T.run('R3.05 rank: word + char', text_models=('word', 'char'), combiner='rankmean')
T.run('R3.06 rank: text3 + ext', text_models=T3, combiner='rankmean_models', block_groups=[('ext__',)])
T.run('R3.07 rank: word + char + vectors', text_models=('word', 'char'), combiner='rankmean_models', block_groups=[('vec__',)])
T.run('R3.08 rank: text3 + v1', text_models=T3, combiner='rankmean_models', block_groups=[('v1__',)])
T.run('R3.09 rank: text3 + psy', text_models=T3, combiner='rankmean_models', block_groups=[('psy__',)])
T.run('R3.10 rank: text3 + llm_tail', text_models=T3, combiner='rankmean_models', block_groups=[('tail__',)])
