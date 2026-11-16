"""v6 round 2 trials (fold-rank evaluation, ranked text scores), day 28, 418 development users. Every result is logged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_research as R

root = Path(sys.argv[1])
out = Path(sys.argv[2])
part = sys.argv[3] if len(sys.argv) > 3 else 'all'
T = R.Trials(out, 28, root)

V1 = ('v1__',)
if part in ('blocks', 'all'):
    T.run('R2.01 v1_static (reference)', V1)
    T.run('R2.02 llm_tail', ('tail__',))
    T.run('R2.03 psycholinguistic lists', ('psy__',))
    T.run('R2.04 vader', ('vader__',))
    T.run('R2.05 empath 40', ('empath__',))
    T.run('R2.06 empath all 194', ('empathall__',))
    T.run('R2.07 spacy vectors', ('vec__',))
    T.run('R2.08 external r/depression model', ('ext__',))
    T.run('R2.09 ARTEFACT url+nonascii (never a candidate)', ('art__',))
    T.run('R2.10 psy+vader+empath40+ext', ('psy__', 'vader__', 'empath__', 'ext__'))
    T.run('R2.11 v1+psy+vader+empath40+ext', V1 + ('psy__', 'vader__', 'empath__', 'ext__'))
if part in ('text', 'all'):
    T.run('R2.20 text word', (), ('word',), 'score')
    T.run('R2.21 text char', (), ('char',), 'score')
    T.run('R2.22 text nb_word', (), ('nb_word',), 'score')
    T.run('R2.23 text word+char+nb rankmean', (), ('word', 'char', 'nb_word'), 'rankmean')
    T.run('R2.24 text word+char+nb logistic', (), ('word', 'char', 'nb_word'))
    T.run('R2.25 ext + text3', ('ext__',), ('word', 'char', 'nb_word'))
    T.run('R2.26 psy+vader+ext + text3', ('psy__', 'vader__', 'ext__'), ('word', 'char', 'nb_word'))
    T.run('R2.27 v1+psy+vader+empath40+ext + text3', V1 + ('psy__', 'vader__', 'empath__', 'ext__'),
          ('word', 'char', 'nb_word'))
    T.run('R2.28 everything (no artefact) + text3',
          V1 + ('tail__', 'psy__', 'vader__', 'empath__', 'ext__', 'vec__'), ('word', 'char', 'nb_word'))
    T.run('R2.29 vectors + text3', ('vec__',), ('word', 'char', 'nb_word'))
