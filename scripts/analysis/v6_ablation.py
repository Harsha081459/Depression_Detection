"""Post-freeze ablation of the final text ensemble on the dev set (day 28). Logged as A.* trials."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_research as R
T = R.Trials(Path(sys.argv[2]), 28, Path(sys.argv[1]))
T.run('A.01 ablation: word + nb (no char)', text_models=('word', 'nb_word'), combiner='rankmean', note='post-freeze ablation')
T.run('A.02 ablation: char + nb (no word)', text_models=('char', 'nb_word'), combiner='rankmean', note='post-freeze ablation')
T.run('A.03 ablation: nb alone', text_models=('nb_word',), combiner='score', note='post-freeze ablation')
