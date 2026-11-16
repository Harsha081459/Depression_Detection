"""Descriptive: which n-grams the word model weights most, fit on the 418 dev users at
day 28 with the same cleaning. Not used for any choice. Writes aggregate terms only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_temporal_features as v2, v6_features as v6, v6_research as R
root, out = Path(sys.argv[1]), Path(sys.argv[2])
ids, y, _ = R.load_users(out, root)
w, _, users = v2.load_writings(root)
posts = pd.read_parquet(root / 'private' / 'cohort_v1' / 'eligible_posts.parquet', columns=['post_id', 'text'])
t = w[['user_id', 'post_id', 'delta_days']].merge(posts, on='post_id')
t = t[(t.delta_days <= 28) & t.user_id.astype(str).isin(ids)]
t['clean'] = t.text.map(v6.strict_clean)
docs = t.groupby('user_id').clean.apply(' '.join).reindex(ids).fillna('')
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=10)
X = vec.fit_transform(docs)
m = LogisticRegression(solver='liblinear', C=10).fit(X, y.to_numpy())
names = np.array(vec.get_feature_names_out()); c = m.coef_[0]; o = np.argsort(c)
res = {'toward_erisk_positive': names[o[::-1][:80]].tolist(), 'toward_control': names[o[:80]].tolist(),
       'min_df_users': 10, 'users': len(ids)}
(out / 'top_terms_day28_dev.json').write_text(json.dumps(res, indent=1))
print('POS:', ', '.join(res['toward_erisk_positive'])); print(); print('CTRL:', ', '.join(res['toward_control']))
