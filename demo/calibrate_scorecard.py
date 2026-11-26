"""Calibrate the scorecard on the 418 development users only. The held-out users are never read here.

Steps
  1. reference distribution: item values on dev chunk-end rows (label-free) -> 1001 quantiles per item
  2. direction check: user-level AUROC of each item's mean (items below 0.5 are dropped, not flipped)
  3. correlation: user-level Spearman; items with |rho| >= 0.7 are merged into one group
  4. rules: fixed percentile constants (logged below); RAG at the LR decision row of each dev user,
     using the dev out-of-fold decisions (B, decide at every writing, standard policy, repeat 0)
  5. agreement with the LR flag, Cohen's kappa, positive rate per colour, composite AUROC
Writes demo/artifacts/scorecard.json and demo/artifacts/scorecard_report.json (aggregates only).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import ART, PRIV, load_split
import os
from scorecard import ITEMS, Scorecard, item_values
WIN = os.environ.get('SC_WINDOW', 'pre')

RHO_MERGE = 0.7
RULES = {'red': {'single': 97.5, 'pair': 90.0}, 'amber': {'single': 90.0, 'pair': 75.0}}
COMP_PCT = {'red': 90.0, 'amber': 75.0}          # composite cuts = these percentiles of dev composite
LOG = ['v1 constants (red 97.5/90, amber 90/75, composite P90/P75) fixed before any held-out case was built',
       'window: cur (last 14 days) and pre (history so far) both computed on dev; pre chosen '
       '(dev composite AUROC 0.91 vs 0.58; red+amber kappa 0.51 vs 0.40). Held-out not read.',
       'no constant changed after seeing dev rates (pre red rate 16.5% vs LR flag rate 24.6%)']


def main():
    users = load_split(); dev = users[users.split == 'train']
    A = pd.read_parquet(PRIV / 'chunksA_d14_PRIVATE.parquet'); A = A[A.user_id.isin(set(dev.user_id))]
    AW = pd.read_parquet(PRIV / 'chunksAW_d14_PRIVATE.parquet'); AW = AW[AW.user_id.isin(set(dev.user_id))]
    iv = item_values(A, WIN)
    quant = {k: np.quantile(iv[k].dropna(), np.linspace(0, 1, 1001)).tolist() for k in ITEMS}

    um = iv.assign(user_id=A.user_id.to_numpy()).groupby('user_id').mean()
    y = dev.set_index('user_id').label.reindex(um.index)
    auroc = {k: float(roc_auc_score(y, um[k].fillna(um[k].median()))) for k in ITEMS}
    keep = [k for k in ITEMS if auroc[k] >= 0.5]
    dropped = [k for k in ITEMS if k not in keep]
    rho = um[keep].corr(method='spearman')

    groups, used = {}, set()                       # greedy merge in item order
    for k in keep:
        if k in used:
            continue
        mem = [k] + [j for j in keep if j not in used and j != k and abs(rho.loc[k, j]) >= RHO_MERGE]
        used.update(mem); groups['+'.join(mem)] = mem
    labels = {g: (ITEMS[mem[0]][0] if len(mem) == 1 else 'Qwen distress signals' if 'self_blame' in mem else ' + '.join(ITEMS[m][0] for m in mem)) for g, mem in groups.items()}

    cfg = {'quantiles': quant, 'groups': groups, 'labels': labels,
           'rules': {**RULES, 'composite_cut': {}}, 'window': WIN,
           'reference': 'dev chunk-end rows, d14, label-free'}
    sc = Scorecard(cfg)
    # composite cut points from the dev chunk-end rows
    comp = np.array([np.mean(list(sc.points(dict(zip(iv.columns, r))).values())) for r in iv.to_numpy()])
    cfg['rules']['composite_cut'] = {c: float(np.percentile(comp, p)) for c, p in COMP_PCT.items()}
    sc = Scorecard(cfg)

    # RAG at each dev user's LR decision row (OOF flags, standard policy, repeat 0)
    dec = pd.read_parquet(PRIV / 'ref' / 'dec_W_B_r0_PRIVATE.parquet').query("objective == 'f_latency'").set_index('user_id')
    oof = pd.read_parquet(PRIV / 'ref' / 'oof_W_B_r0_PRIVATE.parquet')
    rows = []
    for u, x in AW.groupby('user_id'):
        x = x.sort_values('k')
        f = bool(dec.flagged.get(u, 0))
        r = x[x.k == dec.delay[u]].iloc[0] if f else x.iloc[-1]
        card = sc.card(r); last = sc.card(x.iloc[-1])
        rows.append({'user_id': u, 'y': int(dec.y[u]), 'lr_flag': int(f), 'colour': card['colour'],
                     'comp_last': last['composite']})
    R = pd.DataFrame(rows)
    pilot = dev.set_index('user_id').pilot
    R['np'] = ~R.user_id.map(pilot).astype(bool)
    oof_last = oof.sort_values(['user_id', 'k']).groupby('user_id').p.last().reindex(R.user_id).to_numpy()

    def summary(R, oof_last):
        ct = pd.crosstab(R.lr_flag, R.colour).reindex(columns=['green', 'amber', 'red'], fill_value=0)
        return {'n': int(len(R)), 'positives': int(R.y.sum()),
                'colour_rate': R.colour.value_counts(normalize=True).round(3).to_dict(),
                'positive_rate_by_colour': R.groupby('colour').y.mean().round(3).to_dict(),
                'n_by_colour': R.colour.value_counts().to_dict(),
                'crosstab_lrflag_x_colour': {int(k): v for k, v in ct.to_dict('index').items()},
                'kappa_red_vs_flag': float(cohen_kappa_score(R.lr_flag, (R.colour == 'red').astype(int))),
                'kappa_redamber_vs_flag': float(cohen_kappa_score(R.lr_flag, (R.colour != 'green').astype(int))),
                'auroc_composite_full_history': float(roc_auc_score(R.y, R.comp_last)),
                'auroc_lr_full_history_oof': float(roc_auc_score(R.y, oof_last))}

    rep = {'window': WIN, 'item_auroc_dev_user_mean': auroc, 'dropped_items': dropped,
           'spearman_user_level': rho.round(3).to_dict(), 'merged_groups': groups,
           'rules': cfg['rules'], 'log': LOG,
           'dev_all': summary(R, oof_last), 'dev_np': summary(R[R.np.to_numpy()], oof_last[R.np.to_numpy()])}
    if os.environ.get('SC_DRY') == '1':
        print('dry run', WIN, json.dumps(rep['dev_all'], default=float)); print(json.dumps(rep['dev_np'], default=float)); return
    (ART / 'scorecard.json').write_text(json.dumps(cfg))
    (ART / 'scorecard_report.json').write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ('item_auroc_dev_user_mean', 'dropped_items', 'merged_groups', 'rules')}, indent=1))
    hi = [(a, b, round(rho.loc[a, b], 2)) for i, a in enumerate(keep) for b in keep[i + 1:] if abs(rho.loc[a, b]) >= 0.5]
    print('pairs |rho|>=0.5:', hi)
    for k in ('dev_all', 'dev_np'):
        print(k, json.dumps(rep[k], indent=1, default=float))


if __name__ == '__main__':
    main()
