"""Aggregate the 5 outer repeats: mean and SD across repeats. Also flag timing (rep 0)."""
import json, glob, sys, numpy as np, pandas as pd
import seq_eval as SE

def get(r, path):
    for p in path: r = r[p]
    return r

ROWS = [('seq F1 (latency-F1 policy) all', ('selected', 'early_all', 'f_latency', 'f1')),
        ('seq F1 (latency-F1 policy) np', ('selected', 'early_np', 'f_latency', 'f1')),
        ('precision np', ('selected', 'early_np', 'f_latency', 'precision')),
        ('recall np', ('selected', 'early_np', 'f_latency', 'recall')),
        ('precision all', ('selected', 'early_all', 'f_latency', 'precision')),
        ('recall all', ('selected', 'early_all', 'f_latency', 'recall')),
        ('seq F1 (F1 policy) np', ('selected', 'early_np', 'f1', 'f1')),
        ('latency F np', ('selected', 'early_np', 'f_latency', 'f_latency')),
        ('median TP delay (writings)', ('selected', 'early_all', 'f_latency', 'median_tp_delay')),
        ('ERDE5 (ERDE5 policy) all', ('selected', 'early_all', 'erde5', 'erde5')),
        ('ERDE50 (ERDE50 policy) all', ('selected', 'early_all', 'erde50', 'erde50')),
        ('ERDE5 (ERDE5 policy) np', ('selected', 'early_np', 'erde5', 'erde5')),
        ('ERDE50 (ERDE50 policy) np', ('selected', 'early_np', 'erde50', 'erde50')),
        ('AUROC end of history all', ('selected', 'auroc', 'full', 'auroc_all')),
        ('AUROC end of history np', ('selected', 'auroc', 'full', 'auroc_np')),
        ('AUROC after 10 writings np', ('selected', 'auroc', 'k10', 'auroc_np')),
        ('AUROC after 25 writings np', ('selected', 'auroc', 'k25', 'auroc_np')),
        ('AUROC after 50 writings np', ('selected', 'auroc', 'k50', 'auroc_np')),
        ('AUROC after 100 writings np', ('selected', 'auroc', 'k100', 'auroc_np')),
        ('AUROC day 14 np (d7 spec)', ('per_spec', 'd7', 'auroc', 'd14', 'auroc_np')),
        ('AUROC day 28 np (d7 spec)', ('per_spec', 'd7', 'auroc', 'd28', 'auroc_np')),
        ('AUROC day 84 np (d7 spec)', ('per_spec', 'd7', 'auroc', 'd84', 'auroc_np')),
        ('AUROC day 182 np (d7 spec)', ('per_spec', 'd7', 'auroc', 'd182', 'auroc_np')),
        ('AUROC day 28 all (d7 spec)', ('per_spec', 'd7', 'auroc', 'd28', 'auroc_all')),
        ('AUROC active>365d np (d7 spec)', ('per_spec', 'd7', 'auroc', 'd365', 'auroc_active_np'))]

def table(variants, prefix=''):
    out = {}
    for v in variants:
        fs = sorted(glob.glob(f'res_{prefix}{v}_r[0-9].json'))
        rs = [json.load(open(f)) for f in fs]
        if not rs: continue
        d = {'n_repeats': len(rs)}
        for name, path in ROWS:
            x = np.array([get(r, path) if get(r, path) is not None else np.nan for r in rs], float)
            d[name] = (float(np.nanmean(x)), float(np.nanstd(x)))
        ch = [s['chosen_spec'] for r in rs for s in r['selection'] if 'chosen_spec' in s]
        cc = [s['chosen_C'] for r in rs for s in r['selection'] if 'chosen_C' in s]
        d['chosen_spec_counts'] = pd.Series(ch).value_counts().to_dict()
        d['chosen_C_counts'] = {str(k): int(v) for k, v in pd.Series(cc).value_counts().items()}
        # per-spec seq F1 np (sensitivity over chunk size)
        d['per_spec_seqF1_np'] = {s: float(np.mean([r['per_spec'][s]['early_np']['f_latency']['f1'] for r in rs])) for s in SE.SPECS}
        d['per_spec_seqF1_all'] = {s: float(np.mean([r['per_spec'][s]['early_all']['f_latency']['f1'] for r in rs])) for s in SE.SPECS}
        d['per_spec_auroc_end_np'] = {s: float(np.mean([r['per_spec'][s]['auroc']['full']['auroc_np'] for r in rs])) for s in SE.SPECS}
        out[v] = d
    return out

def flag_timing(tag):
    oof = pd.read_parquet(f'oof_{tag}_PRIVATE.parquet'); dec = pd.read_parquet(f'dec_{tag}_PRIVATE.parquet')
    dec = dec[(dec.objective == 'f_latency') & (dec.flagged == 1)]
    day = oof.set_index(['user_id', 'k']).day
    dec = dec.assign(day=[day.get((u, k), np.nan) for u, k in zip(dec.user_id, dec.delay)])
    users = SE.load_users(); npset = set(users.loc[~users.pilot, 'user_id'])
    res = {}
    for nm, m in (('TP', dec.y == 1), ('FP', dec.y == 0)):
        x = dec[m]
        res[nm] = {'n': int(len(x)), 'median_day': float(x.day.median()), 'median_writings': float(x.delay.median()),
                   'share_after_day84': float((x.day > 84).mean()), 'n_after_day84': int((x.day > 84).sum()),
                   'n_np': int(x.user_id.isin(npset).sum()), 'n_np_after_day84': int((x.user_id.isin(npset) & (x.day > 84)).sum())}
    return res

if __name__ == '__main__':
    vs = sys.argv[1:] or ['C', 'O', 'B', 'set3']
    T = table(vs)
    T['flag_timing_rep0'] = {v: flag_timing(f'{v}_r0') for v in vs}
    json.dump(T, open('agg.json', 'w'), indent=1)
    names = [n for n, _ in ROWS]
    print(f"{'metric':<34}" + ''.join(f'{v:>16}' for v in vs))
    for n in names:
        print(f'{n:<34}' + ''.join(f"{T[v][n][0]:>9.3f} ±{T[v][n][1]:.3f}" for v in vs))
    for v in vs:
        print(v, 'specs', T[v]['chosen_spec_counts'], 'C', T[v]['chosen_C_counts'])
        print('   per-spec seq F1 np', {k: round(x, 3) for k, x in T[v]['per_spec_seqF1_np'].items()})
        print('   flag timing', T['flag_timing_rep0'][v])
