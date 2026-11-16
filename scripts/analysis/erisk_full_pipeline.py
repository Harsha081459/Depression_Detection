"""Private, local-only eRisk pipeline. Python >=3.10.

Inputs: posts.parquet(user_id, post_id, date, text),
labels.parquet(user_id, label), pilot_users.parquet(user_id).
Do not publish private inputs, caches, splits, or manual-review outputs.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SIGNALS = ('self_blame', 'helplessness', 'isolation', 'family_strain', 'emptiness')
CUTS = (.25, .5, .75)  # Prespecified candidates; select using training users only.
CODE = ['polarity_jump', 'polarity_flip', 'distress_burst', 'absolutist']
CONTROLS = ['n_posts', 'has_posts']


def pilot_gate_passed(signals):
    return set(signals) == set(SIGNALS) and all(
        v.get('n') == 60 and isinstance(v.get('spearman'), (float, int))
        and np.isfinite(v['spearman']) and v['spearman'] >= .95
        for v in signals.values())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if isinstance(value, pd.DataFrame):
        value.to_parquet(tmp, index=False)
    else:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_data(posts_path, labels_path):
    posts, labels = pd.read_parquet(posts_path), pd.read_parquet(labels_path)
    assert {'user_id', 'post_id', 'date', 'text'} <= set(posts)
    assert {'user_id', 'label'} <= set(labels)
    for frame in (posts, labels):
        assert frame.user_id.notna().all()
        frame['user_id'] = frame.user_id.astype(str)
    assert posts.post_id.notna().all()
    posts['post_id'] = posts.post_id.astype(str)
    assert not posts.duplicated(['user_id', 'post_id']).any()
    assert not labels.user_id.duplicated().any()
    assert set(labels.label) == {0, 1}
    assert set(posts.user_id) == set(labels.user_id)
    assert len(labels) == 523
    posts['date'] = pd.to_datetime(posts.date, utc=True, errors='raise')
    assert posts.date.notna().all()
    posts['text'] = posts.text.fillna('').astype(str)
    return posts.sort_values(['user_id', 'date', 'post_id']), labels


def export_inputs(text_csv, labels_csv, pilot_manifest, directory):
    """Lab-only adapter for the existing CSVs; never prints text or identifiers."""
    assert os.name == 'posix', 'Private export must stay on the lab'
    directory=Path(directory)
    assert not directory.exists(), 'Preserve existing input snapshots'
    data=pd.read_csv(text_csv, dtype={'user_id':str})
    labels=pd.read_csv(labels_csv, usecols=['user_id','label'],dtype={'user_id':str})
    data['text']=(data.title.fillna('')+' '+data.body.fillna('')).str.strip()
    data['date']=pd.to_datetime(data.date,utc=True,errors='raise')
    if 'post_id' not in data:
        data['post_id']=[digest([u,str(d),t,i]) for i,(u,d,t) in
                         enumerate(zip(data.user_id,data.date,data.text))]
    manifest=json.loads(Path(pilot_manifest).read_text())
    pilot=pd.DataFrame({'user_id':sorted({str(t['user_id']) for t in manifest})})
    atomic(directory/'posts.parquet',data[['user_id','post_id','date','text']])
    atomic(directory/'labels.parquet',labels)
    atomic(directory/'pilot_users.parquet',pilot)


# STEP 2 FIRST: persist a split before scoring/selection/inspection.
def freeze_split(labels, pilot_users_path, output):
    """Pilot-exposed users stay in training; test is sampled from remaining users.

    A previously inspected cohort is not a pristine prospective test set.
    This function cannot undo earlier full-cohort analyses.
    """
    labels = labels.sort_values('user_id').reset_index(drop=True)
    pilot = set(pd.read_parquet(pilot_users_path).user_id.astype(str))
    assert pilot <= set(labels.user_id)
    eligible = labels[~labels.user_id.isin(pilot)]
    _, test = train_test_split(eligible.user_id, test_size=int(np.ceil(.2 * len(labels))),
                               stratify=eligible.label, random_state=SEED)
    result = labels.assign(split=np.where(labels.user_id.isin(test), 'test', 'train'),
                           pilot_exposed=labels.user_id.isin(pilot))
    payload = {'seed': SEED, 'labels_hash': digest(labels.to_dict('records')),
               'pilot_users_hash': digest(sorted(pilot)), 'rows': result.to_dict('records')}
    output = Path(output)
    if output.exists():
        assert json.loads(output.read_text()) == payload, 'Existing split/input changed'
    else:
        atomic(output, payload)
    assert not pilot.intersection(test)
    return result.set_index('user_id')


def prepare_posts(posts):
    from final_indirect_pipeline import disclosure, flags
    # LEAKAGE: origin must be computed BEFORE dropping disclosures.
    origins = posts.groupby('user_id').date.min()
    keep = ~posts.text.map(disclosure) & posts.text.str.strip().ne('')
    keep &= ~posts.text.str.strip().str.lower().isin(['[removed]', '[deleted]'])
    filtered = posts.loc[keep].copy()
    parsed = filtered.text.map(flags)
    for signal in SIGNALS:
        key = 'helpless' if signal == 'helplessness' else signal
        filtered['word_' + signal] = parsed.map(lambda x: x[key])
    for key in ['polarity', 'distress', 'absolutist']:
        filtered[key] = parsed.map(lambda x: x[key])
    return filtered, origins


# STEP 1: full BF16 HF weights, NOT an Ollama GGUF path.
class LocalScorer:
    def __init__(self, model_dir, prompts_path, batch_size=1):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import random
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.config = json.loads(Path(prompts_path).read_text())
        assert self.config['prompt_version'] == 1, 'Use frozen LAB v1 prompts; do not edit v0 locally'
        assert set(self.config['signals']) == set(SIGNALS)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        ids = [self.tokenizer.encode(x, add_special_tokens=False) for x in ('Yes', 'No')]
        assert all(len(x) == 1 for x in ids) and ids[0] != ids[1]
        self.ids = [x[0] for x in ids]
        assert [self.tokenizer.decode([i]) for i in self.ids] == ['Yes', 'No']
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        assert self.model.config.model_type == 'qwen2'
        assert self.model.lm_head.bias is None
        self.batch_size = batch_size
        weight_files = sorted(Path(model_dir).glob('*.safetensors'))
        assert weight_files, 'Expected local HF safetensors weights'
        self.meta = {'model': 'Qwen/Qwen2.5-7B-Instruct', 'seed': SEED,
                     'prompt_hash': digest(self.config), 'prompt_version': 1,
                     'weights': {p.name: sha_file(p) for p in weight_files},
                     'tokenizer_hash': digest(self.tokenizer.backend_tokenizer.to_str()),
                     'chat_template_hash': digest(self.tokenizer.chat_template),
                     'model_config_hash': digest(self.model.config.to_dict()),
                     'yes_no_ids': self.ids, 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'batch_size': batch_size,
                     'score_definition': 'P(Yes | next token in {Yes,No})',
                     'pilot_backend_equivalence': 'requires_validation'}

    def predict(self, texts, signal):
        from llm_signal_scorer import payload_for
        # Reuse the frozen prompt builder exactly, including the 2000-char limit.
        messages = [payload_for(t, signal, self.config)['messages'] for t in texts]
        rendered = [self.tokenizer.apply_chat_template(m, tokenize=False,
                    add_generation_prompt=True) for m in messages]
        inputs = self.tokenizer(rendered, padding=True, truncation=False,
                                add_special_tokens=False, return_tensors='pt')
        assert inputs.input_ids.shape[1] <= self.config['options']['num_ctx'], 'Context limit exceeded'
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        position = inputs['attention_mask'].long().cumsum(-1) - 1
        position.masked_fill_(inputs['attention_mask'] == 0, 0)
        with self.torch.no_grad():
            # Last-token projection only: avoids [batch,sequence,vocabulary] logits.
            hidden = self.model.model(**inputs, position_ids=position,
                                      use_cache=False).last_hidden_state[:, -1, :]
            logits = self.torch.nn.functional.linear(hidden, self.model.lm_head.weight[self.ids])
            answer = logits.float().softmax(-1)[:, 0].cpu().numpy()
        return answer

    def check_backend(self, old_phase1, output):
        from llm_signal_scorer import PROBES
        reference = json.loads(Path(old_phase1).read_text())
        assert reference.get('phase1_passed'), 'Reference preflight did not pass'
        assert reference['prompt_hash'] == self.meta['prompt_hash']
        old = {p['probe']: p for p in reference['probes']}
        results = []
        for name, text, signal, yes in PROBES:
            a = float(self.predict([text], signal)[0])
            b = float(self.predict([text] * self.batch_size, signal)[0])
            c = float(self.predict([text], signal)[0])
            results.append({'probe': name, 'signal': signal, 'score': a,
                            'probe_pass': (a > .5) == yes,
                            'batch_delta': abs(a-b), 'repeat_delta': abs(a-c),
                            'reference_delta': abs(a-old[name]['score'])})
        # Cross-backend absolute probabilities can differ under quantization.
        # User-prescribed pilot Spearman is the cross-backend gate; retain
        # absolute probe deltas as diagnostics, not an extra equivalence rule.
        passed = all(r['probe_pass'] and max(r['batch_delta'], r['repeat_delta']) <= .001 for r in results)
        report = {'backend_hash': digest(self.meta), 'passed': passed, 'probes': results,
                  'scope': 'Synthetic answer/repeatability check; cross-backend ranking requires pilot Spearman',
                  'reference_delta_is_gate': False}
        atomic(output, report)
        return report

    def score(self, posts, directory, backend_check):
        if not (backend_check.get('passed') is True
                and backend_check.get('backend_hash') == digest(self.meta)
                and pilot_gate_passed(backend_check.get('signals', {}))
                and backend_check.get('speed_gate_passed') is True
                and backend_check.get('review_gate_passed') is True
                and backend_check.get('validated_batch_size') == self.batch_size):
            raise RuntimeError('Cohort blocked: all three pilot gates must pass for this exact backend/batch size')
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        data = posts[['user_id','post_id','date','text']].sort_values(['user_id','post_id'])
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()
        meta = dict(self.meta, data_hash=data_hash)
        path = directory / 'metadata.json'
        # Single writer. Atomic shards survive crashes; completed shards are reused.
        from filelock import FileLock
        with FileLock(str(directory / '.writer.lock')):
            if path.exists():
                assert json.loads(path.read_text()) == meta, 'Cache/input/model mismatch'
            else:
                atomic(path, meta)
            for i in range(0, len(data), 32):
                part = directory / ('part-%08d.parquet' % i)
                if part.exists():
                    continue
                batch = data.iloc[i:i+32]
                out = batch.drop(columns='text').copy()
                for signal in SIGNALS:
                    values=[]
                    for j in range(0,len(batch),self.batch_size):
                        values.extend(self.predict(batch.text.iloc[j:j+self.batch_size].tolist(), signal))
                    out[signal] = values
                atomic(part, out)
        return pd.concat([pd.read_parquet(p) for p in sorted(directory.glob('part-*.parquet'))], ignore_index=True)


# STEP 3: all candidates are fixed, label-free transforms.
def window_features(posts, scores, origins, users, weeks):
    assert not scores.duplicated(['user_id','post_id']).any()
    data = posts.merge(scores.drop(columns='date'), on=['user_id','post_id'],
                       how='left', validate='one_to_one')
    assert data[list(SIGNALS)].notna().all().all(), 'Missing scores cannot become empty windows'
    assert np.isfinite(data[list(SIGNALS)].to_numpy()).all()
    assert data[list(SIGNALS)].ge(0).all().all() and data[list(SIGNALS)].le(1).all().all()
    data = data[data.date <= data.user_id.map(origins) + pd.Timedelta(weeks=weeks)]
    rows = []
    for uid, group in data.groupby('user_id', sort=True):
        group = group.sort_values(['date','post_id'])
        row = {'user_id': uid, 'n_posts': len(group), 'has_posts': 1}
        for signal in SIGNALS:
            for prefix in ('', 'word_'):
                values = group[prefix+signal].to_numpy(float)
                row[prefix+signal+'__mean'] = values.mean()
                row[prefix+signal+'__max'] = values.max()
                row[prefix+signal+'__top3'] = np.sort(values)[-3:].mean()
                for cut in CUTS:
                    row[prefix+signal+'__fraction_'+str(cut)] = (values > cut).mean()
        polarity = group.polarity.to_numpy()
        row['polarity_flip'] = np.mean(polarity[1:]*polarity[:-1] < 0) if len(group)>1 else 0.
        row['polarity_jump'] = np.abs(np.diff(polarity)).mean()/2 if len(group)>1 else 0.
        row['distress_burst'] = max(0., np.sort(group.distress)[-3:].mean()-group.distress.median())
        row['absolutist'] = group.absolutist.mean()
        rows.append(row)
    columns = CONTROLS + CODE + [p+s+'__'+a for p in ('','word_') for s in SIGNALS
               for a in ['mean','max','top3']+['fraction_'+str(c) for c in CUTS]]
    frame = pd.DataFrame(rows).set_index('user_id') if rows else pd.DataFrame(columns=columns)
    return frame.reindex(index=list(users), columns=columns).fillna(0).astype(float)


def folds(y):
    assert min(np.bincount(np.asarray(y, dtype=int))) >= 5, 'Too few users per class for 5-fold CV'
    return StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)


def choose_feature(X, y, signal, prefix=''):
    candidates = sorted(c for c in X if c.startswith(prefix+signal+'__'))
    cv = list(folds(y)); y = np.asarray(y)
    means = {c: float(np.mean([roc_auc_score(y[v], X[c].iloc[v]) for _, v in cv])) for c in candidates}
    # Fixed positive direction; never flip using test labels.
    return max(candidates, key=lambda c: means[c]), means


class SelectSignals(BaseEstimator, TransformerMixin):
    """Fits inside EACH model CV training fold, avoiding selection leakage."""
    def __init__(self, mode='both'):
        self.mode = mode

    def fit(self, X, y):
        self.columns_ = ['n_posts'] if self.mode == 'count' else CONTROLS.copy()
        if self.mode in ('llm','both'):
            self.columns_ += [choose_feature(X, y, s)[0] for s in SIGNALS]
        if self.mode in ('code','both'):
            self.columns_ += CODE
        return self

    def transform(self, X):
        return X[self.columns_].to_numpy(float)


# STEPS 4/5: paired, stratified user bootstrap; no model retuning in bootstrap.
def bootstrap(y, prediction, metric=roc_auc_score, other=None, n=2000):
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert set(y) == {0,1}
    other = None if other is None else np.asarray(other)
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(y==k) for k in (0,1)]
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in strata])
        val = metric(y[idx], prediction[idx])
        if other is not None:
            val -= metric(y[idx], other[idx])
        values.append(val)
    point = metric(y, prediction) - (metric(y, other) if other is not None else 0.)
    lo, hi = np.quantile(values, [.025,.975])
    return {'estimate': float(point), 'ci_low': float(lo), 'ci_high': float(hi)}


def signal_results(Xtrain, ytrain, Xtest, ytest, weeks):
    results, cv_rows = [], []
    for signal in SIGNALS:
        chosen, candidates = choose_feature(Xtrain, ytrain, signal)
        for name, score in candidates.items():
            cv_rows.append({'signal': signal, 'weeks': weeks, 'aggregation': name,
                            'train_cv_auroc': score, 'selected': name==chosen})
        result = dict(signal=signal, weeks=weeks, aggregation=chosen,
                      **bootstrap(ytest, Xtest[chosen]))
        results.append(result)
    return results, cv_rows


def emptiness_check(Xtrain, ytrain, Xtest, ytest, weeks):
    llm, _ = choose_feature(Xtrain, ytrain, 'emptiness')
    word, _ = choose_feature(Xtrain, ytrain, 'emptiness', 'word_')
    diff = bootstrap(ytest, Xtest[llm], other=Xtest[word])
    # Negative CI excludes zero too, but means LLM is worse, not "keep".
    decision = 'keep' if diff['ci_low']>0 else ('word_rule_better' if diff['ci_high']<0 else 'exploratory')
    return dict(weeks=weeks, llm_aggregation=llm, word_aggregation=word,
                llm_auroc=roc_auc_score(ytest,Xtest[llm]), word_auroc=roc_auc_score(ytest,Xtest[word]),
                decision=decision, **diff)


# STEP 6: locked hyperparameter grids; all fitted preprocessing stays within CV.
def classifier_results(Xtrain, ytrain, Xtest, ytest, weeks):
    results = []
    for mode in ('code','llm','both','count'):
        specifications = [
            ('logistic', LogisticRegression(penalty='l2',solver='liblinear',max_iter=2000,random_state=SEED),
             {'model__C':[.01,.1,1.,10.]}),
            ('boosting', GradientBoostingClassifier(random_state=SEED),
             {'model__n_estimators':[50,100], 'model__max_depth':[1,2], 'model__learning_rate':[.03,.1]})]
        for name, model, params in specifications:
            pipeline = Pipeline([('select',SelectSignals(mode)), ('scale',StandardScaler()), ('model',model)])
            search = GridSearchCV(pipeline,params,scoring='roc_auc',cv=list(folds(ytrain)),n_jobs=1,refit=True,error_score='raise')
            search.fit(Xtrain,ytrain)
            pred = search.predict_proba(Xtest)[:,1]
            for metric_name, metric in [('auroc',roc_auc_score),('auprc',average_precision_score)]:
                results.append(dict(weeks=weeks,ablation=mode,model=name,metric=metric_name,
                    parameters=json.dumps(search.best_params_,sort_keys=True),
                    features=json.dumps(search.best_estimator_.named_steps['select'].columns_),
                    **bootstrap(ytest,pred,metric)))
    return results


# STEP 7: inspect TRAIN users only. Private top-post text never reaches stdout.
def sanity(posts, scores, split, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    ids=split.index[split.split=='train']
    data=posts[posts.user_id.isin(ids)].merge(scores,on=['user_id','post_id','date'],validate='one_to_one')
    hist=[]
    for signal in SIGNALS:
        counts, edges=np.histogram(data[signal],bins=np.linspace(0,1,21))
        hist += [dict(signal=signal,left=edges[i],right=edges[i+1],count=int(n)) for i,n in enumerate(counts)]
        atomic(directory/(signal+'_top10_PRIVATE.parquet'),data.nlargest(10,signal)[['user_id','post_id','date','text',signal]])
    pd.DataFrame(hist).to_csv(directory/'histograms.csv',index=False)
    data[list(SIGNALS)].corr(method='spearman').to_csv(directory/'correlation.csv')
    counts=data.groupby('user_id').size().reindex(ids,fill_value=0)
    audit=pd.DataFrame({'n_posts':counts,'label':split.loc[ids,'label']})
    audit.groupby('label').n_posts.agg(['count','mean','median']).to_csv(directory/'post_count_train.csv')
    # Domain keywords alone are not ground truth for non-emotional content.
    neutral=data[data.text.str.contains(r'\b(game|boss|quest|sport|news)\b',case=False,regex=True)]
    pd.DataFrame([{'signal':s,'candidate_posts':len(neutral),
                   'fraction_above_0.5':(neutral[s]>.5).mean() if len(neutral) else np.nan}
                  for s in SIGNALS]).to_csv(directory/'domain_candidates_not_gold.csv',index=False)


def evaluate_all(posts, scores, origins, split, output):
    from importlib.metadata import version
    from erisk_repeated_cv import repeated_cv,truncation_audit
    output=Path(output)
    assert not output.exists(), 'Test evaluation output exists: do not tune and rerun'
    output.mkdir(parents=True)
    tr=split.index[split.split=='train']; te=split.index[split.split=='test']
    ytr=split.loc[tr,'label']; yte=split.loc[te,'label']
    summary=[]; candidates=[]; empty=[]; classifiers=[]; checks=[]; count_audit=[]; banks={}
    for weeks in (4,8,12):
        X=window_features(posts,scores,origins,split.index,weeks)
        banks[weeks]=X
        for name, ids in [('train',tr),('test',te)]:
            empty.append(dict(weeks=weeks,split=name,users=len(ids),zero_posts=int((X.loc[ids,'has_posts']==0).sum())))
            for label in (0,1):
                group=X.loc[ids].loc[split.loc[ids,'label']==label,'n_posts']
                count_audit.append(dict(weeks=weeks,split=name,label=label,users=len(group),mean=group.mean(),median=group.median()))
        a,b=signal_results(X.loc[tr],ytr,X.loc[te],yte,weeks)
        summary+=a; candidates+=b
        checks.append(emptiness_check(X.loc[tr],ytr,X.loc[te],yte,weeks))
        classifiers+=classifier_results(X.loc[tr],ytr,X.loc[te],yte,weeks)
    for name,rows in [('signal_summary',summary),('train_candidates',candidates),('empty_windows',empty),
                      ('emptiness_check',checks),('classifiers',classifiers),('post_count_audit',count_audit)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    truncation_audit(posts,origins,split.label).to_csv(output/'truncation_by_label.csv',index=False)
    all_cv=repeated_cv(banks,split.label,output/'repeated_cv')
    if 'pilot_exposed' not in split: raise RuntimeError('Pilot exposure flags required for sensitivity CV')
    nonpilot=split.index[~split.pilot_exposed.astype(bool)]
    nonpilot_cv=repeated_cv({w:x.loc[nonpilot] for w,x in banks.items()},split.loc[nonpilot,'label'],
                           output/'repeated_cv_without_pilot',population='without_pilot')
    comparison=all_cv.merge(nonpilot_cv,on=['weeks','model','ablation'],suffixes=('_all','_without_pilot'))
    comparison['auroc_gap_all_minus_without_pilot']=comparison.mean_auroc_all-comparison.mean_auroc_without_pilot
    comparison.to_csv(output/'cv_pilot_exposure_sensitivity.csv',index=False)
    atomic(output/'run.json',{'seed':SEED,'bootstrap_resamples':2000,'test_users':len(te),
        'train_users':len(tr),'warning':'Small AUROC gaps with 523 users may be noise. CIs are conditional on the fitted model and this split.',
        'packages':{k:version(k) for k in ('numpy','pandas','scikit-learn','pyarrow')},
        'pipeline_sha256':sha_file(__file__),
        'headline':'5x5 repeated stratified nested user CV across all 523 users; holdout is a secondary check',
        'multiple_comparisons':'Five signals x three windows x aggregation candidates plus model variants. No best-signal claim from small gaps.',
        'rule_source_sha256':sha_file(Path(__file__).with_name('final_indirect_pipeline.py')),
        'emptiness_decision':'Test-set confirmation only; never use it to change reported classifiers.',
        'test_integrity':'Historical full-cohort exploration prevents calling this a pristine untouched test.'})
    return pd.DataFrame(summary)
