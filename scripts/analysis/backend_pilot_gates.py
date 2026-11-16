"""Ordered pilot-only gates. Never invokes cohort scoring or reads human labels."""
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from erisk_full_pipeline import SIGNALS, atomic, digest, sha_file, pilot_gate_passed

GATE2_MAX_GAP = .01
GATE2_MIN_SPEARMAN = .99


def comparison_stats(hf, reference):
    hf=np.asarray(hf,dtype=float); reference=np.asarray(reference,dtype=float)
    if len(hf)!=len(reference) or len(hf)==0 or not np.isfinite(hf).all() or not np.isfinite(reference).all():
        raise RuntimeError('Invalid paired backend scores')
    def ties(values):
        _,counts=np.unique(values,return_counts=True)
        return float(counts[counts>1].sum()/len(values))
    rho=float(spearmanr(hf,reference).statistic) if np.ptp(hf) and np.ptp(reference) else None
    return {'n':len(hf),'spearman':rho,'max_abs_gap':float(np.max(np.abs(hf-reference))),
            'hf_exact_tie_rate':ties(hf),'ollama_exact_tie_rate':ties(reference),
            'hf_unique_scores':int(len(np.unique(hf))),'ollama_unique_scores':int(len(np.unique(reference))),
            'hf_near_zero_rate':float(np.mean(hf<=1e-6)),
            'ollama_near_zero_rate':float(np.mean(reference<=1e-6)),
            'tie_definition':'fraction of items belonging to an exact tied group; near-zero cutoff 1e-6'}


def load_reference(scorer, manifest, cache, phase1):
    phase=json.loads(Path(phase1).read_text())
    if not phase.get('phase1_passed') or phase['prompt_hash']!=scorer.meta['prompt_hash']:
        raise RuntimeError('Matching frozen version-1 reference required')
    tasks=[t for t in json.loads(Path(manifest).read_text()) if t['split']=='heldout']
    if len(tasks)!=300 or len({(t['post_id'],t['signal']) for t in tasks})!=300 or any(sum(t['signal']==s for t in tasks)!=60 for s in SIGNALS):
        raise RuntimeError('Require locked 300-question held-out pilot; never resample')
    with sqlite3.connect(Path(cache).resolve().as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT post_id,signal,score,text_hash FROM scores WHERE prompt_hash=? AND model_digest=? AND ollama_version=?',
            (phase['prompt_hash'],phase['model_digest'],phase['ollama_version'])).fetchall()
    reference={(p,s):(value,text_hash) for p,s,value,text_hash in rows}
    if len(reference)!=len(rows): raise RuntimeError('Duplicate reference scores')
    pairs=[]
    for task in tasks:
        old=reference.get((task['post_id'],task['signal']))
        if old is None or old[1]!=digest(task['text']) or not np.isfinite(old[0]):
            raise RuntimeError('Reference score/text mismatch')
        pairs.append((task,float(old[0])))
    return pairs


def gate1(scorer, manifest, cache, phase1, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if scorer.batch_size!=1: raise RuntimeError('Gate 1 must start at batch size 1')
    synthetic=scorer.check_backend(phase1,directory/'synthetic.json')
    report={'passed':False,'stage':'synthetic','cohort_scored':False,'backend_hash':digest(scorer.meta)}
    if not synthetic['passed']:
        atomic(directory/'gate1.json',report); return report,[]
    pairs=load_reference(scorer,manifest,cache,phase1)
    identity={'backend_hash':digest(scorer.meta),'manifest_hash':sha_file(manifest),
              'reference_hash':digest([(t['post_id'],t['signal'],v) for t,v in pairs])}
    cp=directory/'pilot_scores_PRIVATE.json'
    state=json.loads(cp.read_text()) if cp.exists() else {'identity':identity,'scores':{}}
    if state['identity']!=identity: raise RuntimeError('Checkpoint identity mismatch')
    for i,(task,_) in enumerate(pairs):
        if str(i) not in state['scores']:
            value=float(scorer.predict([task['text']],task['signal'])[0])
            if not np.isfinite(value) or not 0<=value<=1: raise RuntimeError('Invalid HF score')
            state['scores'][str(i)]=value
            atomic(cp,state)
    signals={}
    for signal in SIGNALS:
        idx=[i for i,(t,_) in enumerate(pairs) if t['signal']==signal]
        hf=np.array([state['scores'][str(i)] for i in idx]); old=np.array([pairs[i][1] for i in idx])
        stats=comparison_stats(hf,old)
        rho=stats['spearman']
        signals[signal]=dict(stats,passed=rho is not None and np.isfinite(rho) and rho>=.95)
    report.update(identity,stage='pilot_spearman',signals=signals,passed=pilot_gate_passed(signals),threshold=.95)
    atomic(directory/'gate1.json',report)
    return report,pairs


def gate2(scorer, pairs, directory):
    """100 DISTINCT pilot posts, all five questions, compared to batch 1."""
    unique={t['post_id']:t for t,_ in pairs}
    selected=sorted(unique.values(),key=lambda t:digest(['padding',t['post_id']]))[:100]
    if len(selected)!=100: raise RuntimeError('Need exactly 100 pilot posts')
    texts=[t['text'] for t in selected]
    baseline={}; results=[]
    for batch in (1,4,8):
        start=time.perf_counter(); output={}; failed=False
        try:
            for signal in SIGNALS:
                values=[]
                for i in range(0,100,batch): values.extend(scorer.predict(texts[i:i+batch],signal))
                output[signal]=np.array(values)
        except scorer.torch.cuda.OutOfMemoryError:
            scorer.torch.cuda.empty_cache(); failed=True
            if batch==1: raise
        elapsed=time.perf_counter()-start
        if batch==1: baseline=output
        differences={s:float(np.max(np.abs(output[s]-baseline[s]))) for s in SIGNALS} if not failed else {}
        correlations={s:comparison_stats(output[s],baseline[s])['spearman'] for s in SIGNALS} if not failed else {}
        passed=not failed and all(np.isfinite(v) and v<=GATE2_MAX_GAP for v in differences.values()) and all(
            r is not None and np.isfinite(r) and r>=GATE2_MIN_SPEARMAN for r in correlations.values())
        results.append({'batch_size':batch,'posts':100,'calls':500,'seconds':elapsed,
                        'oom':failed,'max_differences':differences,'spearman_by_signal':correlations,'passed':passed})
    valid=[r for r in results if r['passed']]
    selected_batch=min(valid,key=lambda r:r['seconds'])['batch_size']
    report={'passed':True,'selected_batch_size':selected_batch,'tolerance':GATE2_MAX_GAP,
            'minimum_spearman':GATE2_MIN_SPEARMAN,'results':results,
            'note':'Only passing batch sizes are eligible; batch 1 remains the safe fallback.'}
    atomic(Path(directory)/'gate2.json',report)
    return report


def gate3_pages(scorer,pairs,directory):
    """Prepare private inspection only. Does not invent a completed review."""
    from llm_pilot_review import review_page
    directory=Path(directory)
    state=json.loads((directory/'pilot_scores_PRIVATE.json').read_text())
    top=[]
    for signal in SIGNALS:
        ranked=sorted([(state['scores'][str(i)],t) for i,(t,_) in enumerate(pairs) if t['signal']==signal],key=lambda x:x[0],reverse=True)
        top.extend(t for _,t in ranked[:10])
    page=review_page(top,scorer.config,'backend_inspection','heldout')
    dest=directory/'top50_PRIVATE.html'; dest.write_text(page,encoding='utf-8'); dest.chmod(0o600)
    game=[t for t,_ in pairs if 'game' in t.get('domains',[])]
    if game:
        path=directory/'game_candidates_PRIVATE.html'
        path.write_text(review_page(game,scorer.config,'backend_game_inspection','heldout'),encoding='utf-8'); path.chmod(0o600)
    # Approval must be an actual recorded inspection, tied to these exact pages.
    report={'status':'awaiting_actual_read','passed':False,'top_questions':len(top),
            'top_page_sha256':sha_file(dest),'game_candidate_questions':len(game),
            'game_page_sha256':sha_file(directory/'game_candidates_PRIVATE.html') if game else None,
            'backend_hash':digest(scorer.meta),
            'note':'Game domain tags are candidates, not non-emotional ground truth.'}
    atomic(directory/'gate3.json',report)
    return report


def assemble_gate_report(scorer,directory,review_path):
    """Call only AFTER an actual read; never manufactures review attestations."""
    directory=Path(directory)
    first=json.loads((directory/'gate1.json').read_text())
    speed=json.loads((directory/'gate2.json').read_text())
    page=json.loads((directory/'gate3.json').read_text())
    review=json.loads(Path(review_path).read_text())
    if not (first.get('passed') and first['backend_hash']==digest(scorer.meta)
            and pilot_gate_passed(first.get('signals',{})) and speed.get('passed')
            and speed.get('tolerance')==GATE2_MAX_GAP
            and speed.get('minimum_spearman')==GATE2_MIN_SPEARMAN):
        raise RuntimeError('Earlier gate failed')
    if not (review.get('passed') is True and review.get('top_posts_read') is True
            and review.get('non_emotional_games_checked') is True and review.get('reviewer')
            and review.get('top_page_sha256')==page['top_page_sha256']
            and page['game_page_sha256'] is not None
            and review.get('game_page_sha256')==page['game_page_sha256']
            and page['backend_hash']==first['backend_hash']):
        raise RuntimeError('Actual top-post and game review is incomplete')
    batch=speed['selected_batch_size']
    if not any(r['batch_size']==batch and r['passed'] for r in speed['results']):
        raise RuntimeError('Selected batch size did not pass padding check')
    meta=dict(scorer.meta,batch_size=batch)
    bundle=dict(first,backend_hash=digest(meta),speed_gate_passed=True,review_gate_passed=True,
                validated_batch_size=batch,review_sha256=sha_file(review_path),passed=True)
    atomic(directory/'all_gates.json',bundle)
    return bundle
