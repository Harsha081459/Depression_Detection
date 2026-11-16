"""Build blinded, offline human review pages ON THE LAB MACHINE only.

The private manifest keeps scores/labels/splits away from the reviewers.
No post text is printed. Held-out pages require frozen version-1 prompts.
"""
import argparse
from collections import Counter, deque
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

from llm_signal_scorer import SIGNALS, canonical_hash, load_prompts, write_json, SafeError


def choose_tasks(records, scores, config, attempt=0):
    # Exact capacity matching can reassign a multi-signal post, avoiding false
    # shortfalls from a greedy sampler. User -> post -> score stratum enforces
    # three distinct posts/user, and thirty per stratum.
    # First prefer 600 distinct posts. If needed, one post can be assessed for
    # two DIFFERENT signals; every signal still has 60 distinct posts/half.
    graph, links, post_links, user_links = {}, [], [], []
    source, sink = ('source',), ('sink',)
    def edge(a, b, cap):
        graph.setdefault(a, [])
        graph.setdefault(b, [])
        forward = [b, cap, len(graph[b])]
        graph[a].append(forward)
        graph[b].append([a, 0, len(graph[a]) - 1])
        return forward
    cells = [(split, signal, high) for split in ('development', 'heldout') for signal in SIGNALS for high in (True, False)]
    for cell in cells:
        edge(('cell',) + cell, sink, 30)
    users = set()
    ordered = sorted(records, key=lambda r: (not bool(r.get('domains')), canonical_hash([attempt, r['post_id']])))
    for r in ordered:
        user, post = ('user', r['user_id']), ('post', r['post_id'])
        if user not in users:
            user_links.append(edge(source, user, 3))
            users.add(user)
        post_links.append(edge(user, post, 1))
        for signal in SIGNALS:
            key = (r['post_id'], signal)
            if key not in scores:
                continue
            cell = (r['split'], signal, scores[key] >= .5)
            links.append((r, signal, edge(post, ('cell',) + cell, 1)))
    total_flow, reuse_enabled = 0, False
    while True:
        level, queue = {source: 0}, deque([source])
        while queue:
            a = queue.popleft()
            for b, cap, _ in graph.get(a, []):
                if cap > 0 and b not in level:
                    level[b] = level[a] + 1
                    queue.append(b)
        if sink not in level:
            if total_flow < 600 and not reuse_enabled:
                for link in post_links:
                    # Only reuse already selected posts. Freeze the per-user
                    # post set so extra judgments cannot add a fourth post.
                    link[1] = 1 if link[1] == 0 else 0
                for link in user_links:
                    link[1] += 3
                reuse_enabled = True
                continue
            break
        cursor = dict.fromkeys(graph, 0)
        def push(a, amount):
            if a == sink:
                return amount
            while cursor[a] < len(graph[a]):
                link = graph[a][cursor[a]]
                b, cap, reverse = link
                if cap > 0 and level.get(b) == level[a] + 1:
                    sent = push(b, min(amount, cap))
                    if sent:
                        link[1] -= sent
                        graph[b][reverse][1] += sent
                        return sent
                cursor[a] += 1
            return 0
        while True:
            sent = push(source, 600)
            if not sent:
                break
            total_flow += sent
    tasks = []
    for r, signal, link in links:
        if link[1] != 0:
            continue
        tasks.append({'item_id': canonical_hash(['blind_item', signal, r['post_id']])[:20],
                      'post_id': r['post_id'], 'user_id': r['user_id'], 'label': r['label'],
                      'split': r['split'], 'signal': signal, 'score': scores[r['post_id'], signal],
                      'sampling_high': scores[r['post_id'], signal] >= .5,
                      'word_list': r['lexical'][signal], 'domains': r.get('domains', []),
                      'text': r['text'], 'model_truncated_flag': int(len(r['text']) > config['max_chars'])})
    counts = Counter((t['split'], t['signal'], t['score'] >= .5) for t in tasks)
    shortfalls = [{'split': split, 'signal': signal, 'high': high, 'selected': counts[split, signal, high], 'required': 30}
                  for split, signal, high in cells if counts[split, signal, high] < 30]
    return tasks, shortfalls


HTML = '''<!doctype html><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'">
<title>Private signal review</title><style>
body{font:17px system-ui;max-width:900px;margin:36px auto;padding:0 24px;background:#f5f7fa;color:#192735}
article{background:white;padding:24px;border-radius:12px;border:1px solid #dce3e9}pre{white-space:pre-wrap;font:17px/1.55 system-ui}
button{font:inherit;margin:8px 8px 8px 0;padding:10px 20px;cursor:pointer}#definition{font-size:15px;color:#465666}#status{margin:20px 0}
</style><h1>Private signal review</h1><p>Judge only the writer's own experience in the displayed post. Work independently. No user label or model score is shown. Keep this page and your answers on the lab machine.</p>
<p id="status"></p><article><h2 id="signal"></h2><p id="definition"></p><hr><pre id="post"></pre>
<button id="yes">Yes</button><button id="no">No</button><p id="answer"></p></article>
<button id="back">Previous</button><button id="next">Next</button><button id="unanswered">Next unanswered</button><button id="save">Save answers</button>
<p><input id="reference" placeholder="Blind item reference"><button id="find">Find reference</button></p>
<script>
const tasks=__TASKS__, key=__KEY__, definitions=__DEFINITIONS__;
let index=0, answers=JSON.parse(localStorage.getItem(key)||'{}');
function show(){const t=tasks[index];document.querySelector('#signal').textContent=t.signal.replaceAll('_',' ');
document.querySelector('#definition').textContent='Yes when the writer '+definitions[t.signal].yes+'. Answer No if '+definitions[t.signal].no+'.';
document.querySelector('#post').textContent=t.text;document.querySelector('#answer').textContent=answers[t.item_id]?'Your answer: '+answers[t.item_id]:'Unanswered';
document.querySelector('#status').textContent=`Item ${index+1} of ${tasks.length}. Answered: ${Object.keys(answers).length}. Reference: ${t.item_id}`;}
function answer(value){answers[tasks[index].item_id]=value;localStorage.setItem(key,JSON.stringify(answers));if(index<tasks.length-1)index++;show();}
document.querySelector('#yes').onclick=()=>answer('Yes');document.querySelector('#no').onclick=()=>answer('No');
document.querySelector('#back').onclick=()=>{index=Math.max(0,index-1);show();};document.querySelector('#next').onclick=()=>{index=Math.min(tasks.length-1,index+1);show();};
document.querySelector('#unanswered').onclick=()=>{const found=tasks.findIndex(t=>!answers[t.item_id]);if(found>=0)index=found;show();};
document.querySelector('#find').onclick=()=>{const found=tasks.findIndex(t=>t.item_id===document.querySelector('#reference').value.trim());if(found>=0)index=found;show();};
document.querySelector('#save').onclick=()=>{const text='item_id,answer\\n'+tasks.map(t=>t.item_id+','+(answers[t.item_id]||'')).join('\\n')+'\\n';
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'}));a.download=key+'.csv';a.click();URL.revokeObjectURL(a.href);};show();
</script>'''


def safe_json(value):
    return json.dumps(value, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def review_page(tasks, config, rater, split):
    blind = [{'item_id': t['item_id'], 'signal': t['signal'], 'text': t['text']} for t in tasks if t['split'] == split]
    blind.sort(key=lambda x: canonical_hash([rater, x['item_id']]))
    key = 'erisk_review_%s_%s_%s' % (split, rater, canonical_hash(config)[:12])
    page = HTML.replace('__TASKS__', safe_json(blind)).replace('__KEY__', safe_json(key)).replace('__DEFINITIONS__', safe_json(config['signals']))
    if rater == 'adjudication':
        page = page.replace('Work independently.', 'Discuss each disagreement and agree one final answer.')
    return page


def adjudication_page(manifest, rater_a, rater_b, output, config, split='development'):
    from evaluate_llm_signals import read_answers
    if os.name != 'posix':
        raise SafeError('private_adjudication_must_stay_on_lab_linux')
    if split == 'heldout' and config['prompt_version'] != 1:
        raise SafeError('heldout_requires_frozen_version1')
    tasks = [t for t in json.loads(Path(manifest).read_text(encoding='utf-8')) if t['split'] == split]
    a, b = read_answers(rater_a), read_answers(rater_b)
    if set(a) != {t['item_id'] for t in tasks} or set(a) != set(b):
        raise SafeError('rater_ids_do_not_match_manifest')
    conflicts = [t for t in tasks if a[t['item_id']] != b[t['item_id']]]
    page = review_page(conflicts, config, 'adjudication', split) if conflicts else '<p>No disagreements. No adjudication file is needed.</p>'
    Path(output).write_text(page, encoding='utf-8')
    os.chmod(str(output), 0o600)
    return {'items': len(tasks), 'disagreements': len(conflicts)}


def build(pool_path, cache_path, output, config, heldout=False):
    if os.name != 'posix':
        raise SafeError('review_pages_must_stay_on_lab_linux')
    if heldout and config['prompt_version'] != 1:
        raise SafeError('freeze_version1_before_releasing_heldout_pages')
    records = [json.loads(line) for line in Path(pool_path).read_text(encoding='utf-8').splitlines()]
    by_post = {r['post_id']: r for r in records}
    db = sqlite3.connect(str(cache_path))
    result = db.execute('SELECT post_id,signal,score FROM scores WHERE prompt_hash=? AND model_digest=?',
                        (canonical_hash(config), config['expected_digest'])).fetchall()
    db.close()
    scores = {(pid, signal): score for pid, signal, score in result}
    manifest_path = Path(output) / 'private_manifest.json'
    locked = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    if locked is not None:
        # Once raters could have seen a task, never resample it after tuning.
        # The original model-score stratum remains the sampling stratum.
        for task in locked:
            task.setdefault('sampling_high', task['score'] >= .5)
            key = (task['post_id'], task['signal'])
            if key not in scores:
                raise SafeError('rescore_locked_pilot_with_current_prompts_before_release')
            if task['post_id'] not in by_post:
                raise SafeError('locked_post_missing_from_approved_pool')
            task['score'] = scores[key]
            task['text'] = by_post[task['post_id']]['text']
            task['model_truncated_flag'] = int(len(task['text']) > config['max_chars'])
        strata = Counter((t['split'], t['signal'], t['sampling_high']) for t in locked)
        if (len(locked) != 600 or len({t['post_id'] for t in locked}) < 550
                or len({(t['post_id'], t['signal']) for t in locked}) != 600
                or max(Counter(t['post_id'] for t in locked).values()) > 2
                or max(len({t['post_id'] for t in locked if t['user_id'] == uid}) for uid in {t['user_id'] for t in locked}) > 3
                or len(strata) != 20 or any(n != 30 for n in strata.values())):
            raise SafeError('locked_pilot_manifest_failed_sampling_invariants')
    # Exact matching proves quota feasibility. Reorder only to satisfy the
    # additional both-cohort-groups/domain checks, never change a quota/cap.
    best = None
    for attempt in range(30):
        if locked is None:
            tasks, shortfalls = choose_tasks(records, scores, config, attempt)
        else:
            tasks, shortfalls = locked, []
        both_groups = all({t['label'] for t in tasks if t['split'] == split and t['signal'] == signal} == {0, 1}
                          for split in ('development', 'heldout') for signal in SIGNALS)
        domains = {split: dict(Counter(d for t in tasks if t['split'] == split for d in t['domains']))
                   for split in ('development', 'heldout')}
        domains_present = all(domains[split].get(domain, 0) > 0 for split in domains for domain in ('game', 'sport', 'news'))
        checks = {'both_cohort_groups_per_signal_and_split': both_groups,
                  'domains_present_in_both_splits': domains_present, 'domains_by_split': domains,
                  'at_least_550_distinct_posts': len({t['post_id'] for t in tasks}) >= 550}
        ready = not shortfalls and both_groups and domains_present and checks['at_least_550_distinct_posts']
        if best is None or (len(tasks), ready) > (len(best[0]), best[3]):
            best = (tasks, shortfalls, checks, ready)
        if ready or len(tasks) < 600 or locked is not None:
            break
    tasks, shortfalls, checks, ready = best
    summary = {'selected_tasks': len(tasks), 'users': len({x['user_id'] for x in tasks}),
               'distinct_posts': len({x['post_id'] for x in tasks}),
               'max_posts_per_user': max((len({x['post_id'] for x in tasks if x['user_id'] == uid}) for uid in {x['user_id'] for x in tasks}), default=0),
               'max_judgments_per_user': max(Counter(x['user_id'] for x in tasks).values(), default=0),
               'max_signals_per_post': max(Counter(x['post_id'] for x in tasks).values(), default=0),
               'model_truncated_judgments': sum(t['model_truncated_flag'] for t in tasks),
               'shortfalls': shortfalls, 'ready_for_review': ready, 'sampling_checks': checks,
               'prompt_hash': canonical_hash(config), 'prompt_version': config['prompt_version'],
               'locked_sample_reused': locked is not None,
               'current_scores_crossing_sampling_threshold': sum(x.get('sampling_high', x['score'] >= .5) != (x['score'] >= .5) for x in tasks),
               'quotas': dict(Counter('%s/%s/%s' % (x['split'], x['signal'], 'high' if x.get('sampling_high', x['score'] >= .5) else 'low') for x in tasks))}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'sampling_summary.json', summary)
    if not ready:
        return summary
    write_json(output / 'private_manifest.json', tasks)
    for rater in ('A', 'B'):
        for split in (('development', 'heldout') if heldout else ('development',)):
            page = review_page(tasks, config, rater, split)
            destination = output / ('review_%s_%s.html' % (split, rater))
            destination.write_text(page, encoding='utf-8')
            os.chmod(str(destination), 0o600)
    summary['released'] = ['development', 'heldout'] if heldout else ['development']
    write_json(output / 'sampling_summary.json', summary)
    return summary


def expand_requests(pool_path, cache_path, summary_path, output_path, config):
    """Ask more signals on already approved posts, never add a new post."""
    records = [json.loads(line) for line in Path(pool_path).read_text(encoding='utf-8').splitlines()]
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    db = sqlite3.connect(str(cache_path))
    scored = set(db.execute('SELECT post_id,signal FROM scores WHERE prompt_hash=? AND model_digest=?',
                            (canonical_hash(config), config['expected_digest'])))
    db.close()
    # Sampling enrichment only: these patterns are never classifier features.
    enrichment = {
        'self_blame': r'\b(worthless|useless|burden|guilty|guilt|ashamed|shame|stupid|failure)\b|not good enough|hate myself',
        'helplessness': r'\b(hopeless|helpless|trapped|powerless|pointless)\b|cannot cope|can.t cope|need help|do this anymore|nothing.*chang',
        'isolation': r'\b(lonely|isolated|unwanted|rejected|abandoned)\b|no one|nobody|no friends|feel alone',
        'family_strain': r'\b(mother|father|parents|family|partner|wife|husband|friend|friends)\b',
        'emptiness': r'\b(numb|empty|exhausted|crying|insomnia|drained)\b|feel anything|no energy|can.t sleep|cannot sleep',
    }
    additions = 0
    for cell in summary.get('shortfalls', []):
        signal, split = cell['signal'], cell['split']
        candidates = [r for r in records if r['split'] == split and (r['post_id'], signal) not in scored
                      and signal not in r['requested_signals']]
        candidates.sort(key=lambda r: (not bool(re.search(enrichment[signal], r['text'], flags=re.I)),
                                       not bool(r['lexical'][signal]), canonical_hash([signal, r['post_id']])))
        per_user = Counter()
        added = 0
        for record in candidates:
            if per_user[record['user_id']] >= 2:
                continue
            record['requested_signals'].append(signal)
            per_user[record['user_id']] += 1
            additions += 1
            added += 1
            if added >= 60:
                break
    with Path(output_path).open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return {'existing_posts': len(records), 'additional_questions': additions, 'new_posts_added': 0}


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompts.json'))
    ap.add_argument('--release-heldout', action='store_true')
    ap.add_argument('--expand-from-summary', type=Path)
    ap.add_argument('--adjudication-manifest', type=Path)
    ap.add_argument('--rater-a', type=Path)
    ap.add_argument('--rater-b', type=Path)
    ap.add_argument('--split', choices=('development', 'heldout'), default='development')
    args = ap.parse_args()
    try:
        if args.adjudication_manifest:
            if args.rater_a is None or args.rater_b is None:
                raise SafeError('adjudication_requires_two_rater_files')
            summary = adjudication_page(args.adjudication_manifest, args.rater_a, args.rater_b,
                                        args.output, load_prompts(args.prompts), args.split)
            print(json.dumps(summary))
            return 0
        if args.pool is None or args.cache is None:
            raise SafeError('sampling_requires_pool_and_cache')
        if args.expand_from_summary:
            if os.name != 'posix':
                raise SafeError('private_expansion_must_run_on_lab_linux')
            summary = expand_requests(args.pool, args.cache, args.expand_from_summary, args.output, load_prompts(args.prompts))
            print(json.dumps(summary))
            return 0
        summary = build(args.pool, args.cache, args.output, load_prompts(args.prompts), args.release_heldout)
        print(json.dumps(summary), flush=True)
        return 0 if summary['ready_for_review'] else 2
    except SafeError as e:
        print(json.dumps({'error': str(e)}))
        return 2
    except Exception:
        print(json.dumps({'error': 'review_build_failed_details_withheld'}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
