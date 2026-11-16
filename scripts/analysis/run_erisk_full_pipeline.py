"""Run on lab Linux. Never uploads text or prints private identifiers."""
import argparse
import os
import json
from pathlib import Path
import pandas as pd
from erisk_full_pipeline import (load_data,freeze_split,prepare_posts,LocalScorer,
                                sanity,evaluate_all)


def main():
    os.umask(0o077)
    ap=argparse.ArgumentParser()
    for name in ('posts','labels','pilot-users','prompts','model-dir','phase1-report','work','gate-report'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=1)
    args=ap.parse_args()
    if os.name!='posix': raise RuntimeError('Run private-data work on the lab Linux machine')
    assert args.batch_size>=1
    work=args.work.resolve(); work.mkdir(parents=True,exist_ok=True)
    raw,labels=load_data(args.posts,args.labels)
    split=freeze_split(labels,args.pilot_users,work/'split.json')
    posts,origins=prepare_posts(raw)
    scorer=LocalScorer(args.model_dir,args.prompts,args.batch_size)
    # Separate pilot-only worker and an actual recorded read must finish first.
    check=json.loads(args.gate_report.read_text())
    scores=scorer.score(posts,work/'scores',check)
    del scorer
    import torch
    torch.cuda.empty_cache()
    sanity(posts,scores,split,work/'private_sanity')
    result=evaluate_all(posts,scores,origins,split,work/'evaluation')
    # This is the only printed cohort table; no posts or user identifiers.
    print('HEADLINE: prespecified 5x5 repeated nested user CV')
    print(pd.read_csv(work/'evaluation/repeated_cv/headline_cv.csv').to_csv(index=False))
    print('SENSITIVITY: repeated nested CV excluding all pilot-exposed users')
    print(pd.read_csv(work/'evaluation/repeated_cv_without_pilot/headline_cv.csv').to_csv(index=False))
    print('SECONDARY CHECK: fixed user holdout, train-selected single-signal summaries')
    print(result.to_csv(index=False))
    print('Small AUROC gaps with 523 users may be noise; inspect confidence intervals.')


if __name__=='__main__': main()
