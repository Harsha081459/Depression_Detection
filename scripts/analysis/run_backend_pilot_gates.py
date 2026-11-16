"""Pilot-only worker. No full-cohort scoring or evaluation path."""
import argparse
import json
import os
from pathlib import Path
from erisk_full_pipeline import LocalScorer,atomic
from backend_pilot_gates import gate1


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','prompts','manifest','cache','phase1-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if os.name!='posix': raise RuntimeError('Private pilot must remain on the lab')
    args.output.mkdir(parents=True,exist_ok=True)
    status=args.output/'status.json'
    try:
        from verify_hf_checkpoint import verify_checkpoint
        setup=json.loads(Path('results/hf_backend_setup.json').read_text())
        verify_checkpoint(args.model_dir,setup['revision'],args.output/'checkpoint_integrity.json')
        atomic(status,{'stage':'loading_hf_model','cohort_scored':False})
        scorer=LocalScorer(args.model_dir,args.prompts,batch_size=1)
        atomic(status,{'stage':'gate1_running','cohort_scored':False})
        first,pairs=gate1(scorer,args.manifest,args.cache,args.phase1_report,args.output)
        if not first['passed']:
            atomic(status,{'stage':'stopped_gate1_failed','cohort_scored':False}); return 2
        atomic(status,{'stage':'gate1_passed_waiting_for_interpretation','cohort_scored':False})
        return 0
    except Exception as e:
        atomic(status,{'stage':'stopped_execution_error','error_type':type(e).__name__,'cohort_scored':False})
        return 2


if __name__=='__main__': raise SystemExit(main())
