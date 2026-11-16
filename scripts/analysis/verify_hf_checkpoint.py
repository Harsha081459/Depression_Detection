"""Validate local weights against the PINNED public HF revision before loading."""
import json
from pathlib import Path
from erisk_full_pipeline import atomic,sha_file


def verify_checkpoint(directory,revision,output):
    from huggingface_hub import HfApi
    directory=Path(directory)
    index=json.loads((directory/'model.safetensors.index.json').read_text())
    expected=set(index['weight_map'].values())
    present={p.name for p in directory.glob('*.safetensors')}
    if present!=expected or list(directory.glob('.cache/huggingface/download/**/*.incomplete')):
        raise RuntimeError('Checkpoint incomplete: missing/extra shards or partial download')
    info=HfApi().model_info('Qwen/Qwen2.5-7B-Instruct',revision=revision,files_metadata=True)
    if info.sha!=revision: raise RuntimeError('Checkpoint revision mismatch')
    remote={s.rfilename:s for s in info.siblings}
    rows=[]
    for name in sorted(expected):
        lfs=remote[name].lfs
        expected_hash=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']
        actual_hash=sha_file(directory/name)
        if actual_hash!=expected_hash: raise RuntimeError('Checkpoint SHA256 mismatch')
        rows.append({'file':name,'sha256':actual_hash,'verified':True,'bytes':(directory/name).stat().st_size})
    report={'passed':True,'revision':revision,'expected_shards':len(expected),'verified_shards':len(rows),'shards':rows}
    atomic(output,report)
    return report
