"""Embed new writings with all-mpnet-base-v2 (mean pooling + L2), same as the stored cohort embeddings.
Runs in the GPU environment (.venv: torch + transformers; no scipy / scikit-learn).

  .venv/Scripts/python demo/embed_live.py <in.json> <out.npy>        in.json = ["text", ...]
  .venv/Scripts/python demo/embed_live.py --check                       stored vs recomputed, 50 cohort writings
  .venv/Scripts/python demo/embed_live.py --symptoms <out.npy>          the 21 symptom statements
Prints numbers only.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

NAME = 'sentence-transformers/all-mpnet-base-v2'
ROOT = Path(__file__).resolve().parents[1]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def embed(texts, bs=32):
    tok = AutoTokenizer.from_pretrained(NAME); m = AutoModel.from_pretrained(NAME).to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            enc = tok([t if t else '[empty]' for t in texts[s:s + bs]], truncation=True, max_length=256,
                      padding=True, return_tensors='pt').to(dev)
            h = m(**enc).last_hidden_state; a = enc['attention_mask'].unsqueeze(-1).float()
            e = (h * a).sum(1) / a.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.vstack(out).astype(np.float32)


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        import pandas as pd
        d = pd.read_csv(ROOT / 'data/processed/erisk/task2_target_text.csv', dtype={'user_id': str})
        d['text'] = (d.title.fillna('') + ' ' + d.body.fillna('')).str.strip()
        d['ts_sec'] = pd.to_datetime(d.date, utc=True).dt.tz_localize(None).to_numpy().astype('datetime64[s]').astype(np.int64)
        d = d.sort_values(['user_id', 'ts_sec'], kind='mergesort').reset_index(drop=True)
        idx = np.random.default_rng(3).choice(len(d), 50, replace=False)
        E = np.load(ROOT / 'private/encoder/embeddings_PRIVATE.npy', mmap_mode='r')
        new = embed([d.text.iloc[i] for i in idx])
        cos = (np.asarray(E[idx], np.float32) * new).sum(1)
        print(json.dumps({'n': 50, 'cos_min': float(cos.min()), 'cos_mean': float(cos.mean())}))
    elif sys.argv[1] == '--all':
        # <in.json> <out_prefix>: mpnet embeddings, go_emotions probabilities, bge symptom similarities
        from transformers import AutoModelForSequenceClassification
        texts = json.load(open(sys.argv[2], encoding='utf-8')); pre = sys.argv[3]
        np.save(pre + '_mpn.npy', embed(texts))
        name = 'SamLowe/roberta-base-go_emotions'
        tok = AutoTokenizer.from_pretrained(name); m = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
        with torch.no_grad():
            enc = tok([t if t else '[empty]' for t in texts], truncation=True, max_length=256, padding=True, return_tensors='pt').to(dev)
            np.save(pre + '_emo.npy', torch.sigmoid(m(**enc).logits.float()).cpu().numpy())
        bge = 'BAAI/bge-large-en-v1.5'
        tok = AutoTokenizer.from_pretrained(bge); m = AutoModel.from_pretrained(bge).to(dev).eval()
        sym = json.load(open(ROOT / 'research_v11/symptoms.json'))['symptoms']

        def cls(tt):
            with torch.no_grad():
                enc = tok([t if t else '[empty]' for t in tt], truncation=True, max_length=256, padding=True, return_tensors='pt').to(dev)
                return torch.nn.functional.normalize(m(**enc).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
        np.save(pre + '_bdi.npy', cls(texts) @ cls(list(sym.values())).T)
        print(json.dumps({'writings': len(texts)}))
    elif sys.argv[1] == '--symptoms':
        sym = json.load(open(ROOT / 'research_v11/symptoms.json'))['symptoms']
        np.save(sys.argv[2], embed(list(sym.values())))
        print(json.dumps({'statements': len(sym)}))
    else:
        texts = json.load(open(sys.argv[1], encoding='utf-8'))
        np.save(sys.argv[2], embed(texts))
        print(json.dumps({'writings': len(texts)}))
