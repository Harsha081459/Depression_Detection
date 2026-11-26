"""FOR TEAM MEMBERS ONLY, in your own terminal: print the raw text of a walkthrough case's key writings so
you can write the paraphrases. Prints to the screen only; writes nothing. Do not paste the output into
chat tools, LLMs, documents or logs.

    python demo/show_key_writings.py TP
"""
import json
import sys

import pandas as pd

from common import PRIV, RQ, load_model, prepare, load_writings


def main(name):
    wt = json.loads((PRIV / 'walkthrough.json').read_text())[name]
    ids = {v: k for k, v in json.loads((PRIV / 'case_ids_PRIVATE.json').read_text()).items()}
    uid = ids[wt['case_id']]
    m = load_model()
    full = pd.read_parquet(RQ / 'writings_PRIVATE.parquet', columns=['user_id', 'ts_sec', 'disclosure', 'text'])
    full = full[(full.user_id.astype(str) == uid) & ~full.disclosure].sort_values('ts_sec', kind='mergesort')
    x = prepare(load_writings([uid]), m['cuts'])
    assert len(x) == len(full)
    full = full.reset_index(drop=True)
    for i in wt['key_writings']:
        print(f'--- {wt["case_id"]} writing #{i} ---')
        print(full.text.iloc[i - 1][:1500])
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'TP')
