"""Export CRSLab rec samples (same construction/order as the KBRD dataloader: Recommender turns x each mentioned movie)
with context text, context entities and label, so our method is scored on exactly the baseline's samples and item pool.
usage (from baselines/crslab): python export_crslab_samples.py <config.yaml> <out_dir>"""
import sys, json, os
sys.path.insert(0, '.')
from crslab.config import Config
from crslab.data import get_dataset
cfg, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
opt = Config(cfg, gpu='-1')
ds = get_dataset(opt, opt['tokenize'], False, False)
V = ds.vocab; i2t = V['ind2tok']; i2e = {v: k for k, v in V['entity2id'].items()}
json.dump({"item_ids": V.get('item_ids') or ds.side_data.get('item_ids'), "id2entity": {str(k): v for k, v in i2e.items()}}, open(f"{out}/vocab.json", "w"))
for sp, data in (("train", ds.train_data), ("valid", ds.valid_data), ("test", ds.test_data)):
    n = 0
    with open(f"{out}/{sp}.jsonl", "w") as f:
        for c in data:
            if c['role'] != 'Recommender': continue
            ctx = [" ".join(i2t.get(t, "") for t in u) for u in c['context_tokens']]
            for m in c['items']:
                f.write(json.dumps({"ctx": ctx, "ctx_entities": c['context_entities'], "item": m, "conv_id": c.get('conv_id')}, ensure_ascii=False) + "\n"); n += 1
    print(sp, n, flush=True)
