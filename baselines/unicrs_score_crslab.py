"""Score a UniCRS WWW27 dump on the KBRD/CRSLab sample definition (RECOMMENDER turns with movies; one sample per movie).
dump row: [processed_line, label_unicrs_id, top100 unicrs ids] where process.py wrote one processed line per turn (all turns).
Checks the label sequence against a KBRD dump of the same split (labels in CRSLab ids via crslab_id.json).
usage: unicrs_score_crslab.py <unicrs_data_name> <unicrs.dump> <kbrd_dump.jsonl>"""
import json, sys, numpy as np
U = "/path/to/base/projects/crs/baselines/unicrs/src/data/" + sys.argv[1]
e2i = json.load(open(f"{U}/entity2id.json")); cid = {int(k): v for k, v in json.load(open(f"{U}/crslab_id.json")).items()}
items = set(json.load(open(f"{U}/item_ids.json")))
dump = {}
for l in open(sys.argv[2]):
    ln, lab, top = json.loads(l); dump[(ln, lab)] = top
samples, off = [], 0
for l in open(f"{U}/test_data_dbpedia.jsonl"):
    turns = json.loads(l)
    first_block = True   # CRSLab merges consecutive same-role turns and skips the first merged block (no context)
    for t, tu in enumerate(turns):
        if first_block and t > 0 and tu["role"] != turns[0]["role"]: first_block = False
        if first_block: continue
        if tu["role"] == "RECOMMENDER":
            for m in tu["movie_link"]:
                if m in e2i: samples.append((off + t, e2i[m]))   # CRSLab does not filter labels by the item pool
    off += len(turns)
rk, miss = [], 0
for s in samples:
    top = dump.get(s)
    if top is None: miss += 1; rk.append(10 ** 6); continue
    rk.append(top.index(s[1]) + 1 if s[1] in top else 10 ** 6)
rk = np.array(rk)
kb = [json.loads(l) for l in open(sys.argv[3])]; ep = max(r[0] for r in kb if r[1] == "test")
kl = [r[4] for r in kb if r[1] == "test" and r[0] == ep]; ul = [cid[s[1]] for s in samples]
print(json.dumps({"n_unicrs": len(samples), "n_kbrd": len(kl), "same_order": ul == kl, "pos_match": sum(a == b for a, b in zip(ul, kl)),
                  "missing_in_dump": miss, **{f"HR@{k}": round(float((rk <= k).mean()), 4) for k in (1, 10, 50)},
                  "NDCG@10": round(float(np.where(rk <= 10, 1 / np.log2(rk + 1), 0).mean()), 4)}))
