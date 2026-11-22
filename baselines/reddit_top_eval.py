"""Score a CRSLab-style '.top' dump (per sample: [conv_id, label, top-100 entity ids, context-masked]) on Reddit-Movie
with the same metric as reddit_crs_eval.py / the ladder: per request, multi-gold Recall/NDCG/HIT@10, ctx items masked;
views all | warm | cold | warm_unseen (own-history items removed from the list and from gold).
usage: reddit_top_eval.py <dump.top> <split> <tag> [id_maps.json]"""
import sys, json, numpy as np
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
top_f, SP, TAG = sys.argv[1:4]; MAPS = sys.argv[4] if len(sys.argv) > 4 else None
E = json.load(open("/path/to/base/projects/crs/baselines/crslab/data/dataset/reddit/nltk/entity2id.json")); i2e = {v: k for k, v in E.items()}
inv = {v: k for k, v in json.load(open(MAPS))["conv"].items()} if MAPS else None
TH = json.load(open(f"{P}/threads.json")); U = [json.loads(l) for l in open(f"{P}/requests_{SP}.jsonl")]
TOP = {}
for l in open(top_f):
    cid, lab, top = json.loads(l)
    if inv is not None: cid = inv[cid]
    TOP.setdefault(cid, top)       # same context for every gold row of a request -> same list
DISC = 1 / np.log2(np.arange(2, 12))
def met(top, gs):
    if not gs: return None
    rel = np.array([t in gs for t in top[:10]], float); idcg = DISC[:min(10, len(gs))].sum()
    return (rel.sum() / len(gs), (rel * DISC[:len(rel)]).sum() / idcg, float(rel.any()))
rows = []; missing = 0; short = 0
for x in U:
    t = TOP.get(x["req"])
    if t is None: missing += 1; t = []
    ids = [i2e.get(e, e) for e in t]; ctx = set(x["ctx_items"]); ids = [i for i in ids if i not in ctx]
    g = set(x["gold"])
    hs = {j for h in x["hist"] for j in TH[h]["items"] + TH[h]["rec"] + [i for f in TH[h]["fb"] for i in f["items"]]}
    idu = [i for i in ids if i not in hs]; short += len(idu) < 10
    rows.append(dict(req=x["req"], user=x["user"], warm=bool(x["hist"]), m=met(ids, g), mu=met(idu, g - hs) if x["hist"] else None, top100=ids))
def agg(sel, key="m"):
    v = [r[key] for r in sel if r[key] is not None]; return [round(float(a), 4) for a in np.mean(v, 0)] + [len(v)] if v else None
res = dict(model=TAG, split=SP, n=len(rows), requests_missing_in_dump=missing, unseen_list_shorter_than_10=short, all=agg(rows),
           warm=agg([r for r in rows if r["warm"]]), cold=agg([r for r in rows if not r["warm"]]), warm_unseen=agg([r for r in rows if r["warm"]], "mu"))
json.dump(res, open(f"{W}/results/reddit_crs_{TAG}_{SP}.json", "w"), indent=1)
with open(f"{W}/results/reddit_crs_{TAG}_{SP}_rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(res))
