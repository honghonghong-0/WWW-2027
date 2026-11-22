"""Score a trained CRSLab KBRD/KGSF on Reddit-Movie requests with EXACTLY the ladder's evaluator (reddit_ladder.py):
per request, rank all 50,287 items; context items masked; multi-gold Recall/NDCG/HIT@10; views all | warm | cold | warm-unseen
(own-history items masked and removed from gold). Context entities = the CRSLab Seeker utterance (ctx movies + mentions), same as training.
Saves per-request metrics + top-100 ids for paired bootstrap / comparison.
usage (from baselines/crslab): python reddit_crs_eval.py <config.yaml> <split> <tag>"""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.data.dataloader.utils import padded_tensor, get_onehot, truncate
from crslab.system import get_system
cfg, SP, TAG = sys.argv[1:4]
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
opt = Config(cfg, gpu=__import__("os").environ.get("CRS_GPU", "0")); TOK = opt['tokenize']; TOK = TOK['rec'] if isinstance(TOK, dict) else TOK  # WWW27: ReDial uses per-stage tokenize
ds = get_dataset(opt, TOK, False, False)
dl = lambda d: get_dataloader(opt, d, ds.vocab); vdl = dl(ds.valid_data)
if isinstance(opt['tokenize'], dict):   # WWW27: per-task structures as in quick_start (same nltk dataset for every task)
    T_ = list(opt['tokenize']); vdl = {t: dl(ds.valid_data) for t in T_}
    system = get_system(opt, {t: dl(ds.train_data) for t in T_}, vdl, {t: dl(ds.test_data) for t in T_}, {t: ds.vocab for t in T_}, {t: ds.side_data for t in T_}, restore_system=True)
    vdl = vdl['rec']
else:
    system = get_system(opt, dl(ds.train_data), vdl, dl(ds.test_data), ds.vocab, ds.side_data, restore_system=True)
MNAME = opt.get('model', opt.get('rec_model')); isReDial = MNAME == 'ReDialRec'  # WWW27: ReDial rec-only
model = system.rec_model if isReDial else system.model; model.eval(); e2i = ds.vocab['entity2id']; isKGSF = MNAME == 'KGSF'
items = json.load(open(f"{P}/items.json"))["items"]; idx = torch.tensor([e2i[i] for i in items], device=system.device); ipos = {i: k for k, i in enumerate(items)}
TH = json.load(open(f"{P}/threads.json"))
raw = {c["conv_id"]: c for c in json.load(open(f"/path/to/base/projects/crs/baselines/crslab/data/dataset/reddit/nltk/{SP}_data.json"))}
U = [json.loads(l) for l in open(f"{P}/requests_{SP}.jsonl")]
DISC = 1 / np.log2(np.arange(2, 12))
def met(top, gs):
    if not gs: return None
    rel = np.array([t in gs for t in top[:10]], float); idcg = DISC[:min(10, len(gs))].sum()
    return (rel.sum() / len(gs), (rel * DISC).sum() / idcg, float(rel.any()))
rows = []; nomen = 0
with torch.no_grad():
    for s in range(0, len(U), 256):
        ch = U[s:s + 256]; ctx = []
        for x in ch:
            u0 = raw[x["req"]]["dialog"][0]; ents = []
            for e in u0["entity"] + u0["movies"]:
                if e in e2i and e2i[e] not in ents: ents.append(e2i[e])
            nomen += not ents; ctx.append(truncate(ents, getattr(vdl, 'entity_truncate', None), truncate_tail=False))
        if isKGSF:
            w2i = ds.vocab['word2id']; wds = [truncate([w2i[w] for w in raw[x["req"]]["dialog"][0]["word"] if w in w2i], vdl.word_truncate, truncate_tail=False) for x in ch]   # concept words of the Seeker utterance (as in training)
            batch = (padded_tensor(ctx, vdl.pad_entity_idx, pad_tail=False).to(system.device), padded_tensor(wds, vdl.pad_word_idx, pad_tail=False).to(system.device),
                     get_onehot(ctx, vdl.n_entity).to(system.device), torch.zeros(len(ch), dtype=torch.long, device=system.device))
            sc = model.recommend(batch, "test")[-1]
        elif isReDial:
            sc = model.forward({'context_entities': get_onehot(ctx, vdl.n_entity).to(system.device), 'item': torch.zeros(len(ch), dtype=torch.long, device=system.device)}, mode='test')[-1]
        else:
            sc = model.recommend({'context_entities': ctx, 'item': torch.zeros(len(ch), dtype=torch.long, device=system.device)}, 'test')[-1]
        S = sc[:, idx].float().cpu().numpy()
        for k, x in enumerate(ch):
            v = S[k].copy(); ci = [ipos[c] for c in x["ctx_items"] if c in ipos]; v[ci] = -1e9
            top = np.argsort(-v)[:100]; g = {ipos[c] for c in x["gold"] if c in ipos}
            hs = {ipos[j] for h in x["hist"] for j in TH[h]["items"] + TH[h]["rec"] + [i for f in TH[h]["fb"] for i in f["items"]] if j in ipos}
            vu = v.copy(); vu[list(hs)] = -1e9; topu = np.argsort(-vu)[:10]
            rows.append(dict(req=x["req"], user=x["user"], warm=bool(x["hist"]), m=met(top, g), mu=met(topu, g - hs) if x["hist"] else None, top100=top.tolist()))
def agg(sel, key="m"):
    v = [r[key] for r in sel if r[key] is not None]; return [round(float(a), 4) for a in np.mean(v, 0)] + [len(v)] if v else None
res = dict(model=MNAME, split=SP, n=len(rows), requests_without_context_entity=nomen, all=agg(rows), warm=agg([r for r in rows if r["warm"]]),
           cold=agg([r for r in rows if not r["warm"]]), warm_unseen=agg([r for r in rows if r["warm"]], "mu"))
json.dump(res, open(f"{W}/results/reddit_crs_{TAG}_{SP}.json", "w"), indent=1)
with open(f"{W}/results/reddit_crs_{TAG}_{SP}_rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(res))
