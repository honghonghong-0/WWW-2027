"""Reddit v2: tuning + dynamic-affordance candidate reranking (same pre-registered design as fullcat_redial_v2.py, fixed 2026-11-17).
Reuses fullcat_episodic.py up to the feature build with TAUS widened to {.01,.02,.05,.1}; SPACES emb,exp,attr.
(T) additive A1 + g1 z(EP_emb) + g2 z(EP_s), g in {0,.5,1,2,3,4,6}.  (R) candidates = topK(A1) U topK(EP_emb) (context items excluded),
local-z rerank A1 + g EP_emb ; (R0) topK(A1) only. K in {50,100,200}. VALID NDCG@10 select, TEST once; user bootstrap; paired vs UniCRS."""
import os, json, numpy as np, torch
os.environ.setdefault("SPACES", "emb,exp,attr")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fullcat_episodic.py")).read()
src = src.split("GG = [0, 0.5, 1.0, 1.5, 2.0]")[0].replace("TAUS = [0.02, 0.05]", "TAUS = [0.01, 0.02, 0.05, 0.1]")
exec(src)
@torch.no_grad()
def score_ndcg(sp, fn):
    A1, EP, G = FE[sp]; out = []
    for b in range(0, A1.shape[0], 512):
        Sb = fn(A1[b:b + 512].float(), {k: x[b:b + 512] for k, x in EP.items()})
        top = torch.topk(Sb, 10, dim=1).indices; Gb = G[b:b + 512]; rel = Gb.gather(1, top).float(); ng = Gb.sum(1).float()
        idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]; v = (rel * DISC).sum(1) / idcg; v[ng == 0] = float("nan"); out.append(v.cpu())
    return torch.cat(out).numpy()
def add_fn(terms):
    def f(A, EP):
        S = A.clone()
        for (s, t, g) in terms:
            if g: S += g * EP[(s, t)].float()
        return S
    return f
def rr_fn(K, union, t, g):
    def f(A, EP):
        E1 = EP[("emb", t)].float(); valid = A > -1e3
        cand = torch.zeros_like(A, dtype=torch.bool); cand.scatter_(1, torch.topk(A, K, 1).indices, True)
        if union: cand.scatter_(1, torch.topk(torch.where(valid, E1, torch.full_like(E1, -1e9)), K, 1).indices, True)
        cand &= valid; n = cand.sum(1, keepdim=True).float()
        def zl(X):
            mu = torch.where(cand, X, 0 * X).sum(1, keepdim=True) / n
            sd = (torch.where(cand, (X - mu) ** 2, 0 * X).sum(1, keepdim=True) / n).sqrt() + 1e-9; return (X - mu) / sd
        return torch.where(cand, zl(A) + g * zl(E1) + 1e4, A)
    return f
GG = [0.5, 1, 2, 3, 4, 6]; RES, ROW = {}, {}
def pick(name, cands, mk):
    vals = {c: float(np.nanmean(score_ndcg("valid", mk(c)))) for c in cands}; best = max(vals, key=vals.get)
    ROW[name] = score_ndcg("test", mk(best)); RES[name] = dict(params=list(best), valid=round(vals[best], 4), test=round(float(np.nanmean(ROW[name])), 4)); log(name, RES[name])
pick("A1", [()], lambda c: add_fn([]))
pick("T:A1+EP_emb", [(t, g) for t in TAUS for g in GG], lambda c: add_fn([("emb", *c)]))
for s in ("exp", "attr"):
    pick(f"T:A1+EP_emb+EP_{s}", [(t1, a, t2, b) for t1 in TAUS for a in GG for t2 in TAUS for b in [0] + GG],
         lambda c, s=s: add_fn([("emb", c[0], c[1]), (s, c[2], c[3])]))
for K in (50, 100, 200):
    pick(f"R0:topK(A1) K{K}", [(t, g) for t in TAUS for g in GG], lambda c, K=K: rr_fn(K, False, *c))
    pick(f"R:union K{K}", [(t, g) for t in TAUS for g in GG], lambda c, K=K: rr_fn(K, True, *c))
users = SPLIT["test"]["user"]; uu, inv = np.unique(users, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    ok = ~np.isnan(d); Sd = np.bincount(inv[ok], d[ok], minlength=len(uu)); Cn = np.bincount(inv[ok], minlength=len(uu))
    est = [Sd[k].sum() / max(Cn[k].sum(), 1) for k in BI]
    return [round(float(d[ok].mean()), 4), round(float(np.percentile(est, 2.5)), 4), round(float(np.percentile(est, 97.5)), 4)]
UC = {}
for s in range(3):
    for l in open(f"{W}/results/reddit_crs_unicrs_s{s}_test_rows.jsonl"):
        x = json.loads(l); UC.setdefault(x["req"], []).append(x["m"][1])
uni = np.array([np.mean(UC[x["req"]]) for x in SPLIT["test"]["L"]])
OUT = {"UniCRS_same_rows": round(float(np.nanmean(uni[~np.isnan(ROW["A1"])])), 4), "systems": RES,
       "vs_UniCRS": {k: boot(r - uni) for k, r in ROW.items()}, "vs_A1": {k: boot(r - ROW["A1"]) for k, r in ROW.items() if k != "A1"}}
for K in (50, 100, 200): OUT[f"union - topK(A1) K{K}"] = boot(ROW[f"R:union K{K}"] - ROW[f"R0:topK(A1) K{K}"])
json.dump(OUT, open(f"{W}/results/fullcat_reddit_v2.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("V2_DONE")
