import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Request-kNN voting baseline vs relational popularity (reviewer request; fixed before running, 2026-11-17).
kNN: retrieve the top-k most similar TRAINING requests, score each item by the number of those neighbours that chose it (count)
or by the sum of their cosines (sim). k in {10,20,50,100,200}. Reported (a) alone (ties broken by popularity) and (b) added to A1
as A1 + g z(log(1+score)), g in {.5,1,2,3,4,6}. RP (ours, additive form): A1 + g z(RP_tau), tau in {.01,.02,.05,.1}.
All settings chosen on VALID NDCG@10, TEST once, paired bootstrap (users on Reddit, conversations on ReDial).
usage: python knn_baseline.py reddit|redial"""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{REPO}/coground/shrink_ep.py").read().split("cnt = M.float().sum(1)")[0])
cntv = M.float().sum(1); lpop = torch.log1p(cntv)
KS = [10, 20, 50, 100, 200]; GG = [0.5, 1, 2, 3, 4, 6]; TAUS = [0.01, 0.02, 0.05, 0.1]
DISC = (1.0 / torch.log2(torch.arange(2, 12).float())).to(D)
def ndcg_rows(Sc, GM):
    top = torch.topk(Sc, 10, 1).indices; rel = GM.gather(1, top).float(); ng = GM.sum(1).float()
    idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]; v = (rel * DISC).sum(1) / idcg; v[ng == 0] = float("nan"); return v
CONF = [("knn_alone", k, w, 0) for k in KS for w in ("count", "sim")] + [("A1+knn", k, w, g) for k in KS for w in ("count", "sim") for g in GG] + \
       [("A1+RP", t, None, g) for t in TAUS for g in GG] + [("A1", 0, None, 0)]
@torch.no_grad()
def run(sp, confs):
    N = NQ[sp]; out = {c: [] for c in confs}
    for b in range(0, N, 128):
        rows = list(range(b, min(N, b + 128))); q = dev(QS[sp][rows]).half()
        A1 = a1_rows(sp, rows).float(); GM = gold_mask(sp, rows); masked = A1 < -1e3
        sm = (q @ QTk.T).float(); mx = sm.max(1, keepdim=True).values
        cache = {}
        for c in confs:
            kind, a, w, g = c
            if kind == "A1": Sc = A1
            elif kind == "A1+RP":
                if ("rp", a) not in cache:
                    L = torch.log((M @ torch.exp((sm - mx) / a).half().T).T.float() + 1e-30)
                    mn = torch.where(has_ep[None], L, torch.full_like(L, 1e30)).min(1, keepdim=True).values
                    cache[("rp", a)] = rzt(torch.where(has_ep[None], L, mn))
                Sc = A1 + g * cache[("rp", a)]
            else:
                if ("knn", a, w) not in cache:
                    tv, ti = torch.topk(sm, a, 1); Wt = torch.zeros_like(sm); Wt.scatter_(1, ti, torch.ones_like(tv) if w == "count" else tv.clamp(min=0))
                    cache[("knn", a, w)] = (M.float() @ Wt.T).T                      # R x NI
                K_ = cache[("knn", a, w)]
                if kind == "knn_alone": Sc = K_ + 1e-6 * lpop[None]; Sc = torch.where(masked, torch.full_like(Sc, -1e9), Sc)
                else: Sc = A1 + g * rzt(torch.log1p(K_))
            out[c].append(ndcg_rows(Sc, GM).cpu())
    return {c: torch.cat(v).numpy() for c, v in out.items()}
V = run("valid", CONF); vm = {c: float(np.nanmean(v)) for c, v in V.items()}
best = {kind: max([c for c in CONF if c[0] == kind], key=vm.get) for kind in ("knn_alone", "A1+knn", "A1+RP", "A1")}
T = run("test", list(best.values()))
uu, inv = np.unique(CL, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    ok = ~np.isnan(d); Sd = np.bincount(inv[ok], d[ok], minlength=len(uu)); Cn = np.bincount(inv[ok], minlength=len(uu))
    e = [Sd[j].sum() / max(Cn[j].sum(), 1) for j in BI]; return [round(float(d[ok].mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4)]
OUT = {"selected": {k: list(map(str, c)) for k, c in best.items()}, "valid": {k: round(vm[c], 4) for k, c in best.items()},
       "test": {k: round(float(np.nanmean(T[c])), 4) for k, c in best.items()},
       "A1+RP - A1+knn": boot(T[best["A1+RP"]] - T[best["A1+knn"]]), "A1+knn - A1": boot(T[best["A1+knn"]] - T[best["A1"]])}
json.dump(OUT, open(f"{W}/results/knn_baseline_{DS}.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("KNN_DONE")
