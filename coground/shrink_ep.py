import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Bayesian shrinkage of relational popularity toward the popularity prior (fixed before running, 2026-11-17).
Kernel mixture with a Dirichlet pseudo-count:
   p(i|C) ∝ sum_{c' in N(i)} k(C,c') + alpha * pi(i),   k(C,c') = exp((sim(C,c') - 1)/tau)   (k = 1 for an identical past request)
   pi(i) = (n_i + 1/2) / sum_j (n_j + 1/2)   (Jeffreys-smoothed popularity)
   EP_alpha(i,C) = logaddexp( log sum k , log alpha + log pi(i) )
alpha is a prior mass in units of "identical past episodes": when the current request has few similar episodes the kernel sum is ~0 and
EP_alpha -> log alpha + log pi(i) (plain popularity); with many similar episodes the evidence dominates. Items without any TRAIN episode get
the prior instead of the ad-hoc row minimum. alpha = 0 reproduces the unshrunk EP (row-min fill).
Score = A1 + g z(EP_alpha_emb); tau in {.01,.02,.05,.1}, g in {.5,1,2,3,4,6}, alpha in {0,1e-3,1e-2,1e-1,1,10}; VALID NDCG@10, TEST once.
Primary: TEST (best shrunk, alpha>0) - (best unshrunk, alpha=0) paired CI. Also the gain by max-similarity quartile (where v1 lost on ReDial).
usage: python shrink_ep.py reddit|redial"""
import os, sys, json, numpy as np, torch
DS = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
if DS == "redial":
    exec(open(f"{REPO}/coground/fullcat_redial.py").read().split("FE = {sp: comps(sp)")[0])
    def conv_ids(L):
        out, c, prev = [], -1, None
        for x in L:
            k = x["ctx"]
            if prev is None or len(k) < len(prev) or k[:len(prev)] != prev: c += 1
            out.append(c); prev = k
        return np.array(out)
    CL = conv_ids(S["test"]); NQ = {sp: len(S[sp]) for sp in ("valid", "test")}
    QS = {sp: Q[sp] for sp in ("valid", "test")}
    def a1_rows(sp, rows):
        q = dev(QS[sp][rows]); return rzt(q @ Vt.T) + 0.75 * ZP[None] + 0.2 * rzt(dev(P[sp][rows]) @ Yt.T)
    def gold_mask(sp, rows):
        g = torch.zeros(len(rows), NI, dtype=torch.bool, device=D); gi = gold[sp][rows]; ok = gi >= 0
        g[torch.as_tensor(np.where(ok)[0], device=D), torch.as_tensor(gi[ok], device=D)] = True; return g
    QTk = QT; UNI = uni_nd
else:
    os.environ["SPACES"] = "emb"
    src = open(f"{REPO}/coground/fullcat_episodic.py").read(); exec(src.split("TAUS = [0.02, 0.05]")[0])
    NQ = {sp: len(SPLIT[sp]["L"]) for sp in ("valid", "test")}; QS = {sp: SPLIT[sp]["Q"] for sp in ("valid", "test")}
    def a1_rows(sp, rows):
        q = dev(QS[sp][rows]); Sc = rzt(q @ Vt.T) + 0.75 * ZPd[None] + 0.2 * rzt(dev(PM["exp"][sp][rows]) @ Yt.T)
        for k, n in enumerate(rows):
            if SPLIT[sp]["ctx"][n]: Sc[k, SPLIT[sp]["ctx"][n]] = -1e4
        return Sc
    def gold_mask(sp, rows):
        g = torch.zeros(len(rows), NI, dtype=torch.bool, device=D)
        for k, n in enumerate(rows): g[k, SPLIT[sp]["gold"][n]] = True
        return g
    QTk = QT.half() if QT.dtype != torch.float16 else QT
    UC = {}
    for s in range(3):
        for l in open(f"{W}/results/reddit_crs_unicrs_s{s}_test_rows.jsonl"):
            x = json.loads(l); UC.setdefault(x["req"], []).append(x["m"][1])
    UNI = np.array([np.mean(UC[x["req"]]) for x in SPLIT["test"]["L"]]); CL = SPLIT["test"]["user"]
cnt = M.float().sum(1); logpi = torch.log((cnt + 0.5) / (cnt + 0.5).sum())
TAUS = [0.01, 0.02, 0.05, 0.1]; ALPHAS = [0, 1e-3, 1e-2, 1e-1, 1, 10]; GG = [0.5, 1, 2, 3, 4, 6]
DISC = (1.0 / torch.log2(torch.arange(2, 12).float())).to(D)
MAXS = {}
@torch.no_grad()
def build(sp):
    N = NQ[sp]; A1 = torch.empty(N, NI, dtype=torch.float16, device=D); EP = {(t, a): torch.empty(N, NI, dtype=torch.float16, device=D) for t in TAUS for a in ALPHAS}
    GM = torch.zeros(N, NI, dtype=torch.bool, device=D); mxs = []
    for b in range(0, N, 128):
        rows = list(range(b, min(N, b + 128))); q = dev(QS[sp][rows]).half()
        A1[rows] = a1_rows(sp, rows).half(); GM[rows] = gold_mask(sp, rows)
        sm = (q @ QTk.T).float(); mx = sm.max(1, keepdim=True).values; mxs.append(mx[:, 0].cpu())
        for t in TAUS:
            L = torch.log((M @ torch.exp((sm - mx) / t).half().T).T.float() + 1e-30) + (mx - 1) / t   # log sum_c' exp((s-1)/tau)
            L = torch.where(has_ep[None], L, torch.full_like(L, -1e30))
            for a in ALPHAS:
                if a == 0:
                    mn = torch.where(has_ep[None], L, torch.full_like(L, 1e30)).min(1, keepdim=True).values; e = torch.where(has_ep[None], L, mn)
                else: e = torch.logaddexp(L, np.log(a) + logpi[None])
                EP[(t, a)][rows] = rzt(e).half()
    MAXS[sp] = torch.cat(mxs).numpy(); return A1, EP, GM
FE = {sp: build(sp) for sp in ("valid", "test")}; log("built")
@torch.no_grad()
def ndcg(sp, t, a, g):
    A1, EP, GM = FE[sp]; out = []
    for b in range(0, A1.shape[0], 512):
        Sb = A1[b:b + 512].float() + (g * EP[(t, a)][b:b + 512].float() if g else 0)
        top = torch.topk(Sb, 10, 1).indices; Gb = GM[b:b + 512]; rel = Gb.gather(1, top).float(); ng = Gb.sum(1).float()
        idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]; v = (rel * DISC).sum(1) / idcg; v[ng == 0] = float("nan"); out.append(v.cpu())
    return torch.cat(out).numpy()
RES, ROW = {}, {}
ROW["A1"] = ndcg("test", TAUS[0], 0, 0); RES["A1"] = dict(test=round(float(np.nanmean(ROW["A1"])), 4))
grid_val = {(t, a, g): float(np.nanmean(ndcg("valid", t, a, g))) for t in TAUS for a in ALPHAS for g in GG}
for name, cond in (("unshrunk", lambda a: a == 0), ("shrunk", lambda a: a > 0)):
    best = max([k for k in grid_val if cond(k[1])], key=grid_val.get); ROW[name] = ndcg("test", *best)
    RES[name] = dict(params=list(best), valid=round(grid_val[best], 4), test=round(float(np.nanmean(ROW[name])), 4)); log(name, RES[name])
best_per_alpha = {a: max([k for k in grid_val if k[1] == a], key=grid_val.get) for a in ALPHAS}
RES["valid_by_alpha"] = {str(a): [list(best_per_alpha[a]), round(grid_val[best_per_alpha[a]], 4)] for a in ALPHAS}
uu, inv = np.unique(CL, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    ok = ~np.isnan(d); Sd = np.bincount(inv[ok], d[ok], minlength=len(uu)); Cn = np.bincount(inv[ok], minlength=len(uu))
    e = [Sd[k].sum() / max(Cn[k].sum(), 1) for k in BI]; return [round(float(d[ok].mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4)]
OUT = {"systems": RES, "UniCRS": round(float(np.nanmean(UNI[~np.isnan(ROW["A1"])])), 4),
       "PRIMARY shrunk - unshrunk": boot(ROW["shrunk"] - ROW["unshrunk"]),
       "shrunk - UniCRS": boot(ROW["shrunk"] - UNI), "unshrunk - UniCRS": boot(ROW["unshrunk"] - UNI)}
mx = MAXS["test"]; ok = ~np.isnan(ROW["A1"]); qs = np.quantile(mx[ok], [0, .25, .5, .75, 1])
OUT["by_maxsim_quartile"] = [dict(lo=round(float(qs[i]), 3), unshrunk_minus_A1=round(float((ROW["unshrunk"] - ROW["A1"])[ok & (mx >= qs[i]) & (mx <= qs[i + 1])].mean()), 4),
                                  shrunk_minus_A1=round(float((ROW["shrunk"] - ROW["A1"])[ok & (mx >= qs[i]) & (mx <= qs[i + 1])].mean()), 4)) for i in range(4)]
json.dump(OUT, open(f"{W}/results/shrink_ep_{DS}.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("SHRINK_DONE")
