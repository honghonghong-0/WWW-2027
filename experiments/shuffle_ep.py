import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""[base copied from shrink_ep.py] Bayesian shrinkage of relational popularity toward the popularity prior (fixed before running, 2026-11-17).
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

# ---------------- SHUFFLE CONTROL (fixed before running, 2026-11-17) ----------------
# Is the gain from the request-episode RELATION or just from popularity re-weighting? Control: permute which request each training
# episode carries (rows of the training-request matrix), keeping every item's episode count n_i and the item sets intact. The kernel
# then sums similarities to RANDOM requests, so RP_shuf ~ log n_i + noise. Same grid (tau x g) selected on VALID NDCG@10, TEST once.
# Primary: TEST (A1 + RP_real) - (A1 + RP_shuf), paired bootstrap (users on Reddit, conversations on ReDial).
TAUS = [0.01, 0.02, 0.05, 0.1]; GG = [0.5, 1, 2, 3, 4, 6]
DISC = (1.0 / torch.log2(torch.arange(2, 12).float())).to(D)
perm = torch.as_tensor(np.random.default_rng(0).permutation(QTk.shape[0]), device=QTk.device)
KEYS = {"real": QTk, "shuf": QTk[perm]}
@torch.no_grad()
def build(sp):
    N = NQ[sp]; A1 = torch.empty(N, NI, dtype=torch.float16, device=D); GM = torch.zeros(N, NI, dtype=torch.bool, device=D)
    EP = {(k, t): torch.empty(N, NI, dtype=torch.float16, device=D) for k in KEYS for t in TAUS}
    for b in range(0, N, 128):
        rows = list(range(b, min(N, b + 128))); q = dev(QS[sp][rows]).half()
        A1[rows] = a1_rows(sp, rows).half(); GM[rows] = gold_mask(sp, rows)
        for k, K_ in KEYS.items():
            sm = (q @ K_.T).float(); mx = sm.max(1, keepdim=True).values
            for t in TAUS:
                L = torch.log((M @ torch.exp((sm - mx) / t).half().T).T.float() + 1e-30)
                mn = torch.where(has_ep[None], L, torch.full_like(L, 1e30)).min(1, keepdim=True).values
                EP[(k, t)][rows] = rzt(torch.where(has_ep[None], L, mn)).half()
    return A1, EP, GM
FE = {sp: build(sp) for sp in ("valid", "test")}; log("built")
@torch.no_grad()
def ndcg(sp, k, t, g):
    A1, EP, GM = FE[sp]; out = []
    for b in range(0, A1.shape[0], 512):
        Sb = A1[b:b + 512].float() + (g * EP[(k, t)][b:b + 512].float() if g else 0)
        top = torch.topk(Sb, 10, 1).indices; Gb = GM[b:b + 512]; rel = Gb.gather(1, top).float(); ng = Gb.sum(1).float()
        idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]; v = (rel * DISC).sum(1) / idcg; v[ng == 0] = float("nan"); out.append(v.cpu())
    return torch.cat(out).numpy()
ROW = {"A1": ndcg("test", "real", TAUS[0], 0)}; RES = {"A1": round(float(np.nanmean(ROW["A1"])), 4)}
for k in KEYS:
    vals = {(t, g): float(np.nanmean(ndcg("valid", k, t, g))) for t in TAUS for g in GG}; best = max(vals, key=vals.get)
    ROW[k] = ndcg("test", k, *best); RES[k] = dict(params=list(best), valid=round(vals[best], 4), test=round(float(np.nanmean(ROW[k])), 4)); log(k, RES[k])
uu, inv = np.unique(CL, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    ok = ~np.isnan(d); Sd = np.bincount(inv[ok], d[ok], minlength=len(uu)); Cn = np.bincount(inv[ok], minlength=len(uu))
    e = [Sd[j].sum() / max(Cn[j].sum(), 1) for j in BI]; return [round(float(d[ok].mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4)]
OUT = {"systems": RES, "PRIMARY real - shuf": boot(ROW["real"] - ROW["shuf"]), "real - A1": boot(ROW["real"] - ROW["A1"]), "shuf - A1": boot(ROW["shuf"] - ROW["A1"])}
json.dump(OUT, open(f"{W}/results/shuffle_ep_{DS}.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("SHUF_DONE")
