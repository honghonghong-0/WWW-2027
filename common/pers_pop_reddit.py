import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""(1) Popularity analysis of the main system and (2) personal affordance inside the relational structure. Reddit. Fixed before running 2026-11-17.
Main system (fixed, valid-selected in fullcat_reddit_v2): candidates = top100(A1) U top100(EP_emb, tau .02), local-z rerank A1 + 3 z(EP).
(1) analysis only (no selection): per-request NDCG by gold-popularity tercile (median TRAIN count of the request's gold), and list metrics of
    top-10 (mean log1p TRAIN count, catalogue coverage, long-tail share = items in the bottom popularity tercile) for A1 / main / UniCRS.
(2) Personal affordance = the user's OWN past episodes in the same kernel mixture (hierarchical: a personal episode counts lam community ones):
      p(i|C,u) ∝ sum_{c' in N(i)} k(C,c') + lam * sum_{h in hist(u)} k(C,c_h) 1[i in O_h],   k = exp((sim-1)/tau)
    hist(u) = the user's PAST threads (before the request; the dataset's own history field), O_h = items recommended in h ("rec") [def A]
    or rec + items the user named in their own follow-ups ("fb") [def B]. lam in {0,1,3,10,30,100}; same rerank, K, tau, g as main.
    VALID NDCG@10 selects (def, lam); TEST once. Primary: TEST (best lam>0) - (lam=0) on WARM requests, user bootstrap.
    Context items stay masked (as everywhere)."""
import os, json, numpy as np, torch
os.environ["SPACES"] = "emb"
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{REPO}/coground/fullcat_episodic.py").read(); exec(src.split("TAUS = [0.02, 0.05]")[0])
TAU, G3, K = 0.02, 3.0, 100
cnt = M.float().sum(1).cpu().numpy()
# own-episode embeddings: past threads of the requester
def emb_of(h): return ET[T[h]] if h in T else None
def own(sp, defn):
    out = []
    for x in SPLIT[sp]["L"]:
        hs = []
        for h in x["hist"]:
            e = emb_of(h); th = TH.get(h)
            if e is None or th is None: continue
            it = list(th["rec"]) + ([i for f in th["fb"] for i in f["items"]] if defn == "B" else [])
            it = sorted({iidx[j] for j in it if j in iidx})
            if it: hs.append((e, it))
        out.append(hs)
    return out
OWN = {(sp, d): own(sp, d) for sp in ("valid", "test") for d in ("A", "B")}
for sp in ("valid", "test"):
    w = SPLIT[sp]["warm"]; log(sp, "warm", int(w.sum()), "with usable own episodes A/B", sum(bool(o) for o in OWN[(sp, "A")]), sum(bool(o) for o in OWN[(sp, "B")]))
LAMS = [0, 1, 3, 10, 30, 100]
DISC = (1.0 / torch.log2(torch.arange(2, 12).float())).to(D)
@torch.no_grad()
def run(sp, defn, lam, want_top=False):
    S = SPLIT[sp]; N = len(S["L"]); nds, tops = [], []
    for b in range(0, N, 128):
        rows = list(range(b, min(N, b + 128))); q = dev(S["Q"][rows])
        A = rzt(q @ Vt.T) + 0.75 * ZPd[None] + 0.2 * rzt(dev(PM["exp"][sp][rows]) @ Yt.T)
        valid = torch.ones_like(A, dtype=torch.bool)
        for k, n in enumerate(rows):
            if S["ctx"][n]: A[k, S["ctx"][n]] = -1e4; valid[k, S["ctx"][n]] = False
        sm = (q.half() @ QT.half().T).float(); mx = sm.max(1, keepdim=True).values
        L = torch.log((M @ torch.exp((sm - mx) / TAU).half().T).T.float() + 1e-30) + (mx - 1) / TAU
        L = torch.where(has_ep[None], L, torch.full_like(L, -1e30))
        if lam:
            for k, n in enumerate(rows):
                for e, it in OWN[(sp, defn)][n]:
                    s = float(np.dot(S["Q"][n], e)); lw = np.log(lam) + (s - 1) / TAU
                    idx_ = torch.as_tensor(it, device=D); L[k, idx_] = torch.logaddexp(L[k, idx_], torch.full((len(it),), lw, device=D))
        hasL = L > -1e29; mn = torch.where(hasL, L, torch.full_like(L, 1e30)).min(1, keepdim=True).values
        E = rzt(torch.where(hasL, L, mn))
        cand = torch.zeros_like(A, dtype=torch.bool); cand.scatter_(1, torch.topk(A, K, 1).indices, True)
        cand.scatter_(1, torch.topk(torch.where(valid, E, torch.full_like(E, -1e9)), K, 1).indices, True); cand &= valid
        n_ = cand.sum(1, keepdim=True).float()
        def zl(X):
            mu = torch.where(cand, X, 0 * X).sum(1, keepdim=True) / n_
            return (X - mu) / ((torch.where(cand, (X - mu) ** 2, 0 * X).sum(1, keepdim=True) / n_).sqrt() + 1e-9)
        Sc = torch.where(cand, zl(A) + G3 * zl(E) + 1e4, A)
        top = torch.topk(Sc, 10, 1).indices
        Gb = torch.zeros_like(A, dtype=torch.bool)
        for k, n in enumerate(rows): Gb[k, S["gold"][n]] = True
        rel = Gb.gather(1, top).float(); ng = Gb.sum(1).float(); idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]
        v = (rel * DISC).sum(1) / idcg; v[ng == 0] = float("nan"); nds.append(v.cpu())
        if want_top: tops.append(top.cpu())
    return torch.cat(nds).numpy(), (torch.cat(tops).numpy() if want_top else None)
@torch.no_grad()
def run_a1(sp):
    S = SPLIT[sp]; N = len(S["L"]); tops, nds = [], []
    for b in range(0, N, 256):
        rows = list(range(b, min(N, b + 256))); q = dev(S["Q"][rows])
        A = rzt(q @ Vt.T) + 0.75 * ZPd[None] + 0.2 * rzt(dev(PM["exp"][sp][rows]) @ Yt.T)
        for k, n in enumerate(rows):
            if S["ctx"][n]: A[k, S["ctx"][n]] = -1e4
        tops.append(torch.topk(A, 10, 1).indices.cpu())
    return torch.cat(tops).numpy()
users = SPLIT["test"]["user"]; warm = SPLIT["test"]["warm"]
def boot(d, mask):
    ok = mask & ~np.isnan(d); uu, inv = np.unique(users[ok], return_inverse=True); dd = d[ok]
    Sd = np.bincount(inv, dd, minlength=len(uu)); Cn = np.bincount(inv, minlength=len(uu)); rb = np.random.default_rng(1)
    e = [Sd[k].sum() / Cn[k].sum() for k in (rb.integers(0, len(uu), len(uu)) for _ in range(2000))]
    return [round(float(dd.mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4), int(ok.sum())]
OUT = {}
# ---------- (2) personal
vals = {}
for d in ("A", "B"):
    for lam in LAMS:
        if lam == 0 and d == "B": continue
        vv, _ = run("valid", d, lam); vals[(d, lam)] = (float(np.nanmean(vv)), float(np.nanmean(vv[SPLIT["valid"]["warm"]])))
        log("valid", d, lam, [round(x, 4) for x in vals[(d, lam)]])
best = max([k for k in vals if k[1] > 0], key=lambda k: vals[k][0])
r0, top0 = run("test", "A", 0, True); r1, _ = run("test", *best)
OUT["personal"] = {"valid": {f"{k[0]}:{k[1]}": [round(x, 4) for x in v] for k, v in vals.items()}, "selected": list(best),
    "test_all": {"lam0": round(float(np.nanmean(r0)), 4), "best": round(float(np.nanmean(r1)), 4)},
    "test_warm": {"lam0": round(float(np.nanmean(r0[warm])), 4), "best": round(float(np.nanmean(r1[warm])), 4)},
    "PRIMARY warm best - lam0": boot(r1 - r0, warm), "all best - lam0": boot(r1 - r0, np.ones_like(warm))}
# ---------- (1) popularity analysis (main = lam 0)
iid_list = [None] * NI
for k_, v_ in iidx.items(): iid_list[v_] = k_
UT = {}
posd = {x["req"]: n for n, x in enumerate(SPLIT["test"]["L"])}
for sd in range(3):
    for l in open(f"{W}/results/reddit_crs_unicrs_s{sd}_test_rows.jsonl"):
        x = json.loads(l); cx = set(SPLIT["test"]["ctx"][posd[x["req"]]])
        UT.setdefault(x["req"], []).append([iidx[i] for i in x["top100"] if i in iidx and iidx[i] not in cx][:10])
topA1 = run_a1("test"); reqs = [x["req"] for x in SPLIT["test"]["L"]]
t1, t2 = np.quantile(cnt[cnt > 0], [1 / 3, 2 / 3])
tail = cnt <= t1
def listm(tops):
    T_ = np.array(tops); return dict(mean_log_pop=round(float(np.log1p(cnt[T_]).mean()), 3), coverage=int(len(np.unique(T_))), tail_share=round(float(tail[T_].mean()), 4))
OUT["popularity_lists"] = {"A1": listm(topA1), "main": listm(top0), "UniCRS_s0": listm([UT[r][0] for r in reqs]),
                           "tercile_cuts_train_count": [float(t1), float(t2)]}
gpop = np.array([np.median(cnt[g]) if g else np.nan for g in SPLIT["test"]["gold"]])
R = np.load(f"{W}/results/fullcat_base_rows.npz"); a1nd, uni = R["A1"], R["uni"]
terc = np.where(gpop <= t1, "tail", np.where(gpop <= t2, "mid", "head"))
OUT["by_gold_popularity"] = {}
for tname in ("tail", "mid", "head"):
    m = terc == tname
    OUT["by_gold_popularity"][tname] = dict(n=int((m & ~np.isnan(r0)).sum()), A1=round(float(np.nanmean(a1nd[m])), 4), main=round(float(np.nanmean(r0[m])), 4),
        UniCRS=round(float(np.nanmean(uni[m])), 4), main_minus_A1=boot(r0 - a1nd, m), main_minus_UniCRS=boot(r0 - uni, m))
json.dump(OUT, open(f"{W}/results/pers_pop_reddit.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("PP_DONE")
