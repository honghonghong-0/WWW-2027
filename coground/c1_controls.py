"""c1 / control experiments for the affordance term c1 on Reddit-Movie large. CPU only.
Reads existing embeddings / prep files; writes ONLY under WWW-2027/c1/.

Question: is the gain of c1 due to the EXPERIENCE axes, or simply due to supervision on TRAIN golds?

Same ladder and evaluator as scripts/reddit_ladder_test.py:
    s_i = rz(base_i) + g_pop * z(pop_i) + g1 * rz(c_i),      c_i = p_C . f_i
    p_C = ridge(request emb -> mean item feature f of the request's golds), fit on TRAIN, lam=10,
          centered by the split mean (as in the original script)
    gammas selected once on the full VALID split (NDCG@10, all requests), applied to TEST once.
Only the item feature f changes between variants.

Experiment 2 (which axes?)  -- genome-based features, uncovered items ridge-filled from title emb (as original):
    exp69          69 Tag Genome experience tags                         (= the paper's c1; must reproduce 0.1563)
    attr447        447 Tag Genome property tags
    all516         experience + property tags
    attr_rand69_s* 69 random property tags (5 seeds)
    attr_pca69     top-69 principal components of the property tags
    exp69_perm     experience profiles shuffled across genome movies (placebo: axes kept, item link destroyed)
Experiment 1 (no experience bottleneck) -- no genome at all, every item covered:
    direct_emb     f = title embedding (4096-d): request emb -> mean gold title emb   (lam 10 / 100 / 1000)
    emb_pca69      top-69 principal components of the title embeddings
    emb_rand69     69-d random projection of the title embeddings
Combinations (g_pop fixed at 0.75): does exp69 still add on top of a control term?
"""
import json, os, time, numpy as np, torch
torch.set_num_threads(int(os.environ.get("NT", 16)))
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
OUTD = f"{W}/c1/results"; os.makedirs(OUTD, exist_ok=True)
PILOT = int(os.environ.get("PILOT", 0)); SUF = "_pilot" if PILOT else ""
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

# ---------------- data (same as reddit_ladder_test.py) ----------------
Z = np.load(f"{W}/results/reddit_large_emb.npz")
items = list(Z["items"]); iidx = {m: k for k, m in enumerate(items)}; NI = len(items)
V = Z["V"].astype(np.float32); T = dict(zip(Z["tids"], range(len(Z["tids"])))); ET = Z["T"].astype(np.float32)
TH = json.load(open(f"{P}/threads.json")); IJ = json.load(open(f"{P}/items.json"))
U = {sp: [json.loads(l) for l in open(f"{P}/requests_{sp}.jsonl")] for sp in ("train", "valid", "test")}
if PILOT: U["valid"] = U["valid"][:PILOT]; U["test"] = U["test"][:PILOT]
G = np.load("/path/to/base/projects/crs/processed/item_chain/item_chain_v1.npz")
grow = {int(m): r for r, m in enumerate(G["items"])}
has = np.array([(m is not None and int(m) in grow) for m in IJ["ml"]]); idx = np.where(has)[0]
gr = np.array([grow[int(IJ["ml"][k])] for k in idx])
GY, GX = G["Y"].astype(np.float32), G["X"].astype(np.float32)
log(f"items {NI} genome-covered {has.sum()} | train {len(U['train'])} valid {len(U['valid'])} test {len(U['test'])}")

def tt(a): return torch.as_tensor(np.ascontiguousarray(a))
def mm(a, b): return (tt(a) @ tt(b)).numpy()          # torch matmul (numpy BLAS is very slow on this machine)
def mk_solver(X, lam):
    Xb = tt(np.hstack([X, np.ones((len(X), 1), np.float32)])); I = torch.eye(Xb.shape[1], dtype=torch.float64); I[-1, -1] = 0
    A = (Xb.T @ Xb).double() + lam * I
    return lambda Yt: torch.linalg.solve(A, (Xb.T @ tt(Yt).float()).double()).float().numpy()
pred = lambda Wt, Xa: mm(np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]), Wt)

POP = np.zeros(NI)
for x in U["train"]:
    for g in x["gold"]: POP[iidx[g]] += 1
POP = np.log1p(POP); ZP = torch.tensor((POP - POP.mean()) / POP.std(), dtype=torch.float32)

# train rows: 'cov' = requests with >=1 genome-covered gold (original c1 fit rows); 'all' = every request with a gold
gl_cov = [[iidx[g] for g in x["gold"] if has[iidx[g]]] for x in U["train"]]
gl_all = [[iidx[g] for g in x["gold"]] for x in U["train"]]
Qtrain = np.stack([ET[T[x["req"]]] for x in U["train"]])
ROWSETS = {"cov": [n for n, l in enumerate(gl_cov) if l], "all": [n for n, l in enumerate(gl_all) if l]}
GL = {"cov": gl_cov, "all": gl_all}
SOLVERS = {}
def reader(rowset, Fraw, lam=10.0):
    k = (rowset, lam)
    if k not in SOLVERS: SOLVERS[k] = mk_solver(Qtrain[ROWSETS[rowset]], lam)
    Tt = np.stack([Fraw[GL[rowset][n]].mean(0) for n in ROWSETS[rowset]])
    return SOLVERS[k](Tt)
fill_solver = mk_solver(V[idx], 10.0)
def genome_feats(Fg):
    """Fg (n_genome x d) -> raw (NI x d, zeros where uncovered), filled+centered (NI x d); same procedure as Yf in the original"""
    Fm = np.zeros((NI, Fg.shape[1]), np.float32); Fm[idx] = Fg[gr]; fbar = Fm[idx].mean(0)
    Wy = fill_solver(Fm[idx] - fbar)
    return Fm, np.where(has[:, None], Fm - fbar, pred(Wy, V)).astype(np.float32)

SPLIT = {}
for sp in ("valid", "test"):
    L = U[sp]; Q = np.stack([ET[T[x["req"]]] for x in L])
    ctx = [[iidx[c] for c in x["ctx_items"] if c in iidx] for x in L]; gold = [[iidx[g] for g in x["gold"]] for x in L]
    hs = [sorted({iidx[j] for h in x["hist"] for j in TH[h]["items"] + TH[h]["rec"] + [i for f in TH[h]["fb"] for i in f["items"]] if j in iidx}) for x in L]
    SPLIT[sp] = dict(L=L, Q=Q, ctx=ctx, gold=gold, hs=hs, warm=np.array([bool(x["hist"]) for x in L]), user=np.array([x["user"] for x in L]))

# ---------------- variants ----------------
FF, PM, INFO = {}, {}, {}
def add(name, Fraw, Ff, rowset, lam=10.0):
    Gw = reader(rowset, Fraw, lam); PM[name] = {}
    for sp in ("valid", "test"):
        Pm = pred(Gw, SPLIT[sp]["Q"]).astype(np.float32); Pm -= Pm.mean(0); PM[name][sp] = Pm
    FF[name] = torch.tensor(Ff); INFO[name] = dict(dim=int(Ff.shape[1]), fit_rows=len(ROWSETS[rowset]), rowset=rowset, lam=lam)
    log("built", name, INFO[name])
rng = np.random.default_rng(0)
add("exp69", *genome_feats(GY), "cov")
add("attr447", *genome_feats(GX), "cov")
add("all516", *genome_feats(np.hstack([GY, GX])), "cov")
for s in range(5):
    cols = np.random.default_rng(100 + s).choice(GX.shape[1], 69, replace=False)
    add(f"attr_rand69_s{s}", *genome_feats(GX[:, cols]), "cov")
Xc = GX - GX.mean(0); Vh = torch.linalg.svd(tt(Xc), full_matrices=False)[2].numpy()
add("attr_pca69", *genome_feats(mm(Xc, Vh[:69].T.copy())), "cov")
add("exp69_perm", *genome_feats(GY[rng.permutation(len(GY))]), "cov")
Vc = V - V.mean(0)
for lam in (10.0, 100.0, 1000.0):
    add(f"direct_emb_l{int(lam)}", V, Vc, "all", lam)
evec = torch.linalg.eigh((tt(Vc).T @ tt(Vc)).double())[1].float().numpy(); Pc = np.ascontiguousarray(evec[:, -69:])
Fp = mm(Vc, Pc); add("emb_pca69", Fp, Fp - Fp.mean(0), "all")
Fr = mm(Vc, rng.standard_normal((V.shape[1], 69)).astype(np.float32) / np.float32(np.sqrt(69))); add("emb_rand69", Fr, Fr - Fr.mean(0), "all")

# ---------------- evaluation (definitions identical to reddit_ladder_test.py) ----------------
Vt = torch.tensor(V); DISC = 1.0 / torch.log2(torch.arange(2, 12).float())
def mt(S, Gm):
    top = torch.topk(S, 10, dim=1).indices; rel = Gm.gather(1, top).float(); ng = Gm.sum(1).float()
    idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]
    o = torch.stack([rel.sum(1) / ng.clamp(min=1), (rel * DISC).sum(1) / idcg, (rel.sum(1) > 0).float()], 1); o[ng == 0] = float("nan"); return o.numpy()
rz = lambda M: (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-9)
@torch.no_grad()
def evaluate(sp, comps, grid, unseen):
    D = SPLIT[sp]; N = len(D["L"]); res = {g: np.zeros((N, 3)) for g in grid}; resu = {g: np.zeros((N, 3)) for g in grid}
    for s in range(0, N, 1024):
        rows = list(range(s, min(N, s + 1024))); r = len(rows)
        B = rz(torch.tensor(D["Q"][rows]) @ Vt.T); Cs = [rz(torch.tensor(PM[c][sp][rows]) @ FF[c].T) for c in comps]
        cm = torch.zeros(r, NI, dtype=torch.bool); hm = torch.zeros_like(cm); Gm = torch.zeros_like(cm)
        for k, n in enumerate(rows):
            if D["ctx"][n]: cm[k, D["ctx"][n]] = True
            if D["hs"][n]: hm[k, D["hs"][n]] = True
            Gm[k, D["gold"][n]] = True
        for g in grid:
            S = B + g[0] * ZP[None]
            for gk, Ck in zip(g[1:], Cs):
                if gk: S = S + gk * Ck
            S = S.masked_fill(cm, -1e9); res[g][rows] = mt(S, Gm)
            if unseen: resu[g][rows] = mt(S.masked_fill(hm, -1e9), Gm & ~hm)
    return res, resu
def boot(A, B, users, nb=2000):
    d = A[:, 1] - B[:, 1]; ok = ~np.isnan(d); d = d[ok]; uu, inv = np.unique(users[ok], return_inverse=True)
    S = np.bincount(inv, d); C = np.bincount(inv); rb = np.random.default_rng(1); est = []
    for _ in range(nb):
        k = rb.integers(0, len(uu), len(uu)); est.append(S[k].sum() / C[k].sum())
    lo, hi = np.percentile(est, [2.5, 97.5]); return [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]

GP = [0.5, 0.75, 1.0]; G1 = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
OUT, ROWS = {}, {}
def do(name, comps, grid, refs=()):
    grid = [tuple(g) for g in grid]
    rv, _ = evaluate("valid", comps, grid, False); best = max(grid, key=lambda g: np.nanmean(rv[g][:, 1]))
    rt, rtu = evaluate("test", comps, [best], True); A, Au = rt[best], rtu[best]; D = SPLIT["test"]; warm = D["warm"]
    m = lambda M, mask: [round(float(v), 4) for v in np.nanmean(M[mask], 0)] + [int((~np.isnan(M[mask][:, 1])).sum())]
    o = dict(comps=comps, gammas_valid=list(best), valid_ndcg=round(float(np.nanmean(rv[best][:, 1])), 4),
             all=m(A, np.ones(len(warm), bool)), warm=m(A, warm), cold=m(A, ~warm), warm_unseen=m(Au, warm),
             info={c: INFO[c] for c in comps})
    for ref in refs: o["test_ndcg_diff_vs_" + ref] = boot(A, ROWS[ref], D["user"])
    OUT[name] = o; ROWS[name] = A
    log(f"{name:28s} g {best} valid {o['valid_ndcg']} | TEST R/NDCG/HIT {o['all'][:3]} cold {o['cold'][1]} warm {o['warm'][1]} | " +
        " ".join(f"vs {r}: {o['test_ndcg_diff_vs_' + r]}" for r in refs))
    json.dump(OUT, open(f"{OUTD}/c1_controls{SUF}.json", "w"), indent=1)

do("base", [], [(0,)])
do("base+pop", [], [(g,) for g in GP], refs=["base"])
g1grid = [(g, a) for g in GP for a in G1]
do("exp69", ["exp69"], g1grid, refs=["base+pop"])
log("SANITY: original reddit_ladder_test.json -> base+pop 0.1474, base+pop+c1 0.1563 (gammas 0.75, 0.2)")
singles = ["attr447", "all516"] + [f"attr_rand69_s{s}" for s in range(5)] + ["attr_pca69", "exp69_perm",
           "direct_emb_l10", "direct_emb_l100", "direct_emb_l1000", "emb_pca69", "emb_rand69"]
for v in singles: do(v, [v], g1grid, refs=["base+pop", "exp69"])
bd = max(("direct_emb_l10", "direct_emb_l100", "direct_emb_l1000"), key=lambda v: OUT[v]["valid_ndcg"])
g2grid = [(0.75, a, b) for a in G1 for b in G1]
for v in (bd, "attr447", "attr_pca69", "emb_pca69"):
    do(f"{v}+exp69", [v, "exp69"], g2grid, refs=["base+pop", "exp69", v])
np.savez_compressed(f"{OUTD}/c1_controls_rows{SUF}.npz", req=np.array([x["req"] for x in SPLIT["test"]["L"]]),
                    user=SPLIT["test"]["user"], **{k.replace("+", "__"): v for k, v in ROWS.items()})
log("DONE")
