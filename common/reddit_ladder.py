"""Reddit-Movie large, valid only. Stage 1 (references) + Stage 2 (light affordance write), one cross-fitted ladder.
score(u, i) = z(base) + g_pop z(pop) + g1 z(c1) + g2 z(c_k)            (z = per-request standardization over candidates)
  base  : Qwen3-Emb cosine of request (or history-concatenated request) and item title
  pop   : log1p #first-level recommendations in TRAIN threads (no valid/test leakage)
  c1    : p_C . y_i      p_C = ridge G(request emb) -> mean experience profile of covered golds (fit on train)
  y_i   : Tag Genome experience profile (69), uncovered items filled by ridge from title embedding (F2), centered
Stage 2 (no LLM; hierarchical-Bayes shrinkage write per user, read gated by current desire):
  write : delta_u = sum_j (y_j - ybar) / (n_u + lam), j = items the seeker named in EARLIER threads (opening + own replies)
  topic : delta_{u,k} = (sum_{j in topic k} (y_j - ybar) + lam_t delta_u) / (n_{u,k} + lam_t), k = k-means topic of the request
  nb    : collective branch = mean delta of the 50 most similar TRAIN requests' seekers (works for cold users)
  read  : c_taste = delta . y_i ; c_gate = (p_C * delta) . y_i
Controls: delta of a random other warm user (count-free), shuffled-topic delta.
Metrics per request (multi-gold, He et al.): Recall@10, NDCG@10, HIT@10. Views: all | warm (has prior thread) | cold |
warm-unseen (golds not in own history; own-history items removed from candidates).
Gammas: 2-fold user cross-fitting on valid (NDCG@10 on all requests). CI: paired user-level bootstrap vs reference."""
import json, sys, collections, numpy as np
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
rng = np.random.default_rng(0)
Z = np.load(f"{W}/results/reddit_large_emb.npz")
items = list(Z["items"]); iidx = {m: k for k, m in enumerate(items)}; NI = len(items)
V = Z["V"].astype(np.float32); T = dict(zip(Z["tids"], range(len(Z["tids"])))); ET = Z["T"].astype(np.float32)
H = dict(zip(Z["hq_keys"], range(len(Z["hq_keys"])))); EH = Z["H"].astype(np.float32)
TH = json.load(open(f"{P}/threads.json")); IJ = json.load(open(f"{P}/items.json"))
U = {sp: [json.loads(l) for l in open(f"{P}/requests_{sp}.jsonl")] for sp in ("train", "valid")}
names = json.load(open(f"{W}/data/reddit_movie/large/id2name.json")); noname = np.array([m not in names for m in items])
# ---- genome profiles ----
G = np.load("/path/to/base/projects/crs/processed/item_chain/item_chain_v1.npz")
grow = {int(m): r for r, m in enumerate(G["items"])}
has = np.array([(m is not None and int(m) in grow) for m in IJ["ml"]])
Y = np.zeros((NI, G["Y"].shape[1]), np.float32)
for k, m in enumerate(IJ["ml"]):
    if has[k]: Y[k] = G["Y"][grow[int(m)]]
def ridge(Xa, Ya, lam):
    Xb = np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]); I = np.eye(Xb.shape[1]); I[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ Ya)
pred = lambda Wt, Xa: np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]) @ Wt
ybar = Y[has].mean(0); idx = np.where(has)[0]
Wy = ridge(V[idx], Y[idx] - ybar, 10.0); Yf = np.where(has[:, None], Y - ybar, pred(Wy, V)).astype(np.float32)
# ---- popularity from TRAIN threads only ----
POP = np.zeros(NI)
for x in U["train"]:
    for g in x["gold"]: POP[iidx[g]] += 1
POP = np.log1p(POP)
# ---- p_C ----
tr = [x for x in U["train"] if any(has[iidx[g]] for g in x["gold"])]
Qtr = np.stack([ET[T[x["req"]]] for x in tr]); Ttr = np.stack([Y[[iidx[g] for g in x["gold"] if has[iidx[g]]]].mean(0) for x in tr])
Gw = ridge(Qtr, Ttr, 10.0)
import os
VA = U["valid"][:int(os.environ["PILOT"])] if os.environ.get("PILOT") else U["valid"]; NV = len(VA)
SUF = "_pilot" if os.environ.get("PILOT") else ""
Qv = np.stack([ET[T[x["req"]]] for x in VA]); Pv = pred(Gw, Qv); Pv -= Pv.mean(0)
Qh = np.stack([EH[H[x["req"]]] if x["req"] in H else ET[T[x["req"]]] for x in VA])
warm = np.array([bool(x["hist"]) for x in VA]); uid = np.array([x["user"] for x in VA])
gold = [[iidx[g] for g in x["gold"]] for x in VA]; ctx = [[iidx[c] for c in x["ctx_items"] if c in iidx] for x in VA]
def hist_items(x): return [iidx[j] for h in x["hist"] for j in TH[h]["items"] + [i for f in TH[h]["fb"] for i in f["items"]] if j in iidx]
HIST = [hist_items(x) for x in VA]
HSEEN = [sorted(set(HIST[n]) | {iidx[j] for h in VA[n]["hist"] for j in TH[h]["rec"] if j in iidx}) for n in range(NV)]
# ---- stage 2 write ----
from sklearn.cluster import MiniBatchKMeans
def deltas(lam, K=None, lam_t=5.0, km=None):
    D = np.zeros((NV, Yf.shape[1]), np.float32)
    for n, x in enumerate(VA):
        J = HIST[n]
        if not J: continue
        dev = Yf[J]  # already centered
        dg = dev.sum(0) / (len(J) + lam)
        if K is None: D[n] = dg; continue
        kc = KV[n]
        Jk = [j for h in x["hist"] if KT[T[h]] == kc for j in [iidx[i] for i in TH[h]["items"] + [i for f in TH[h]["fb"] for i in f["items"]] if i in iidx]]
        D[n] = (Yf[Jk].sum(0) + lam_t * dg) / (len(Jk) + lam_t) if Jk else dg
    return D
def nb_delta(lam, k=50):
    # collective branch: seekers of the most similar TRAIN requests, their delta from all their train-time named items
    tq = np.stack([ET[T[x["req"]]] for x in U["train"]]); tu = [x["user"] for x in U["train"]]
    named = collections.defaultdict(list)
    for x in U["train"]:
        named[x["user"]] += [iidx[i] for i in x["ctx_items"] if i in iidx]
    ud = {u: Yf[J].sum(0) / (len(J) + lam) for u, J in named.items() if J}
    D = np.zeros((NV, Yf.shape[1]), np.float32)
    for s in range(0, NV, 512):
        S = Qv[s:s + 512] @ tq.T
        for r, row in enumerate(S):
            n = s + r; top = np.argpartition(-row, k)[:k]
            ds = [ud[tu[t]] for t in top if tu[t] in ud and tu[t] != uid[n]]
            if ds: D[n] = np.mean(ds, 0)
    return D
# ---- eval ----
def zs(x): s = x.std(); return (x - x.mean()) / s if s > 0 else x * 0
ZPOP = zs(POP)
def _m(S, rows, golds):
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    sc = np.take_along_axis(S, top, 1); top = np.take_along_axis(top, np.argsort(-sc, 1), 1)
    out = np.full((len(rows), 3), np.nan)
    for r in range(len(rows)):
        gs = golds[r]
        if not gs: continue
        rel = [t in gs for t in top[r]]
        dcg = sum(1 / np.log2(k + 2) for k, x in enumerate(rel) if x); idcg = sum(1 / np.log2(k + 2) for k in range(min(10, len(gs))))
        out[r] = (sum(rel) / len(gs), dcg / idcg, float(any(rel)))
    return out
def metrics_for(S, rows):
    for r, n in enumerate(rows): S[r, ctx[n]] = -1e9
    return _m(S, rows, [set(gold[n]) for n in rows])
def unseen_metrics(S, rows):
    for r, n in enumerate(rows): S[r, HSEEN[n]] = -1e9
    return _m(S, rows, [set(gold[n]) - set(HSEEN[n]) for n in rows])
users = sorted(set(uid)); fold = {u: i % 2 for i, u in enumerate(rng.permutation(users))}; fv = np.array([fold[u] for u in uid])
def build(qsrc, terms):
    """terms: list of (name, per-request vector fn or global) -> returns function g -> score chunk"""
    pass
CH = 1024
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
Vt = torch.tensor(V, device=DEV); ZPOPt = torch.tensor(ZPOP, dtype=torch.float32, device=DEV)
DISC = 1.0 / torch.log2(torch.arange(2, 12, device=DEV).float())
def _mt(S, G):
    """S scores (r x NI, masked), G gold bool (r x NI) -> (r x 3) recall, ndcg, hit; NaN where no gold"""
    top = torch.topk(S, 10, dim=1).indices; rel = G.gather(1, top).float(); ng = G.sum(1).float()
    idcg = torch.stack([DISC[:int(min(10, k))].sum() if k > 0 else torch.tensor(1.0, device=DEV) for k in ng.tolist()])
    out = torch.stack([rel.sum(1) / ng.clamp(min=1), (rel * DISC).sum(1) / idcg, (rel.sum(1) > 0).float()], 1)
    out[ng == 0] = float("nan"); return out.cpu().numpy()
def evaluate(qmat, comps, grid):
    """GPU version (same definitions as the numpy pilot): ctx items masked; unseen view also masks own-history items and drops them from gold"""
    res = {g: np.zeros((NV, 3)) for g in grid}; resu = {g: np.zeros((NV, 3)) for g in grid}
    for s in range(0, NV, CH):
        rows = list(range(s, min(NV, s + CH))); r = len(rows)
        B = torch.tensor(qmat[rows], device=DEV) @ Vt.T; B = (B - B.mean(1, keepdim=True)) / (B.std(1, keepdim=True) + 1e-9)
        Cs = [torch.tensor(c(rows), dtype=torch.float32, device=DEV) for c in comps]
        cm = torch.zeros(r, NI, dtype=torch.bool, device=DEV); hm = torch.zeros_like(cm); G = torch.zeros_like(cm)
        for k, n in enumerate(rows):
            if ctx[n]: cm[k, ctx[n]] = True
            if HSEEN[n]: hm[k, HSEEN[n]] = True
            G[k, gold[n]] = True
        Gu = G & ~hm
        for g in grid:
            S = B + g[0] * ZPOPt[None]
            for gk, Ck in zip(g[1:], Cs):
                if gk: S = S + gk * Ck
            S = S.masked_fill(cm, -1e9); res[g][rows] = _mt(S, G)
            resu[g][rows] = _mt(S.masked_fill(hm, -1e9), Gu)
    return res, resu
def rowz(M): return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
def c_lin(Pm):  # (p . y_i) with p per request
    return lambda rows: rowz(Pm[rows] @ Yf.T)
def c_gate(Pm, Dm):
    return lambda rows: rowz((Pm[rows] * Dm[rows]) @ Yf.T)
def crossfit(res):
    # pick gamma on the other fold (NDCG@10, all requests)
    pick = {f: max(res, key=lambda g: np.nanmean(res[g][fv == 1 - f, 1])) for f in (0, 1)}
    return pick
REF = {}
def run(name, qmat, comps, grids, ref=None):
    grid = [tuple(g) for g in grids]
    res, resu = evaluate(qmat, comps, grid); pick = crossfit(res)
    A = np.zeros((NV, 3)); Au = np.zeros((NV, 3))
    for f in (0, 1): A[fv == f] = res[pick[f]][fv == f]; Au[fv == f] = resu[pick[f]][fv == f]
    REF[name] = (A, Au)
    def m(M, mask): M = M[mask]; M = M[~np.isnan(M[:, 1])]; return [round(float(v), 4) for v in M.mean(0)] + [int(len(M))]
    out = dict(all=m(A, np.ones(NV, bool)), warm=m(A, warm), cold=m(A, ~warm), warm_unseen=m(Au, warm), gammas={str(k): v for k, v in pick.items()})
    if ref:
        out["ci_vs_" + ref] = {v: boot(A, REF[ref][0], mask) for v, mask in (("all", np.ones(NV, bool)), ("warm", warm), ("cold", ~warm))}
        out["ci_vs_" + ref]["warm_unseen"] = boot(Au, REF[ref][1], warm)
    print(f"{name:30s} R@10/NDCG@10/HIT@10 all {out['all']} warm {out['warm']} cold {out['cold']} warm-unseen {out['warm_unseen']} | g {out['gammas']}", flush=True)
    if ref: print("   CI NDCG@10 vs", ref, {k: v for k, v in out["ci_vs_" + ref].items()}, flush=True)
    return out
def boot(A, B, mask, B_=1000):
    ok = mask & ~np.isnan(A[:, 1]) & ~np.isnan(B[:, 1]); d = A[ok, 1] - B[ok, 1]; us = uid[ok]
    uu, inv = np.unique(us, return_inverse=True); S = np.bincount(inv, d); C = np.bincount(inv); rb = np.random.default_rng(1); est = []
    for _ in range(B_):
        k = rb.integers(0, len(uu), len(uu)); est.append(S[k].sum() / C[k].sum())
    lo, hi = np.percentile(est, [2.5, 97.5]); return [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]
# ---- sanity prints (pilot checklist) ----
gall = [g for gs in gold for g in gs]
print(f"valid requests {NV} warm {warm.sum()} | golds {len(gall)} genome-covered {np.mean(has[gall]):.3f} nameless {np.mean(noname[gall]):.3f} "
      f"| genome items {has.sum()} | G train fit rows {len(tr)} | mean |delta-row| check later", flush=True)
GP = [0, 0.25, 0.5, 0.75, 1.0]; G1 = [0, 0.1, 0.2, 0.3, 0.5]
R = {}
R["base"] = run("base", Qv, [], [(0,)])
R["base+pop"] = run("base+pop", Qv, [], [(g,) for g in GP], ref="base")
R["base+pop+c1"] = run("base+pop+c1", Qv, [c_lin(Pv)], [(g, a) for g in GP for a in G1], ref="base+pop")
R["histcat+pop"] = run("histcat+pop", Qh, [], [(g,) for g in GP], ref="base+pop")
R["histcat+pop+c1"] = run("histcat+pop+c1", Qh, [c_lin(Pv)], [(g, a) for g in GP for a in G1], ref="base+pop+c1")
json.dump(R, open(f"{W}/results/reddit_ladder_stage1{SUF}.json", "w"), indent=1)
GPf = R["base+pop+c1"]["gammas"]["0"][0]; print("pop gamma fixed for stage 2:", GPf, flush=True)
# ---- stage 2 ----
G2 = [0, 0.05, 0.1, 0.2, 0.3]
grid2 = [(GPf, a, b) for a in G1 for b in G2]
for lam in (1.0, 5.0, 20.0):
    Dg = deltas(lam); print(f"lam {lam}: warm delta norm median {np.median(np.linalg.norm(Dg[warm], axis=1)):.3f}", flush=True)
    R[f"aff_taste_l{lam}"] = run(f"+c1+taste(l{lam})", Qv, [c_lin(Pv), c_lin(Dg)], grid2, ref="base+pop+c1")
    R[f"aff_gate_l{lam}"] = run(f"+c1+gate(l{lam})", Qv, [c_lin(Pv), c_gate(Pv, Dg)], grid2, ref="base+pop+c1")
    perm = rng.permutation(np.where(warm)[0]); Ds = Dg.copy(); Ds[np.where(warm)[0]] = Dg[perm]
    R[f"ctrl_otheruser_taste_l{lam}"] = run(f"  ctrl other-user taste(l{lam})", Qv, [c_lin(Pv), c_lin(Ds)], grid2, ref="base+pop+c1")
json.dump(R, open(f"{W}/results/reddit_ladder_stage2{SUF}.json", "w"), indent=1)
best_lam = max((1.0, 5.0, 20.0), key=lambda l: R[f"aff_taste_l{l}"]["all"][1])
km = MiniBatchKMeans(16, random_state=0, n_init=3).fit(np.stack([ET[T[x["req"]]] for x in U["train"]]))
KT = km.predict(ET); KV = km.predict(Qv)
Dt = deltas(best_lam, K=16, km=km)
R["aff_topic_taste"] = run(f"+c1+topic-taste(K16,l{best_lam})", Qv, [c_lin(Pv), c_lin(Dt)], grid2, ref="base+pop+c1")
R["aff_topic_gate"] = run(f"+c1+topic-gate(K16,l{best_lam})", Qv, [c_lin(Pv), c_gate(Pv, Dt)], grid2, ref="base+pop+c1")
Dn = nb_delta(best_lam)
R["aff_nb_taste"] = run(f"+c1+collective-taste(l{best_lam})", Qv, [c_lin(Pv), c_lin(Dn)], grid2, ref="base+pop+c1")
Dmix = np.where(warm[:, None], Dt, Dn)
R["aff_hier"] = run("+c1+own(topic) else collective", Qv, [c_lin(Pv), c_lin(Dmix)], grid2, ref="base+pop+c1")
json.dump(R, open(f"{W}/results/reddit_ladder_stage2{SUF}.json", "w"), indent=1); print("DONE", flush=True)
