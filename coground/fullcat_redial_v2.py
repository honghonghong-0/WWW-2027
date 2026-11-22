"""ReDial v2: tuning + dynamic-affordance candidate reranking. Fixed before running, 2026-11-17 (after v1 results; v1 grid edge g=2).
Reuses fullcat_redial.py up to the feature build (same A1, EP features; embedding cache).
(T) Tuning: additive A1 + g1 z(EP_emb,tau1) [+ g2 z(EP_prof,tau2)], wider grid tau in {.01,.02,.05,.1}, g in {0,.5,1,2,3,4,6}.
(R) Dynamic-affordance candidate set (two-stage): candidates = topK(A1) UNION topK(EP_emb)  (the community's relational links can bring in
    items the content match missed; K in {50,100,200}); inside the candidates rerank by z_local(A1) + g z_local(EP_emb) (+ g2 z_local(EP_prof));
    non-candidates ranked after all candidates by A1. g in {.5,1,2,3,4,6}.
    Control (R0): candidates = topK(A1) only (same local rerank) -> isolates the candidate-set change.
All choices on VALID NDCG@10, TEST once. Sample-level bootstrap (conv ids unavailable in the CRSLab export)."""
import os, json, numpy as np, torch
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fullcat_redial.py")).read()
src = src.split("crit = lambda r:")[0].replace("TAUS = [0.02, 0.05]", "TAUS = [0.01, 0.02, 0.05, 0.1]")
exec(src)
gp, g1c = 0.75, 0.2   # A1 gammas fixed from v1 (valid-selected there)
@torch.no_grad()
def a1(sp):
    B, C1, EP = FE[sp]; return B + gp * ZP[None] + g1c * C1
@torch.no_grad()
def rank_from(Sc, sp):
    gi = torch.as_tensor(gold[sp], device=D); gs = Sc.gather(1, gi.clamp(min=0)[:, None]); return ((Sc > gs).sum(1).float() + 1).cpu().numpy()
@torch.no_grad()
def additive(sp, terms):
    Sc = a1(sp).clone(); EP = FE[sp][2]
    for (s, t, g) in terms:
        if g: Sc += g * EP[(s, t)]
    return rank_from(Sc, sp)
@torch.no_grad()
def rerank(sp, K, union, t1, g, t2=None, g2=0):
    A = a1(sp); EP = FE[sp][2]; E1 = EP[("emb", t1)]
    cand = torch.zeros_like(A, dtype=torch.bool); cand.scatter_(1, torch.topk(A, K, 1).indices, True)
    if union: cand.scatter_(1, torch.topk(E1, K, 1).indices, True)
    def zl(X):
        m = torch.where(cand, X, torch.zeros_like(X)); n = cand.sum(1, keepdim=True).float()
        mu = m.sum(1, keepdim=True) / n; sd = (torch.where(cand, (X - mu) ** 2, torch.zeros_like(X)).sum(1, keepdim=True) / n).sqrt() + 1e-9
        return (X - mu) / sd
    L = zl(A) + g * zl(E1)
    if g2: L = L + g2 * zl(EP[("prof", t2)])
    Sc = torch.where(cand, L + 1e4, A); return rank_from(Sc, sp)
nd = lambda r: np.where(r <= 10, 1 / np.log2(r + 1), 0.0)
v = lambda r: float(nd(r).mean())
GG = [0.5, 1, 2, 3, 4, 6]; RES, RK = {}, {}
def pick(name, cands, fn):
    vals = {c: v(fn("valid", c)) for c in cands}; best = max(vals, key=vals.get); RK[name] = fn("test", best)
    r = RK[name]; RES[name] = dict(params=list(best), valid=round(vals[best], 4), NDCG10=round(v(r), 4), HR10=round(float((r <= 10).mean()), 4), HR50=round(float((r <= 50).mean()), 4)); log(name, RES[name])
pick("A1", [()], lambda sp, c: additive(sp, []))
pick("T:A1+EP_emb", [(t, g) for t in TAUS for g in GG], lambda sp, c: additive(sp, [("emb", c[0], c[1])]))
pick("T:A1+EP_emb+EP_prof", [(t1, a, t2, b) for t1 in TAUS for a in GG for t2 in TAUS for b in [0] + GG],
     lambda sp, c: additive(sp, [("emb", c[0], c[1]), ("prof", c[2], c[3])]))
for K in (50, 100, 200):
    pick(f"R0:topK(A1) K{K}", [(t, g) for t in TAUS for g in GG], lambda sp, c, K=K: rerank(sp, K, False, c[0], c[1]))
    pick(f"R:union K{K}", [(t, g) for t in TAUS for g in GG], lambda sp, c, K=K: rerank(sp, K, True, c[0], c[1]))
N = len(uni_nd); rb = np.random.default_rng(1); BI = [rb.integers(0, N, N) for _ in range(2000)]
bt = lambda d: [round(float(d.mean()), 4)] + [round(float(x), 4) for x in np.percentile([d[i].mean() for i in BI], [2.5, 97.5])]
OUT = {"UniCRS": round(float(uni_nd.mean()), 4), "systems": RES, "vs_UniCRS": {k: bt(nd(r) - uni_nd) for k, r in RK.items()},
       "vs_A1": {k: bt(nd(r) - nd(RK["A1"])) for k, r in RK.items() if k != "A1"}}
for K in (50, 100, 200): OUT[f"union - topK(A1) K{K}"] = bt(nd(RK[f"R:union K{K}"]) - nd(RK[f"R0:topK(A1) K{K}"]))
json.dump(OUT, open(f"{W}/results/fullcat_redial_v2.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("V2_DONE")
