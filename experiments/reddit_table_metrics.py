import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Reddit-Movie Table 1 metrics (HR@10, MRR@10, NDCG@10; multi-gold; context movies removed) for all systems from top-100 lists.
Ours = final system (union K100, tau .02, g 3, lambda 0) and A1, recomputed with top-100 saved. Baselines = saved top-100 rows (seed mean).
Analysis only (no selection). 2026-11-17."""
import os, json, glob, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{REPO}/common/pers_pop_reddit.py").read().split("OUT = {}")[0]
src = src.replace("top = torch.topk(Sc, 10, 1).indices", "topfull = torch.topk(Sc, 100, 1).indices; top = topfull[:, :10]")
src = src.replace("if want_top: tops.append(top.cpu())", "if want_top: tops.append(topfull.cpu())")
src = src.replace("tops.append(torch.topk(A, 10, 1).indices.cpu())", "tops.append(torch.topk(A, 100, 1).indices.cpu())")
exec(src)
L_ = SPLIT["test"]["L"]; posd = {x["req"]: n for n, x in enumerate(L_)}; GOLD = [set(g) for g in SPLIT["test"]["gold"]]
CTX = [set(c) for c in SPLIT["test"]["ctx"]]
def metr(tops):   # tops: list (per request, in test order) of item-index lists (already ctx-free)
    hr, mrr, nd = [], [], []
    for n, t in enumerate(tops):
        g = GOLD[n]
        if not g: continue
        t = t[:10]; hits = [r for r, j in enumerate(t) if j in g]
        hr.append(1.0 if hits else 0.0); mrr.append(1.0 / (hits[0] + 1) if hits else 0.0)
        idcg = sum(1 / np.log2(k + 2) for k in range(min(len(g), 10))); nd.append(sum(1 / np.log2(r + 2) for r in hits) / idcg)
    return np.array([np.mean(hr), np.mean(mrr), np.mean(nd)])
_, T0 = run("test", "A", 0, True); TA1 = run_a1("test")
OUT = {"ours": dict(zip(("HR", "MRR", "NDCG"), np.round(metr([list(t) for t in T0]), 4).tolist())),
       "A1": dict(zip(("HR", "MRR", "NDCG"), np.round(metr([[j for j in t if j not in CTX[n]] for n, t in enumerate(TA1)]), 4).tolist()))}
def rows_metric(pattern, field="top100"):
    fs = sorted(glob.glob(f"{W}/results/{pattern}")); ms = []
    for f in fs:
        T = [None] * len(L_)
        for l in open(f):
            x = json.loads(l); n = posd.get(x["req"])
            if n is None: continue
            T[n] = [(i if isinstance(i, int) else iidx.get(i)) for i in x.get(field, [])]; T[n] = [j for j in T[n] if j is not None and j not in CTX[n]]  # CRSLab rows store item indices
        T = [t if t is not None else [] for t in T]; ms.append(metr(T))
    return (dict(n_seeds=len(fs), **dict(zip(("HR", "MRR", "NDCG"), np.round(np.mean(ms, 0), 4).tolist()))) if ms else None)
for name, pat, fld in (("UniCRS", "reddit_crs_unicrs_s[0-9]_test_rows.jsonl", "top100"), ("KBRD", "reddit_crs_kbrd_s[0-9]_test_rows.jsonl", "top100"),
                       ("UCCR", "reddit_crs_uccr_reddit_ep1_s[0-9]_test_rows.jsonl", "top100"), ("MemoCRS", "reddit_crs_memocrs_reddit_s[0-9]_test_rows.jsonl", "top100"),
                       ("ZeroShotLLM", "llm_zs_reddit_test_none_ranked.jsonl", "ranked"), ("KGSF_new", "reddit_crs_kgsftune_lr1e-3_s0_test_rows.jsonl", "top100"),
                       ("ReDial_new", "reddit_crs_redialfix8_s0_test_rows.jsonl", "top100")):
    OUT[name] = rows_metric(pat, fld)
json.dump(OUT, open(f"{W}/results/reddit_table_metrics.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("RTM_DONE")
