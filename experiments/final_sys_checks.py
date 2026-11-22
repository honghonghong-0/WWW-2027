import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Analyses re-run on the FINAL system (union K100, tau .02, g 3) (2026-11-17). Analysis only, no selection.
(a) seeker-confirmed NDCG@10 (gold = test items the requester later said they watched & liked; and liked+will-watch), main vs A1 vs UniCRS.
(b) near-duplicate removal for the final system: keep test requests whose max cosine to any TRAIN request < .95/.90/.85.
(c) circularity check on the requester side: personal episodes (def A, lam 1/10/100) evaluated against the SEEKER-confirmed gold,
    i.e. the requester's own later choice rather than the community's picks. Warm requests only."""
import os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{REPO}/common/pers_pop_reddit.py").read().split("OUT = {}")[0])
L_ = SPLIT["test"]["L"]; posd = {x["req"]: n for n, x in enumerate(L_)}
R0, TOP0 = run("test", "A", 0, True)
TOPL = {lam: run("test", "A", lam, True)[1] for lam in (1, 10, 100)}
TOPA1 = run_a1("test")
UT = {}
for sd in range(3):
    for l in open(f"{W}/results/reddit_crs_unicrs_s{sd}_test_rows.jsonl"):
        x = json.loads(l); cx = set(SPLIT["test"]["ctx"][posd[x["req"]]])
        UT.setdefault(x["req"], []).append([iidx[i] for i in x["top100"] if i in iidx and iidx[i] not in cx][:10])
disc = 1 / np.log2(np.arange(2, 12))
def nd(top, g):
    idcg = disc[:min(len(g), 10)].sum(); return sum(disc[r] for r, j in enumerate(top[:10]) if j in g) / idcg
FG = {"liked": {}, "liked_or_will": {}}
for u in json.load(open(f"{W}/results/fb_labels.json")):
    if u["split"] != "test" or u["item"] not in iidx or u["req"] not in posd: continue
    if u["label"] == "WATCHED_LIKED": FG["liked"].setdefault(u["req"], set()).add(iidx[u["item"]])
    if u["label"] in ("WATCHED_LIKED", "WILL_WATCH"): FG["liked_or_will"].setdefault(u["req"], set()).add(iidx[u["item"]])
def cboot(d, us):
    uu, inv = np.unique(us, return_inverse=True); Sd = np.bincount(inv, d, minlength=len(uu)); Cn = np.bincount(inv, minlength=len(uu)); rb = np.random.default_rng(2)
    e = [Sd[k].sum() / Cn[k].sum() for k in (rb.integers(0, len(uu), len(uu)) for _ in range(2000))]
    return [round(float(d.mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4), len(d)]
OUT = {"seeker": {}, "personal_on_seeker_gold": {}, "dup_final": {}}
for gk, GD in FG.items():
    reqs = list(GD); ix = np.array([posd[r] for r in reqs]); us = users[ix]
    main = np.array([nd(TOP0[posd[r]], GD[r]) for r in reqs]); a1 = np.array([nd(TOPA1[posd[r]], GD[r]) for r in reqs])
    uni = np.array([np.mean([nd(t, GD[r]) for t in UT[r]]) for r in reqs])
    OUT["seeker"][gk] = dict(n=len(reqs), main=round(float(main.mean()), 4), A1=round(float(a1.mean()), 4), UniCRS=round(float(uni.mean()), 4),
                             main_minus_A1=cboot(main - a1, us), main_minus_UniCRS=cboot(main - uni, us))
    wm = warm[ix]
    for lam in (1, 10, 100):
        pl = np.array([nd(TOPL[lam][posd[r]], GD[r]) for r in reqs])
        OUT["personal_on_seeker_gold"][f"{gk} lam{lam}-lam0 (warm)"] = cboot((pl - main)[wm], us[wm]) if wm.sum() > 1 else None
QTt = QT.half(); mx = []
with torch.no_grad():
    for b in range(0, len(L_), 512):
        q = dev(SPLIT["test"]["Q"][b:b + 512]).half(); mx.append((q @ QTt.T).float().max(1).values.cpu())
mx = torch.cat(mx).numpy()
Rb = np.load(f"{W}/results/fullcat_base_rows.npz"); a1r, unir = Rb["A1"], Rb["uni"]
for th in (1.01, 0.95, 0.90, 0.85):
    k = (mx < th) & ~np.isnan(R0)
    OUT["dup_final"][f"keep<{th}"] = dict(n=int(k.sum()), main=round(float(np.nanmean(R0[k])), 4), main_minus_A1=boot(R0 - a1r, k), main_minus_UniCRS=boot(R0 - unir, k))
json.dump(OUT, open(f"{W}/results/final_sys_checks.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("FSC_DONE")
