import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Collective episodic affordance over the FULL catalogue (fixed before running, 2026-11-17).
Theory: popularity log n_i is a prior log p(i). Re-weighting each past choice episode c' of i by its similarity to the current request
(encoding specificity: past episodes are retrieved in proportion to context overlap) gives
   EP_s(i,C) = log sum_{c' in N(i)} exp(sim_s(C,c')/tau)  ~  log p(i) + log p(C | i)   (kernel mixture over i's episodes)
i.e. the posterior strength of the community's link between i and THIS kind of request. Scored for every item (no top-100 cut), so the
candidate set itself changes with the request. Items with no TRAIN episode get the request's minimum (= weakest link).
Spaces s: emb (request embedding), exp (read desired-experience profile p^exp), attr (read property profile p^attr),
          slot (LLM episode slots: companion/occasion/mood/sought outcomes, results/request_slots.json), slot_shuf (TRAIN slots permuted).
Score: A1 (= z base + .75 z pop + .2 z c1, all items) + g1 z(EP_emb) + g2 z(EP_s).  tau in {.02,.05}; g1,g2 in {0,.5,1,1.5,2}; VALID select, TEST once.
Primary (pre-stated): TEST (A1+EP_emb+EP_slot) - (A1+EP_emb) CI>0 AND (A1+EP_emb+EP_slot) - (A1+EP_emb+EP_slot_shuf) CI>0.
Also: paired TEST NDCG@10 vs UniCRS (mean of its 3 seeds, per request), user bootstrap 2000.
Seeker-confirmed evaluation (added 2026-11-17 before the slot run): gold = test items the SEEKER later said they watched and liked
(results/fb_labels.json, WATCHED_LIKED), and a wider gold LIKED+WILL_WATCH; only requests with >=1 such item; same top-10 lists.
UniCRS: its saved top-100 (seed-wise NDCG, mean of 3 seeds), context items removed."""
import os, json, re, numpy as np, torch
W = "/path/to/base/www27"
SPACES = os.environ.get("SPACES", "emb,exp,attr").split(",")
src = open(f"{REPO}/coground/c1_controls.py").read()
exec(src.split("# ---------------- variants")[0].replace('OUTD = f"{W}/c1/results"; os.makedirs(OUTD, exist_ok=True)', ""))
import re
PM, PT = {}, {}
def add(name, Fraw, Ff):
    Gw = reader("cov", Fraw); PM[name] = {}
    for sp in ("valid", "test"):
        Pm = pred(Gw, SPLIT[sp]["Q"]).astype(np.float32); Pm -= Pm.mean(0); PM[name][sp] = Pm
    Pt = pred(Gw, Qtrain).astype(np.float32); PT[name] = Pt - Pt.mean(0); return Ff
Yf = add("exp", *genome_feats(GY)); add("attr", *genome_feats(GX))
if any(s.startswith("slot") for s in SPACES):
    SJ = json.load(open(f"{W}/results/" + os.environ.get("SLOTFILE", "request_slots.json")))
    KEYS = [("companion", v) for v in ["alone", "partner", "friends", "family", "kids"]] + [("occasion", v) for v in ["date", "party_group", "holiday_season", "late_night", "background_casual"]] + \
           [("mood_now", v) for v in ["sad_down", "stressed_tired", "bored", "happy"]]
    OUTC = ["fun_laugh", "thrill_tension", "fear_scare", "comfort_warmth", "sad_catharsis", "awe_wonder", "think_reflect", "learn_inform", "mind_bending", "absorption", "bond_together", "talk_about", "nostalgia", "personal_relevance"]
    def svec(r):
        d = SJ.get(r, {}); v = [float(d.get(k) == x) for k, x in KEYS] + [float(o in d.get("outcomes", [])) for o in OUTC]; return v
    PT["slot"] = np.array([svec(x["req"]) for x in U["train"]], np.float32)
    PT["slot_shuf"] = PT["slot"][np.random.default_rng(0).permutation(len(PT["slot"]))]
    for s in ("slot", "slot_shuf"): PM[s] = {sp: np.array([svec(x["req"]) for x in SPLIT[sp]["L"]], np.float32) for sp in ("valid", "test")}
    print("slot coverage train/valid/test", [float((A.sum(1) > 0).mean()) for A in (PT["slot"], PM["slot"]["valid"], PM["slot"]["test"])], flush=True)
D = "cuda"; dev = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=D)
Vt = dev(V); ZPd = ZP.to(D); Yt = dev(Yf); QT = dev(Qtrain)
M = torch.zeros(NI, len(Qtrain), dtype=torch.float16, device=D)
for n, l in enumerate(gl_all):
    if l: M[l, n] = 1
FBB = float(os.environ.get("FBBETA", 0))
if FBB:   # realized-outcome weighting (fixed 2026-11-17): a TRAIN episode whose seeker later said they watched i and liked it counts (1+b),
          # disliked counts 1/(1+b) (symmetric likelihood ratio); labels results/fb_labels_train.json (LLM, noisy)
    tpos = {x["req"]: n for n, x in enumerate(U["train"])}; nl = nd_ = 0
    for u in json.load(open(f"{W}/results/fb_labels_train.json")):
        n = tpos.get(u["req"]); i = iidx.get(u["item"])
        if n is None or i is None or M[i, n] == 0: continue
        if u["label"] == "WATCHED_LIKED": M[i, n] = 1 + FBB; nl += 1
        elif u["label"] == "WATCHED_DISLIKED": M[i, n] = 1 / (1 + FBB); nd_ += 1
    print("FB weighting beta", FBB, "liked", nl, "disliked", nd_, flush=True)
has_ep = M.float().sum(1) > 0
nrm = lambda X: X / (X.norm(dim=1, keepdim=True) + 1e-9)
PTn = {s: nrm(dev(PT[s]).float()).half() for s in SPACES if s != "emb"}
rzt = lambda X: (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-9)
TAUS = [0.02, 0.05]
FE = {}
@torch.no_grad()
def build(sp):
    S = SPLIT[sp]; N = len(S["L"]); A1 = torch.empty(N, NI, dtype=torch.float16, device=D)
    EP = {(s, t): torch.empty(N, NI, dtype=torch.float16, device=D) for s in SPACES for t in TAUS}
    for b in range(0, N, 64):
        rows = list(range(b, min(N, b + 64))); q = dev(S["Q"][rows])
        Sc = rzt(q @ Vt.T) + 0.75 * ZPd[None] + 0.2 * rzt(dev(PM["exp"][sp][rows]) @ Yt.T)
        for k, n in enumerate(rows):
            if S["ctx"][n]: Sc[k, S["ctx"][n]] = -1e4
        A1[rows] = Sc.half()
        for s in SPACES:
            sm = (q @ QT.T) if s == "emb" else (nrm(dev(PM[s][sp][rows]).float()).half() @ PTn[s].T).float()
            mx = sm.max(1, keepdim=True).values
            for t in TAUS:
                E = torch.exp((sm - mx) / t).half()
                den = (M @ E.T).T.float()                                   # R x NI
                ep = torch.log(den + 1e-30)
                ep = torch.where(has_ep[None], ep, torch.full_like(ep, float("inf")))
                mn = ep.min(1, keepdim=True).values; ep = torch.where(has_ep[None], ep, mn)
                EP[(s, t)][rows] = rzt(ep).half()
    G = torch.zeros(N, NI, dtype=torch.bool, device=D)
    for n in range(N): G[n, S["gold"][n]] = True
    return A1, EP, G
for sp in ("valid", "test"): FE[sp] = build(sp); log("built", sp)
DISC = (1.0 / torch.log2(torch.arange(2, 12).float())).to(D)
@torch.no_grad()
def ndcg(sp, terms):
    A1, EP, G = FE[sp]; out = []
    for b in range(0, A1.shape[0], 512):
        Sb = A1[b:b + 512].float()
        for (s, t, g) in terms:
            if g: Sb = Sb + g * EP[(s, t)][b:b + 512].float()
        top = torch.topk(Sb, 10, dim=1).indices; Gb = G[b:b + 512]; rel = Gb.gather(1, top).float(); ng = Gb.sum(1).float()
        idcg = torch.cumsum(DISC, 0)[(ng.clamp(1, 10) - 1).long()]; v = (rel * DISC).sum(1) / idcg; v[ng == 0] = float("nan"); out.append(v.cpu())
    return torch.cat(v for v in out).numpy() if False else torch.cat(out).numpy()
GG = [0, 0.5, 1.0, 1.5, 2.0]
SYS = {"A1": []}
SYS["A1+EP_emb"] = ["emb"]
for s in SPACES:
    if s != "emb": SYS[f"A1+EP_{s}"] = [s]; SYS[f"A1+EP_emb+EP_{s}"] = ["emb", s]
RES, ROW = {}, {}
for k, sp_ in SYS.items():
    if not sp_: grid = [()]
    elif len(sp_) == 1: grid = [((sp_[0], t, g),) for t in TAUS for g in GG[1:]]
    else: grid = [(("emb", t1, g1), (sp_[1], t2, g2)) for t1 in TAUS for g1 in GG[1:] for t2 in TAUS for g2 in GG]
    vals = {p: float(np.nanmean(ndcg("valid", p))) for p in grid}; best = max(vals, key=vals.get)
    ROW[k] = ndcg("test", best); RES[k] = dict(params=[list(x) for x in best], valid=round(vals[best], 4), test=round(float(np.nanmean(ROW[k])), 4)); log(k, RES[k])
users = SPLIT["test"]["user"]; uu, inv = np.unique(users, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    ok = ~np.isnan(d); Sd = np.bincount(inv[ok], d[ok], minlength=len(uu)); C = np.bincount(inv[ok], minlength=len(uu))
    est = [Sd[k].sum() / max(C[k].sum(), 1) for k in BI]
    return [round(float(d[ok].mean()), 4), round(float(np.percentile(est, 2.5)), 4), round(float(np.percentile(est, 97.5)), 4), int(ok.sum())]
UC = {}
for s in range(3):
    for l in open(f"{W}/results/reddit_crs_unicrs_s{s}_test_rows.jsonl"):
        x = json.loads(l); UC.setdefault(x["req"], []).append(x["m"][1])
uni = np.array([np.mean(UC[x["req"]]) for x in SPLIT["test"]["L"]])
OUT = {"systems": RES, "unicrs_mean_same_rows": round(float(np.nanmean(uni[~np.isnan(ROW["A1"])])), 4), "contrasts": {}}
for k in ROW:
    if k != "A1": OUT["contrasts"][f"{k} - A1"] = boot(ROW[k] - ROW["A1"])
    OUT["contrasts"][f"{k} - UniCRS"] = boot(ROW[k] - uni)
for s in SPACES:
    if s not in ("emb",): OUT["contrasts"][f"EP_emb+EP_{s} - EP_emb"] = boot(ROW[f"A1+EP_emb+EP_{s}"] - ROW["A1+EP_emb"])
if "slot" in SPACES and "slot_shuf" in SPACES: OUT["contrasts"]["PRIMARY2 EP_emb+EP_slot - EP_emb+EP_slot_shuf"] = boot(ROW["A1+EP_emb+EP_slot"] - ROW["A1+EP_emb+EP_slot_shuf"])
if "attr" in SPACES and "exp" in SPACES: OUT["contrasts"]["EP_emb+EP_exp - EP_emb+EP_attr"] = boot(ROW["A1+EP_emb+EP_exp"] - ROW["A1+EP_emb+EP_attr"])
if "slot" in SPACES and "attr" in SPACES: OUT["contrasts"]["EP_emb+EP_slot - EP_emb+EP_attr"] = boot(ROW["A1+EP_emb+EP_slot"] - ROW["A1+EP_emb+EP_attr"])
tag = os.environ.get("TAG", "base")
np.savez_compressed(f"{W}/results/fullcat_{tag}_rows.npz", req=np.array([x["req"] for x in SPLIT["test"]["L"]]), user=users, uni=uni, **{k.replace("+", "__"): v for k, v in ROW.items()})
FBL = json.load(open(f"{W}/results/fb_labels.json")); posd = {x["req"]: n for n, x in enumerate(SPLIT["test"]["L"])}
FG = {"liked": {}, "liked_or_will": {}}
for u in FBL:
    if u["split"] != "test" or u["item"] not in iidx: continue
    if u["label"] == "WATCHED_LIKED": FG["liked"].setdefault(u["req"], set()).add(iidx[u["item"]])
    if u["label"] in ("WATCHED_LIKED", "WILL_WATCH"): FG["liked_or_will"].setdefault(u["req"], set()).add(iidx[u["item"]])
disc = DISC.cpu().numpy()
def nd(top, g):
    idcg = disc[:min(len(g), 10)].sum(); return sum(disc[r] for r, j in enumerate(top[:10]) if j in g) / idcg
@torch.no_grad()
def top10(params):
    A1, EP, G = FE["test"]; out = []
    for b in range(0, A1.shape[0], 512):
        Sb = A1[b:b + 512].float()
        for (sx, t, g) in params:
            if g: Sb = Sb + g * EP[(sx, t)][b:b + 512].float()
        out.append(torch.topk(Sb, 10, dim=1).indices.cpu())
    return torch.cat(out).numpy()
UT = {}
for sd in range(3):
    for l in open(f"{W}/results/reddit_crs_unicrs_s{sd}_test_rows.jsonl"):
        x = json.loads(l); cx = set(SPLIT["test"]["ctx"][posd[x["req"]]])
        UT.setdefault(x["req"], []).append([iidx[i] for i in x["top100"] if i in iidx and iidx[i] not in cx][:10])
TOPS = {k: top10([tuple(p) for p in RES[k]["params"]]) for k in RES}
OUT["seeker_confirmed"] = {}
for gk, GD in FG.items():
    reqs = [r for r in GD]; ix = np.array([posd[r] for r in reqs]); uq = users[ix]; uu2, inv2 = np.unique(uq, return_inverse=True)
    rows = {k: np.array([nd(TOPS[k][posd[r]], GD[r]) for r in reqs]) for k in TOPS}
    rows["UniCRS"] = np.array([np.mean([nd(t, GD[r]) for t in UT[r]]) for r in reqs])
    rb2 = np.random.default_rng(2); B2 = [rb2.integers(0, len(uu2), len(uu2)) for _ in range(2000)]
    def bt(d):
        Sd = np.bincount(inv2, d, minlength=len(uu2)); C = np.bincount(inv2, minlength=len(uu2)); e = [Sd[k].sum() / C[k].sum() for k in B2]
        return [round(float(d.mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4)]
    OUT["seeker_confirmed"][gk] = {"n_requests": len(reqs), "ndcg": {k: round(float(v.mean()), 4) for k, v in rows.items()},
        "vs_UniCRS": {k: bt(v - rows["UniCRS"]) for k, v in rows.items() if k != "UniCRS"}, "vs_A1": {k: bt(v - rows["A1"]) for k, v in rows.items() if k not in ("A1", "UniCRS")}}
json.dump(OUT, open(f"{W}/results/fullcat_{tag}.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("FULL_DONE")
