"""Personal mapping vs SELF labels (fixed before running, 2026-11-15). Tests cause 1: 'the personal signal exists but labels written by
OTHERS cannot reward it'. ReDial seekers answered, after the dialogue, whether they liked each movie the RECOMMENDER suggested
(initiatorQuestions: suggested=1, liked 1/0; 2=unknown dropped). These are self-labels.
Model (as designed 11/14): e_uj = A y_j + delta_u + eps ; requested item-side experience p_item = p_C - delta_u.
 e_uj   : Reddit request map Gw applied to a SEEKER sentence mentioning movie j (title -> "a movie"), minus Reddit train mean;
          experience-bearing only (d > 2 x null spread, same null as delta_u v2).
 y_j    : Tag Genome profile (title+year link) else Reddit text-ridge fill (same as ladder_crslab).
 A      : ridge y -> e over all units.  r_uj = e_uj - A y_j.
 delta_u: empirical-Bayes shrunk mean of r over the user's units available BEFORE the target:
          H  (history)  = units from the same seeker's EARLIER conversations (conversationId order; = ReDial-MS chaining)
          HS (history+session) = H + seeker units earlier in the SAME conversation (before the target's first mention)
          shrink w = n/(n + s2/tau2), s2 = pooled within-user residual variance, tau2 = between-user variance (method of moments).
 p_C    : Gw on the seeker turns of the conversation before the target's first mention, minus Reddit train mean.
Scores: shared = p_C . y_j ; pers = (p_C - delta_u) . y_j ; negative control = delta of a RANDOM other seeker.
Primary: within-conversation pairwise AUC (liked vs disliked suggested movies in the same conversation), pooled; seeker bootstrap.
Decision: personal mapping is rewarded by self-labels iff AUC(pers) - AUC(shared) CI > 0 AND AUC(pers) - AUC(negctrl) CI > 0.
Only pairs whose seeker has n >= 1 prior unit are scored (else pers == shared). Counts are reported."""
import os, re, json, ast, collections, numpy as np, torch
W = "/path/to/base/www27"
RAW = "/path/to/base/00_Research/01_Recommender_Systems/AAAI-27/baselines/UniCRS/data/redial"
IC = "/path/to/base/projects/crs/processed/item_chain"
GEN = "/path/to/base/datasets/tag_genome_2021/unzip/movie_dataset_public_final/raw/metadata.json"
P = f"{W}/data/reddit_movie/prep_large"; rng = np.random.default_rng(0)
lit = lambda v: v if isinstance(v, dict) else (ast.literal_eval(v) if v else {})
convs = sorted([json.loads(l) for f in ("train", "valid", "test") for l in open(f"{RAW}/{f}_data_dbpedia_raw.jsonl")], key=lambda r: int(r["conversationId"]))
# ---------- units (seeker sentences mentioning a movie) and targets
units = []; targets = []; names = {}
for ci, c in enumerate(convs):
    u = str(c["initiatorWorkerId"]); mm = lit(c["movieMentions"]); q = lit(c["initiatorQuestions"]); names.update({k: v for k, v in mm.items() if v})
    first = {}
    for mi, m in enumerate(c["messages"]):
        for mid in re.findall(r"@(\d+)", m["text"]): first.setdefault(mid, mi)
        if str(m["senderWorkerId"]) != u: continue
        for s in re.split(r"(?<=[.!?])\s+", m["text"]):
            for mid in dict.fromkeys(re.findall(r"@(\d+)", s)):
                if mid in mm and len(s) < 600: units.append((u, mid, re.sub(r"@\d+", "a movie", s).strip(), ci, mi))
    for mid, a in q.items():
        a = lit(a)
        if a.get("suggested") == 1 and a.get("liked") in (0, 1) and mid in first and first[mid] > 0:
            ctx = " ".join(re.sub(r"@\d+", "a movie", m["text"]) for m in c["messages"][:first[mid]] if str(m["senderWorkerId"]) == u).strip()
            if ctx: targets.append((u, mid, a["liked"], ci, first[mid], ctx[-3000:]))
print("convs", len(convs), "seeker units", len(units), "targets (suggested, liked/disliked, with seeker context)", len(targets),
      "liked share", round(np.mean([t[2] for t in targets]), 3), flush=True)
# ---------- y_j : genome link by title(+year), else Reddit text-ridge filler
tn = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", re.sub(r"^(the|a|an) ", "", re.sub(r"\(.*?\)", "", s.lower()).strip())).split())
Gz = np.load(f"{IC}/item_chain_v1.npz"); grow = {int(m): r for r, m in enumerate(Gz["items"])}; byt = {}
for l in open(GEN):
    m = json.loads(l); y = re.search(r"\((\d{4})\)\s*$", m["title"])
    if m["item_id"] in grow: byt.setdefault(tn(m["title"]), []).append((y.group(1) if y else None, grow[m["item_id"]]))
mids = sorted({t[1] for t in targets} | {x[1] for x in units}); mpos = {m: k for k, m in enumerate(mids)}; link = np.full(len(mids), -1)
for k, m in enumerate(mids):
    n = names.get(m, ""); c = byt.get(tn(n), []); y = re.search(r"(\d{4})", n)
    if y: c2 = [r for yy, r in c if yy == y.group(1)]; c = [(None, r) for r in c2] if c2 else (c if len(c) == 1 else [])
    if len(c) == 1: link[k] = c[0][1]
print("movies", len(mids), "genome-linked", int((link >= 0).sum()), flush=True)
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("Qwen/Qwen3-Embedding-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda"); enc.max_seq_length = 1024
INST = ("Instruct: Given a movie-seeking conversation (with optional past sessions of the same user), "
        "retrieve the movie the user wants next\nQuery: ")
def E(tx, query=True, bs=32):
    o = np.argsort([-len(t) for t in tx]); out = np.zeros((len(tx), 4096), np.float32)
    out[o] = enc.encode([tx[i] for i in o], prompt=INST if query else None, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return out
def ridge(Xa, Ya, lam):
    Xb = np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]); I = np.eye(Xb.shape[1]); I[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ Ya)
pred = lambda Wt, Xa: np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]) @ Wt
Z = np.load(f"{W}/results/reddit_large_emb.npz"); IJ = json.load(open(f"{P}/items.json"))
rl = [(k, grow[int(m)]) for k, m in enumerate(IJ["ml"]) if m is not None and int(m) in grow]; hasR = np.zeros(len(IJ["ml"]), bool); hasR[[k for k, _ in rl]] = True
ybar = Gz["Y"][[r for _, r in rl]].mean(0)
Wf = ridge(Z["V"][[k for k, _ in rl]].astype(np.float32), Gz["Y"][[r for _, r in rl]] - ybar, 10.0)
Vi = E([names.get(m, " ") or " " for m in mids], query=False, bs=128)
Y = np.where((link >= 0)[:, None], Gz["Y"][np.maximum(link, 0)] - ybar, pred(Wf, Vi)).astype(np.float32)
# Reddit request map Gw (same construction as the Reddit prelude)
iidx = {m: k for k, m in enumerate(Z["items"])}; T = dict(zip(Z["tids"], range(len(Z["tids"])))); ET = Z["T"].astype(np.float32)
YR = np.zeros((len(IJ["ml"]), Gz["Y"].shape[1]), np.float32)
for k, r in rl: YR[k] = Gz["Y"][r]
tr = [json.loads(l) for l in open(f"{P}/requests_train.jsonl")]; tr = [x for x in tr if any(g in iidx and hasR[iidx[g]] for g in x["gold"])]
Gw = ridge(np.stack([ET[T[x["req"]]] for x in tr]), np.stack([YR[[iidx[g] for g in x["gold"] if g in iidx and hasR[iidx[g]]]].mean(0) for x in tr]), 10.0)
pmu = pred(Gw, np.stack([ET[T[x["req"]]] for x in tr])).mean(0) - ybar   # Gw targets are uncentred genome; centre to Y's frame
NULL = ["Movies like a movie, a movie, a movie.", "Any recommendations similar to a movie?", "I watched a movie.", "Shows like a movie and a movie."]
eN = pred(Gw, E(NULL)) - ybar - pmu; c0 = eN.mean(0); spread = np.linalg.norm(eN - c0, axis=1).max()
eU = pred(Gw, E([x[2] for x in units])) - ybar - pmu; dU = np.linalg.norm(eU - c0, axis=1); keep = dU > 2 * spread
PC = pred(Gw, E([t[5] for t in targets])) - ybar - pmu
del enc; torch.cuda.empty_cache()
units = [x for x, k in zip(units, keep) if k]; eU = eU[keep]
print("experience-bearing seeker units", len(units), "of", len(keep), flush=True)
yU = Y[[mpos[x[1]] for x in units]]; A = ridge(yU, eU, 10.0); R = eU - pred(A, yU)
# EB hyper-parameters (pooled over dims)
byu = collections.defaultdict(list)
for k, x in enumerate(units): byu[x[0]].append(k)
multi = [ks for ks in byu.values() if len(ks) >= 2]
s2 = float(np.mean([((R[ks] - R[ks].mean(0)) ** 2).sum(1).sum() / (len(ks) - 1) for ks in multi]))   # per-unit noise (trace)
mv = np.array([(R[ks].mean(0) ** 2).sum() for ks in byu.values()]); nn = np.array([len(ks) for ks in byu.values()])
tau2 = max(float(np.mean(mv - s2 / nn)), 1e-6); kappa = s2 / tau2
print("EB: s2", round(s2, 4), "tau2", round(tau2, 5), "kappa (units for half-weight)", round(kappa, 2), flush=True)
def delta(u, ci, mi, same_session):
    ks = [k for k in byu.get(u, []) if units[k][3] < ci or (same_session and units[k][3] == ci and units[k][4] < mi)]
    if not ks: return np.zeros(R.shape[1], np.float32), 0
    return (len(ks) / (len(ks) + kappa)) * R[ks].mean(0), len(ks)
yT = Y[[mpos[t[1]] for t in targets]]; lab = np.array([t[2] for t in targets]); cid = np.array([t[3] for t in targets]); usr = np.array([t[0] for t in targets])
shared = np.sum(PC * yT, 1)
OUT = {"n_targets": len(targets), "n_units_kept": len(units), "EB": {"s2": s2, "tau2": tau2, "kappa": kappa}}
for tag, ss in (("H_history", False), ("HS_history_plus_session", True)):
    D = []; N = []
    for t in targets: d_, n_ = delta(t[0], t[3], t[4], ss); D.append(d_); N.append(n_)
    D = np.array(D); N = np.array(N); pers = np.sum((PC - D) * yT, 1)
    perm = rng.permutation(len(targets)); negc = np.sum((PC - D[perm]) * yT, 1)
    ok = N >= 1
    pairs = []   # within-conversation (liked, disliked) pairs
    for c_ in np.unique(cid[ok]):
        idx = np.where(ok & (cid == c_))[0]; L1 = idx[lab[idx] == 1]; L0 = idx[lab[idx] == 0]
        pairs += [(i, j) for i in L1 for j in L0]
    pairs = np.array(pairs)
    if len(pairs) < 30: OUT[tag] = {"pairs": int(len(pairs)), "scored_targets": int(ok.sum())}; continue
    pu = usr[pairs[:, 0]]; us_ = np.unique(pu); grp = {u: np.where(pu == u)[0] for u in us_}
    def auc(s, ix): a = s[pairs[ix, 0]] - s[pairs[ix, 1]]; return float(np.mean((a > 0) + 0.5 * (a == 0)))
    allix = np.arange(len(pairs)); res = {"shared": auc(shared, allix), "pers": auc(pers, allix), "negctrl": auc(negc, allix)}
    bs = collections.defaultdict(list)
    for _ in range(2000):
        ix = np.concatenate([grp[u] for u in rng.choice(us_, len(us_))])
        a_s, a_p, a_n = auc(shared, ix), auc(pers, ix), auc(negc, ix); bs["pers-shared"].append(a_p - a_s); bs["pers-negctrl"].append(a_p - a_n); bs["shared"].append(a_s)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    OUT[tag] = {"scored_targets": int(ok.sum()), "pairs": int(len(pairs)), "seekers": int(len(us_)), "median_prior_units": float(np.median(N[ok])),
                "AUC": {k: round(v, 4) for k, v in res.items()}, "shared_ci": q(bs["shared"]),
                "pers_minus_shared": [round(res["pers"] - res["shared"], 4)] + q(bs["pers-shared"]),
                "pers_minus_negctrl": [round(res["pers"] - res["negctrl"], 4)] + q(bs["pers-negctrl"])}
json.dump(OUT, open(f"{W}/results/redial_selflabel_personal.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("SELFLABEL_DONE")
