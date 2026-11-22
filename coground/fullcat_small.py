"""Relational popularity on Inspired and TG-ReDial (CRSLab samples), same protocol as ReDial. Fixed before running, 2026-11-17.
A1 = ladder_crslab recipe (gamma_pop, gamma_1 on VALID by HR@1+HR@50). Relational popularity on the request embedding:
  (T) additive A1 + g z(RP_tau), tau in {.01,.02,.05,.1}, g in {.5,1,2,3,4,6}, VALID NDCG@10;
  (R) union candidates top100(A1) U top100(RP), local-z rerank, (tau, g) on VALID NDCG@10 (K=100 fixed from Reddit-Movie).
TEST once. Paired vs UniCRS and KBRD (3 seeds where available), conversation bootstrap (conversation = run of samples whose context
extends the previous one). Also reports episode density (train samples, median max cosine of test requests to train requests).
usage: python fullcat_small.py inspired|tgredial"""
import sys, json, re, os, time, numpy as np, torch
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
DS = sys.argv[1]; H = "/path/to/base"; W = f"{H}/00_Research/01_Recommender_Systems/WWW-2027"; X = f"{W}/data/crslab_export/{DS}"
TOKD = {"inspired": "nltk", "tgredial": "pkuseg", "redialorig": "nltk"}[DS]; TAGD = {"inspired": "insp", "tgredial": "tg", "redialorig": "redialorig"}[DS]
C = (f"{H}/projects/crs/baselines/crslab_orig/data/dataset/redial/nltk" if DS == "redialorig" else f"{H}/projects/crs/baselines/crslab/data/dataset/{DS}/{TOKD}")
IC = f"{H}/projects/crs/processed/item_chain"; GEN = f"{H}/datasets/tag_genome_2021/unzip/movie_dataset_public_final/raw/metadata.json"
ITEMS = json.load(open(f"{C}/movie_ids.json")); pos = {e: k for k, e in enumerate(ITEMS)}; NI = len(ITEMS)
i2e = json.load(open(f"{X}/vocab.json"))["id2entity"]
S = {sp: [json.loads(l) for l in open(f"{X}/{sp}.jsonl")] for sp in ("train", "valid", "test")}
log(DS, {k: len(v) for k, v in S.items()}, "items", NI, "| test labels outside pool", sum(x["item"] not in pos for x in S["test"]))
iname = lambda e: re.sub(r"^<http://dbpedia.org/resource/|>$", "", i2e.get(str(e), "")).replace("_", " ")
names = [iname(e) or " " for e in ITEMS]
tn = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", re.sub(r"^(the|a|an) ", "", re.sub(r"\(.*?\)", "", s.lower()).strip())).split())
Gz = np.load(f"{IC}/item_chain_v1.npz"); grow = {int(m): r for r, m in enumerate(Gz["items"])}
byt = {}
for l in open(GEN):
    m = json.loads(l); y = re.search(r"\((\d{4})\)\s*$", m["title"]); k = tn(m["title"])
    if m["item_id"] in grow: byt.setdefault(k, []).append((y.group(1) if y else None, grow[m["item_id"]]))
link = np.full(NI, -1)
for k, n in enumerate(names):
    c = byt.get(tn(n), []); y = re.search(r"(\d{4})", n)
    if y: c2 = [r for yy, r in c if yy == y.group(1)]; c = [(None, r) for r in c2] if c2 else (c if len(c) == 1 else [])
    if len(c) == 1: link[k] = c[0][1]
log("genome-linked", int((link >= 0).sum()))
INST = ("Instruct: Given a movie-seeking conversation (with optional past sessions of the same user), retrieve the movie the user wants next\nQuery: ")
clip = lambda s, n: s if len(s) <= n else s[-n:]
ctxt = lambda x: clip(("\n" if DS == "inspired" else "\n").join(u for u in x["ctx"] if u.strip()), 3000)
_cf = f"{W}/results/emb_cache_{DS}.npz"
if os.path.exists(_cf):
    _z = np.load(_cf); Vi = _z["Vi"]; Q = {sp: _z["Q_" + sp] for sp in S}; log("embeddings from cache")
else:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda"); enc.max_seq_length = 2048
    def E(texts, query, bs=32):
        order = np.argsort([-len(t) for t in texts]); out = np.zeros((len(texts), 4096), np.float32)
        out[order] = enc.encode([texts[i] for i in order], prompt=INST if query else None, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return out
    Vi = E(names, False, 256); Q = {sp: E([ctxt(x) for x in S[sp]], True) for sp in S}; log("encoded")
    del enc; torch.cuda.empty_cache(); np.savez(_cf, Vi=Vi, **{"Q_" + sp: Q[sp] for sp in S})
def ridge(Xa, Ya, lam):
    Xt = torch.as_tensor(np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]), device="cuda", dtype=torch.float64); I = torch.eye(Xt.shape[1], device="cuda", dtype=torch.float64); I[-1, -1] = 0
    return torch.linalg.solve(Xt.T @ Xt + lam * I, Xt.T @ torch.as_tensor(Ya, device="cuda", dtype=torch.float64)).float().cpu().numpy()
pred = lambda Wt, Xa: np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]) @ Wt
Z = np.load(f"{W}/results/reddit_large_emb.npz"); IJ = json.load(open(f"{W}/data/reddit_movie/prep_large/items.json"))
rl = [(k, grow[int(m)]) for k, m in enumerate(IJ["ml"]) if m is not None and int(m) in grow]
ybar = Gz["Y"][[r for _, r in rl]].mean(0)
Wf = ridge(Z["V"][[k for k, _ in rl]].astype(np.float32), Gz["Y"][[r for _, r in rl]] - ybar, 10.0)
Yf = np.where((link >= 0)[:, None], Gz["Y"][np.maximum(link, 0)] - ybar, pred(Wf, Vi)).astype(np.float32)
POP = np.zeros(NI)
for x in S["train"]:
    if x["item"] in pos: POP[pos[x["item"]]] += 1
POP = np.log1p(POP)
trk = [n for n, x in enumerate(S["train"]) if x["item"] in pos]
Gw = ridge(Q["train"][trk], Yf[[pos[S["train"][n]["item"]] for n in trk]], 10.0)
P = {sp: pred(Gw, Q[sp]).astype(np.float32) for sp in S}; Pm = P["train"].mean(0)
for sp in S: P[sp] -= Pm
log("ridge done")
def conv_ids(L):
    out, c, prev = [], -1, None
    for x in L:
        k = x["ctx"]
        if prev is None or len(k) < len(prev) or k[:len(prev)] != prev: c += 1
        out.append(c); prev = k
    return np.array(out)
CL = conv_ids(S["test"])
def dump_ranks(path):
    rows = [json.loads(l) for l in open(path)]; t = [r for r in rows if r[1] == "test"]
    eps = set(r[0] for r in t); ep = "final" if "final" in eps else max(e for e in eps if not isinstance(e, str))
    t = [r for r in t if r[0] == ep]
    assert [r[4] for r in t] == [x["item"] for x in S["test"]], f"order mismatch {path}"
    return np.array([r[2] for r in t], float)
D_ = f"{W}/results/dumps"
BASE = {"KBRD": [dump_ranks(f"{D_}/kbrd_{TAGD}_s{s}.jsonl") for s in range(3)]}
for nm, pat in (("KGSF", "kgsf_{t}_s{s}"), ("ReDial", "redialfix_{t}_s{s}"), ("TGReDial", "tgrec_{t}_s{s}"), ("ZeroShotLLM", "qwen_zs_{t}")):
    fs = [f"{D_}/" + pat.format(t=TAGD, s=s) + ".jsonl" for s in range(3)]; fs = [f for f in dict.fromkeys(fs) if os.path.exists(f)]
    if fs: BASE[nm] = [dump_ranks(f) for f in fs]
if DS == "inspired":
    BASE["UniCRS"] = [dump_ranks(f"{D_}/unicrs_insp_sel_s{s}.jsonl") for s in range(3) if os.path.exists(f"{D_}/unicrs_insp_sel_s{s}.jsonl")] or [dump_ranks(f"{D_}/unicrs_insp_s0.jsonl")]
elif DS == "redialorig":
    U = f"{H}/projects/crs/baselines/unicrs/src/data/redialorig_cl"
    e2i = json.load(open(f"{U}/entity2id.json")); cm = {int(k): v for k, v in json.load(open(f"{U}/crslab_id.json")).items()}
    smp, off = [], 0
    for l in open(f"{U}/test_data_dbpedia.jsonl"):
        turns = json.loads(l); first = True
        for t, tu in enumerate(turns):
            if first and t > 0 and tu["role"] != turns[0]["role"]: first = False
            if first: continue
            if tu["role"] == "RECOMMENDER":
                for m in tu["movie_link"]:
                    if m in e2i: smp.append((off + t, e2i[m]))
        off += len(turns)
    assert [cm[x[1]] for x in smp] == [x["item"] for x in S["test"]], "ReDial UniCRS alignment"
    ur = []
    for sd in range(3):
        dd = {}
        for l in open(f"{D_}/unicrs_redialorig_cl_s{sd}.dump"):
            ln, lab, top = json.loads(l); dd[(ln, lab)] = top
        ur.append(np.array([(dd[x].index(x[1]) + 1) if x[1] in dd[x] else 10 ** 6 for x in smp], float))
    BASE["UniCRS"] = ur
else:
    U = f"{H}/projects/crs/baselines/unicrs/src/data/tgredial_uccr"; cid = {int(k): v for k, v in json.load(open(f"{U}/crslab_id.json")).items()}
    ur = []
    for s in range(3):
        rows = [json.loads(l) for l in open(f"{D_}/unicrs_tgredial_uccr_s{s}.dump")]; rows.sort(key=lambda r: r[0])
        assert [cid.get(r[1]) for r in rows] == [x["item"] for x in S["test"]], "TG UniCRS order"
        ur.append(np.array([r[2].index(r[1]) + 1 if r[1] in r[2] else 10 ** 6 for r in rows], float))
    BASE["UniCRS"] = ur
    BASE["UCCR"] = [dump_ranks(f"{D_}/uccr_tg_s{s}.jsonl") for s in range(3)]
D = "cuda"; dev = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=D)
rzt = lambda A: (A - A.mean(1, keepdim=True)) / (A.std(1, keepdim=True) + 1e-9)
Vt = dev(Vi); Yt = dev(Yf); ZP = dev(((POP - POP.mean()) / POP.std()).astype(np.float32)); QT = dev(Q["train"]).half()
M = torch.zeros(NI, len(S["train"]), dtype=torch.float16, device=D)
for n, x in enumerate(S["train"]):
    if x["item"] in pos: M[pos[x["item"]], n] = 1
has_ep = M.float().sum(1) > 0
gold = {sp: np.array([pos.get(x["item"], -1) for x in S[sp]]) for sp in ("valid", "test")}
TAUS = [0.01, 0.02, 0.05, 0.1]; GG = [0.5, 1, 2, 3, 4, 6]; MAXS = {}
@torch.no_grad()
def comps(sp):
    q = dev(Q[sp]); B = rzt(q @ Vt.T); C1 = rzt(dev(P[sp]) @ Yt.T); EP = {}
    sm = (q.half() @ QT.T).float(); MAXS[sp] = sm.max(1).values.cpu().numpy(); mx = sm.max(1, keepdim=True).values
    for t in TAUS:
        ep = torch.log((M @ torch.exp((sm - mx) / t).half().T).T.float() + 1e-30)
        mn = torch.where(has_ep[None], ep, torch.full_like(ep, float("inf"))).min(1, keepdim=True).values
        EP[t] = rzt(torch.where(has_ep[None], ep, mn))
    return B, C1, EP
FE = {sp: comps(sp) for sp in ("valid", "test")}; log("built")
@torch.no_grad()
def rank_from(Sc, sp):
    gi = torch.as_tensor(gold[sp], device=D); gs = Sc.gather(1, gi.clamp(min=0)[:, None]); r = (Sc > gs).sum(1).float() + 1
    return torch.where(gi >= 0, r, torch.full_like(r, 1e6)).cpu().numpy()
def a1(sp, gp, g1): B, C1, _ = FE[sp]; return B + gp * ZP[None] + g1 * C1
crit = lambda r: np.mean(r <= 1) + np.mean(r <= 50)
gp, g1 = max([(a, b) for a in [0, 0.25, 0.5, 0.75, 1.0, 1.5] for b in [0, 0.1, 0.2, 0.3, 0.5]], key=lambda g: crit(rank_from(a1("valid", *g), "valid")))
log("A1 gammas", gp, g1)
nd = lambda r: np.where(r <= 10, 1 / np.log2(r + 1), 0.0)
@torch.no_grad()
def additive(sp, t, g): return rank_from(a1(sp, gp, g1) + g * FE[sp][2][t], sp)
@torch.no_grad()
def union(sp, t, g, K=100):
    A = a1(sp, gp, g1); E = FE[sp][2][t]
    cand = torch.zeros_like(A, dtype=torch.bool); cand.scatter_(1, torch.topk(A, K, 1).indices, True); cand.scatter_(1, torch.topk(E, K, 1).indices, True)
    n_ = cand.sum(1, keepdim=True).float()
    def zl(Xx):
        mu = torch.where(cand, Xx, 0 * Xx).sum(1, keepdim=True) / n_
        return (Xx - mu) / ((torch.where(cand, (Xx - mu) ** 2, 0 * Xx).sum(1, keepdim=True) / n_).sqrt() + 1e-9)
    return rank_from(torch.where(cand, zl(A) + g * zl(E) + 1e4, A), sp)
RK, RES = {"A1": rank_from(a1("test", gp, g1), "test")}, {}
for name, fn in (("A1+RP", additive), ("A1+RP+union", union)):
    vals = {(t, g): float(nd(fn("valid", t, g)).mean()) for t in TAUS for g in GG}; best = max(vals, key=vals.get)
    RK[name] = fn("test", *best); RES[name] = dict(params=list(best), valid=round(vals[best], 4))
mt = lambda r: dict(NDCG10=round(float(nd(r).mean()), 4), HR10=round(float((r <= 10).mean()), 4), HR50=round(float((r <= 50).mean()), 4))
uu, inv = np.unique(CL, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    Sd = np.bincount(inv, d, minlength=len(uu)); Cn = np.bincount(inv, minlength=len(uu)); e = [Sd[k].sum() / Cn[k].sum() for k in BI]
    return [round(float(d.mean()), 4), round(float(np.percentile(e, 2.5)), 4), round(float(np.percentile(e, 97.5)), 4)]
OUT = {"dataset": DS, "n_train": len(S["train"]), "n_test": len(S["test"]), "n_test_convs": len(uu), "items": NI,
       "median_max_sim_test_to_train": round(float(np.median(MAXS["test"])), 4), "A1_gammas": [gp, g1], "selected": RES,
       "ours": {k: mt(v) for k, v in RK.items()}, "baselines": {}, "contrasts_ndcg10": {}}
for b, rs in BASE.items():
    OUT["baselines"][b] = dict(n_seeds=len(rs), **{k: round(float(np.mean([f(r) for r in rs])), 4) for k, f in
                               (("NDCG10", lambda r: nd(r).mean()), ("HR10", lambda r: (r <= 10).mean()), ("HR50", lambda r: (r <= 50).mean()))})
    bnd = np.mean([nd(r) for r in rs], 0)
    for k in ("A1", "A1+RP", "A1+RP+union"): OUT["contrasts_ndcg10"][f"{k} - {b}"] = boot(nd(RK[k]) - bnd)
for k in ("A1+RP", "A1+RP+union"): OUT["contrasts_ndcg10"][f"{k} - A1"] = boot(nd(RK[k]) - nd(RK["A1"]))
mx = MAXS["test"]; qs = np.quantile(mx, [0, .25, .5, .75, 1])
OUT["gain_by_maxsim_quartile"] = [round(float((nd(RK["A1+RP"]) - nd(RK["A1"]))[(mx >= qs[i]) & (mx <= qs[i + 1])].mean()), 4) for i in range(4)]
mrr = lambda r: np.where(r <= 10, 1 / r, 0.0)
OUT["table_metrics@10"] = {k: dict(HR=round(float((r <= 10).mean()), 4), MRR=round(float(mrr(r).mean()), 4), NDCG=round(float(nd(r).mean()), 4)) for k, r in RK.items()}
for b, rs in BASE.items():
    OUT["table_metrics@10"][b] = dict(n_seeds=len(rs), HR=round(float(np.mean([(r <= 10).mean() for r in rs])), 4), MRR=round(float(np.mean([mrr(r).mean() for r in rs])), 4), NDCG=round(float(np.mean([nd(r).mean() for r in rs])), 4))
np.savez_compressed(f"{W}/results/fullcat_{DS}_ranks.npz", conv=CL, **{k.replace("+", "__"): v for k, v in RK.items()})
json.dump(OUT, open(f"{W}/results/fullcat_{DS}.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("SMALL_DONE")
