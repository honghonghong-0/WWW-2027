"""Collective episodic affordance on ReDial (original split, CRSLab samples) - second dataset for WWW'27 short. Fixed before running, 2026-11-17.
Same method as fullcat_episodic.py (Reddit):
   EP_s(i,C) = log sum_{c' in N(i)} exp(sim_s(C,c')/tau),  N(i) = TRAIN samples whose label is i  (past choice episodes of i)
A1 (ladder_crslab.py recipe, gammas on VALID by HR@1+HR@50 as there): z base + gp z pop + g1 z c1 over the full item pool.
Spaces: emb (request embedding cosine), prof (read Tag Genome profile p_C cosine; ReDial has one genome profile space, not the exp/attr split).
Score = A1 + g1 z(EP_emb) + g2 z(EP_prof); tau in {.02,.05}; g in {0,.5,1,1.5,2}; VALID NDCG@10 select, TEST once.
Protocol = CRSLab / baselines: one sample per recommended movie, no context-item masking, full pool. Paired vs UniCRS (3 seeds, mean per
sample, ranks from its saved top-100 via the verified unicrs_score_crslab alignment) and KBRD s0; conversation bootstrap 2000.
usage: python fullcat_redial.py   (env TAG)"""
import sys, json, re, os, time, numpy as np, torch
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
H = "/path/to/base"; W = f"{H}/00_Research/01_Recommender_Systems/WWW-2027"; DS = "redialorig"; X = f"{W}/data/crslab_export/{DS}"
C = f"{H}/projects/crs/baselines/crslab_orig/data/dataset/redial/nltk"
IC = f"{H}/projects/crs/processed/item_chain"
GEN = f"{H}/datasets/tag_genome_2021/unzip/movie_dataset_public_final/raw/metadata.json"
ITEMS = json.load(open(f"{C}/movie_ids.json")); pos = {e: k for k, e in enumerate(ITEMS)}; NI = len(ITEMS)
i2e = json.load(open(f"{X}/vocab.json"))["id2entity"]
S = {sp: [json.loads(l) for l in open(f"{X}/{sp}.jsonl")] for sp in ("train", "valid", "test")}
log("samples", {k: len(v) for k, v in S.items()}, "items", NI, "| test labels outside pool", sum(x["item"] not in pos for x in S["test"]))
def iname(e):
    return re.sub(r"^<http://dbpedia.org/resource/|>$", "", i2e.get(str(e), "")).replace("_", " ")
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
log("genome-linked", int((link >= 0).sum()), f"({(link >= 0).mean():.3f})")
INST = ("Instruct: Given a movie-seeking conversation (with optional past sessions of the same user), "
        "retrieve the movie the user wants next\nQuery: ")
clip = lambda s, n: s if len(s) <= n else s[-n:]
ctxt = lambda x: clip("\n".join(u for u in x["ctx"] if u.strip()), 3000)
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
    Vi = E(names, False, 256); Q = {}
    for sp in S: Q[sp] = E([ctxt(x) for x in S[sp]], True); log("encoded", sp)
    del enc; torch.cuda.empty_cache(); np.savez(_cf, Vi=Vi, **{"Q_" + sp: Q[sp] for sp in S})
def ridge(Xa, Ya, lam):
    Xb = np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]); I = np.eye(Xb.shape[1]); I[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ Ya)
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
# ---- sample order / conversation ids from the KBRD dump (as ladder_crslab.py), UniCRS + KBRD per-sample ranks
rows = [json.loads(l) for l in open(f"{W}/results/dumps/kbrd_redialorig_s0.jsonl")]
eps_t = max(r[0] for r in rows if r[1] == "test" and r[0] != "final") if any(r[1] == "test" and r[0] != "final" for r in rows) else "final"
cid, kbrd_rank = {}, {}
for sp, md in (("valid", ("valid", "val")), ("test", ("test",))):
    eps = [r[0] for r in rows if r[1] in md]; ep = max(eps, key=lambda e: (isinstance(e, str), e))
    rr = [r for r in rows if r[0] == ep and r[1] in md]
    assert [r[4] for r in rr] == [x["item"] for x in S[sp]], f"sample order mismatch {sp}"
    cid[sp] = np.array([r[5] if r[5] is not None else -1 for r in rr]); kbrd_rank[sp] = np.array([r[2] for r in rr], float)
log("KBRD order verified; test conv ids", len(set(cid["test"].tolist())))
U = f"{H}/projects/crs/baselines/unicrs/src/data/redialorig_cl"
e2i = json.load(open(f"{U}/entity2id.json")); cmap = {int(k): v for k, v in json.load(open(f"{U}/crslab_id.json")).items()}
samples, off = [], 0
for l in open(f"{U}/test_data_dbpedia.jsonl"):
    turns = json.loads(l); first = True
    for t, tu in enumerate(turns):
        if first and t > 0 and tu["role"] != turns[0]["role"]: first = False
        if first: continue
        if tu["role"] == "RECOMMENDER":
            for m in tu["movie_link"]:
                if m in e2i: samples.append((off + t, e2i[m]))
    off += len(turns)
assert [cmap[s[1]] for s in samples] == [x["item"] for x in S["test"]], "UniCRS alignment"
uni_rank = []
for sd in range(3):
    dump = {}
    for l in open(f"{W}/results/dumps/unicrs_redialorig_cl_s{sd}.dump"):
        ln, lab, top = json.loads(l); dump[(ln, lab)] = top
    uni_rank.append(np.array([(dump[s].index(s[1]) + 1) if s[1] in dump[s] else 10 ** 6 for s in samples], float))
ndcg_r = lambda r: np.where(r <= 10, 1 / np.log2(r + 1), 0.0)
uni_nd = np.mean([ndcg_r(r) for r in uni_rank], 0); uni_hr10 = np.mean([(r <= 10) for r in uni_rank], 0); uni_hr50 = np.mean([(r <= 50) for r in uni_rank], 0)
log("UniCRS mean NDCG@10", round(float(uni_nd.mean()), 4), "HR@10", round(float(uni_hr10.mean()), 4))
# ---- GPU scoring
D = "cuda"; dev = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=D)
rzt = lambda A: (A - A.mean(1, keepdim=True)) / (A.std(1, keepdim=True) + 1e-9)
nrm = lambda A: A / (A.norm(dim=1, keepdim=True) + 1e-9)
Vt = dev(Vi); Yt = dev(Yf); ZP = dev(((POP - POP.mean()) / POP.std()).astype(np.float32))
QT = dev(Q["train"]).half(); PT = nrm(dev(P["train"])).half()
M = torch.zeros(NI, len(S["train"]), dtype=torch.float16, device=D)
for n, x in enumerate(S["train"]):
    if x["item"] in pos: M[pos[x["item"]], n] = 1
has_ep = M.float().sum(1) > 0; log("items with >=1 train episode", int(has_ep.sum()))
gold = {sp: np.array([pos.get(x["item"], -1) for x in S[sp]]) for sp in ("valid", "test")}
TAUS = [0.02, 0.05]; GG = [0, 0.5, 1.0, 1.5, 2.0]
@torch.no_grad()
def comps(sp):
    q = dev(Q[sp]); B = rzt(q @ Vt.T); C1 = rzt(dev(P[sp]) @ Yt.T); EP = {}
    for s in ("emb", "prof"):
        for t in TAUS:
            out = torch.empty(len(S[sp]), NI, device=D)
            for b in range(0, len(S[sp]), 256):
                qq = q[b:b + 256].half() if s == "emb" else nrm(dev(P[sp][b:b + 256])).half()
                sm = (qq @ (QT if s == "emb" else PT).T).float(); mx = sm.max(1, keepdim=True).values
                ep = torch.log((M @ torch.exp((sm - mx) / t).half().T).T.float() + 1e-30)
                mn = torch.where(has_ep[None], ep, torch.full_like(ep, float("inf"))).min(1, keepdim=True).values
                out[b:b + 256] = rzt(torch.where(has_ep[None], ep, mn))
            EP[(s, t)] = out
    return B, C1, EP
FE = {sp: comps(sp) for sp in ("valid", "test")}; log("built")
@torch.no_grad()
def rank_of(sp, gp, g1, terms):
    B, C1, EP = FE[sp]; Sc = B + gp * ZP[None] + g1 * C1
    for (s, t, g) in terms:
        if g: Sc = Sc + g * EP[(s, t)]
    gi = torch.as_tensor(gold[sp], device=D); ok = gi >= 0
    gs = Sc.gather(1, gi.clamp(min=0)[:, None]); r = (Sc > gs).sum(1).float() + 1
    r = torch.where(ok, r, torch.full_like(r, 1e6)); return r.cpu().numpy()
crit = lambda r: np.mean(r <= 1) + np.mean(r <= 50)
GP = [0, 0.25, 0.5, 0.75, 1.0, 1.5]; G1 = [0, 0.1, 0.2, 0.3, 0.5]
gp, g1 = max([(a, b) for a in GP for b in G1], key=lambda g: crit(rank_of("valid", g[0], g[1], [])))
log("A1 gammas (valid HR@1+HR@50)", gp, g1)
SYS = {"A1": [], "A1+EP_emb": ["emb"], "A1+EP_prof": ["prof"], "A1+EP_emb+EP_prof": ["emb", "prof"]}
RES, RK = {}, {}
for k, sp_ in SYS.items():
    if not sp_: grid = [()]
    elif len(sp_) == 1: grid = [((sp_[0], t, g),) for t in TAUS for g in GG[1:]]
    else: grid = [(("emb", t1, a), ("prof", t2, b)) for t1 in TAUS for a in GG[1:] for t2 in TAUS for b in GG]
    vals = {p: float(ndcg_r(rank_of("valid", gp, g1, p)).mean()) for p in grid}; best = max(vals, key=vals.get)
    RK[k] = rank_of("test", gp, g1, best)
    RES[k] = dict(params=[list(x) for x in best], valid_ndcg10=round(vals[best], 4), **{m: round(float(f(RK[k]).mean()), 4) for m, f in
              (("HR@1", lambda r: r <= 1), ("HR@10", lambda r: r <= 10), ("HR@50", lambda r: r <= 50), ("NDCG@10", ndcg_r))})
    log(k, RES[k])
conv = cid["test"]; uu, inv = np.unique(conv, return_inverse=True); rb = np.random.default_rng(1); BI = [rb.integers(0, len(uu), len(uu)) for _ in range(2000)]
def boot(d):
    Sd = np.bincount(inv, d, minlength=len(uu)); Cn = np.bincount(inv, minlength=len(uu)); est = [Sd[k].sum() / Cn[k].sum() for k in BI]
    return [round(float(d.mean()), 4), round(float(np.percentile(est, 2.5)), 4), round(float(np.percentile(est, 97.5)), 4), len(d)]
kb = kbrd_rank["test"]
OUT = {"n_test": len(S["test"]), "A1_gammas": [gp, g1], "genome_linked": int((link >= 0).sum()), "systems": RES,
       "UniCRS_3seed": {"HR@1": round(float(np.mean([(r <= 1) for r in uni_rank])), 4), "HR@10": round(float(uni_hr10.mean()), 4), "HR@50": round(float(uni_hr50.mean()), 4), "NDCG@10": round(float(uni_nd.mean()), 4)},
       "KBRD_s0": {"HR@10": round(float((kb <= 10).mean()), 4), "HR@50": round(float((kb <= 50).mean()), 4), "NDCG@10": round(float(ndcg_r(kb).mean()), 4)},
       "contrasts_ndcg10": {}, "contrasts_hr10": {}}
for k in RK:
    OUT["contrasts_ndcg10"][f"{k} - UniCRS"] = boot(ndcg_r(RK[k]) - uni_nd); OUT["contrasts_hr10"][f"{k} - UniCRS"] = boot((RK[k] <= 10) - uni_hr10)
    if k != "A1": OUT["contrasts_ndcg10"][f"{k} - A1"] = boot(ndcg_r(RK[k]) - ndcg_r(RK["A1"]))
OUT["contrasts_ndcg10"]["EP_emb+EP_prof - EP_emb"] = boot(ndcg_r(RK["A1+EP_emb+EP_prof"]) - ndcg_r(RK["A1+EP_emb"]))
tag = os.environ.get("TAG", "redial")
json.dump(OUT, open(f"{W}/results/fullcat_{tag}.json", "w"), indent=1)
np.savez_compressed(f"{W}/results/fullcat_{tag}_ranks.npz", conv=conv, uni=np.array(uni_rank), kbrd=kb, **{k.replace("+", "__"): v for k, v in RK.items()})
print(json.dumps(OUT, indent=1)); print("REDIAL_DONE")
