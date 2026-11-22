"""Our method on CRSLab Inspired / TG-ReDial, scored on EXACTLY the baselines' samples and item pool (order verified against the
KBRD dumps). Same ladder as Reddit: score = z(base) + g_pop z(pop) + g1 z(c1)
  base: Qwen3-Embedding-8B cosine(context (same instruction as Reddit), item name)
  pop : log1p #label occurrences in TRAIN samples
  c1  : p_C . y_i ; p_C = ridge(context emb -> y of label) fit on TRAIN samples ; y_i = Tag Genome experience profile (centered)
        when the item links to the genome by normalized title(+year), else ridge-filled from the item-name embedding
        (filler fit on Reddit-Movie genome-linked items, English titles; for TG the names are Chinese -> filler is cross-lingual, weak)
Gammas: chosen on VALID by HR@1+HR@50 (the baselines' early-stop criterion), applied to TEST once. No context masking (as CRSLab).
Writes dump rows [epoch, mode, rank, ctx_repeat, label, conv_id] (valid epoch 0, test 'final') -> score with score_dumps.py.
usage: ladder_crslab.py <inspired|tgredial>"""
import sys, json, re, os, numpy as np, torch
DS = sys.argv[1]; W = "/path/to/base/www27"; X = f"{W}/data/crslab_export/{DS}"
C = "/path/to/base/projects/crs/baselines/crslab/data/dataset"
IC = "/path/to/base/projects/crs/processed/item_chain"
GEN = "/path/to/base/datasets/tag_genome_2021/unzip/movie_dataset_public_final/raw/metadata.json"
ITEMS = json.load(open(f"{C}/{DS}/{'nltk' if DS == 'inspired' else 'pkuseg'}/movie_ids.json")); pos = {e: k for k, e in enumerate(ITEMS)}
i2e = json.load(open(f"{X}/vocab.json"))["id2entity"]
S = {sp: [json.loads(l) for l in open(f"{X}/{sp}.jsonl")] for sp in ("train", "valid", "test")}
def iname(e):
    n = i2e.get(str(e), ""); n = re.sub(r"^<http://dbpedia.org/resource/|>$", "", n).replace("_", " ")
    return n
names = [iname(e) for e in ITEMS]
nl = {e for e, n in zip(ITEMS, names) if not n}
print("pool items without a name (kept in pool, empty text):", len(nl), "| test labels on them:", sum(x["item"] in nl for x in S["test"]), flush=True)
names = [n if n else " " for n in names]
if os.environ.get("ITEM_TEXT"):   # item text = name + TRAIN utterances mentioning the movie (+ next utterance); train split only (no test leakage)
    import collections
    raw = json.load(open(f"{C}/{DS}/{'nltk' if DS == 'inspired' else 'pkuseg'}/train_data.json")); e2i_ = {v: k for k, v in i2e.items()}
    tx = collections.defaultdict(list); sep = " " if DS == "inspired" else ""
    for c in raw:
        ms = c["messages"] if "messages" in c else c["dialog"]
        for k, m in enumerate(ms):
            for mv in (m.get("movie") or m.get("movies") or []):
                tx[mv].append(sep.join(m["text"]))
                if k + 1 < len(ms): tx[mv].append(sep.join(ms[k + 1]["text"]))
    def key(e): return i2e.get(str(e), "")
    got = 0
    for k, e in enumerate(ITEMS):
        t = tx.get(key(e)) or tx.get(iname(e))
        if t: names[k] = (names[k] + "\n" + "\n".join(dict.fromkeys(t)))[:1500]; got += 1
    print("item text from train utterances:", got, "of", len(ITEMS), flush=True)
# ---- genome link by normalized title (+year if the name carries one)
tn = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", re.sub(r"^(the|a|an) ", "", re.sub(r"\(.*?\)", "", s.lower()).strip())).split())
Gz = np.load(f"{IC}/item_chain_v1.npz"); grow = {int(m): r for r, m in enumerate(Gz["items"])}
byt = {}
for l in open(GEN):
    m = json.loads(l); y = re.search(r"\((\d{4})\)\s*$", m["title"]); k = tn(m["title"])
    if m["item_id"] in grow: byt.setdefault(k, []).append((y.group(1) if y else None, grow[m["item_id"]]))
link = np.full(len(ITEMS), -1)
for k, n in enumerate(names):
    c = byt.get(tn(n), []); y = re.search(r"(\d{4})", n)
    if y: c2 = [r for yy, r in c if yy == y.group(1)]; c = [(None, r) for r in c2] if c2 else (c if len(c) == 1 else [])
    if len(c) == 1: link[k] = c[0][1]
print(DS, "items", len(ITEMS), "genome-linked", int((link >= 0).sum()), f"({(link >= 0).mean():.3f})", flush=True)
# ---- embeddings
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("Qwen/Qwen3-Embedding-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda"); enc.max_seq_length = 2048
INST = ("Instruct: Given a movie-seeking conversation (with optional past sessions of the same user), "
        "retrieve the movie the user wants next\nQuery: ")
def E(texts, query, bs=32):
    order = np.argsort([-len(t) for t in texts]); out = np.zeros((len(texts), 4096), np.float32)
    out[order] = enc.encode([texts[i] for i in order], prompt=INST if query else None, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return out
clip = lambda s, n: s if len(s) <= n else s[-n:]   # keep the most recent part of the context
ctxt = lambda x: clip("\n".join(u for u in x["ctx"] if u.strip()), 3000)
_cf = f"{W}/results/emb_cache_{DS}{'_itx' if os.environ.get('ITEM_TEXT') else ''}.npz"   # embedding cache (same inputs -> same vectors)
if os.path.exists(_cf):
    _z = np.load(_cf); Vi = _z["Vi"]; Q = {sp: _z["Q_" + sp] for sp in S}; print("embeddings from cache", _cf, flush=True)
else:
    Vi = E(names, False, int(os.environ.get("ITEM_BS", 256))); Q = {sp: E([ctxt(x) for x in S[sp]], True) for sp in S}; print("encoded", flush=True)
    del enc; torch.cuda.empty_cache()
    np.savez(_cf, Vi=Vi, **{"Q_" + sp: Q[sp] for sp in S})
# ---- y_i: genome where linked, else filler from Reddit genome-linked English titles
def ridge(Xa, Ya, lam):
    Xb = np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]); I = np.eye(Xb.shape[1]); I[-1, -1] = 0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ Ya)
pred = lambda Wt, Xa: np.hstack([Xa, np.ones((len(Xa), 1), np.float32)]) @ Wt
Z = np.load(f"{W}/results/reddit_large_emb.npz"); IJ = json.load(open(f"{W}/data/reddit_movie/prep_large/items.json"))
rl = [(k, grow[int(m)]) for k, m in enumerate(IJ["ml"]) if m is not None and int(m) in grow]
ybar = Gz["Y"][[r for _, r in rl]].mean(0)
Wf = ridge(Z["V"][[k for k, _ in rl]].astype(np.float32), Gz["Y"][[r for _, r in rl]] - ybar, 10.0)
Yf = np.where((link >= 0)[:, None], Gz["Y"][np.maximum(link, 0)] - ybar, pred(Wf, Vi)).astype(np.float32)
TAG = os.environ.get("TAG", "")
if os.environ.get("TEXT_PROFILE"):   # TG: experience profile from in-dataset Chinese text (train utterances) via review->Y ridge
    tp = np.load(os.environ["TEXT_PROFILE"]); tpos = {int(e): k for k, e in enumerate(tp["items"])}
    use = np.array([e in tpos and link[k] < 0 for k, e in enumerate(ITEMS)])
    for k, e in enumerate(ITEMS):
        if use[k]: Yf[k] = tp["Y"][tpos[e]] + tp["ybar"] - ybar   # same centering as genome rows
    print("text-profile rows used", int(use.sum()), "| genome-linked rows kept", int((link >= 0).sum()), flush=True)
# ---- pop and p_C from TRAIN samples
POP = np.zeros(len(ITEMS))
for x in S["train"]:
    if x["item"] in pos: POP[pos[x["item"]]] += 1
POP = np.log1p(POP)
trk = [n for n, x in enumerate(S["train"]) if x["item"] in pos]
Gw = ridge(Q["train"][trk], Yf[[pos[S["train"][n]["item"]] for n in trk]], 10.0)
if os.environ.get("MFILE"):   # transfer: affordance map p_C = M^T [e;1] learned on Reddit with the ranking loss (pretraining)
    M_ = np.load(os.environ["MFILE"]).astype(np.float32); MIX = float(os.environ.get("MIX", "1"))
    Gw = MIX * M_ + (1 - MIX) * Gw; print("p_C map: Reddit-learned M, mix", MIX, flush=True)
zr = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
def scores(sp, gp, g1):
    B = zr(Q[sp] @ Vi.T); P = pred(Gw, Q[sp]); P -= P.mean(0); C1 = zr(P @ Yf.T); ZP = (POP - POP.mean()) / POP.std()
    return B + gp * ZP[None] + g1 * C1
def ranks(sp, Sc):
    out = []
    for n, x in enumerate(S[sp]):
        j = pos.get(x["item"]); out.append(None if j is None else int((Sc[n] > Sc[n, j]).sum()) + 1)
    return out
crit = lambda r: np.mean([v <= 1 for v in r if v]) + np.mean([v <= 50 for v in r if v])
GP = [0, 0.25, 0.5, 0.75, 1.0, 1.5]; G1 = [0, 0.1, 0.2, 0.3, 0.5]
cid = {}
dump = {"inspired": "kbrd_insp_s0", "tgredial": "kbrd_tg_s0"}[DS]
rows = [json.loads(l) for l in open(f"{W}/results/dumps/{dump}.jsonl")]
for sp, ep, md in (("valid", 0, ("valid", "val")), ("test", "final", ("test",))):
    cid[sp] = [r[5] for r in rows if r[0] == ep and r[1] in md]
    assert [r[4] for r in rows if r[0] == ep and r[1] in md] == [x["item"] for x in S[sp]], "sample order mismatch"
for var, grid in (("base", [(0, 0)]), ("base_pop", [(g, 0) for g in GP]), ("base_pop_c1", [(g, a) for g in GP for a in G1])):
    best = max(grid, key=lambda g: crit(ranks("valid", scores("valid", *g))))
    f = f"{W}/results/dumps/ours_{var}_{'insp' if DS == 'inspired' else 'tg'}{TAG}.jsonl"
    with open(f, "w") as h:
        for sp, ep in (("valid", 0), ("test", "final")):
            for n, (x, r) in enumerate(zip(S[sp], ranks(sp, scores(sp, *best)))):
                if r is None: continue
                h.write(json.dumps([ep, sp, r, int(x["item"] in set(x["ctx_entities"])), x["item"], cid[sp][n]]) + "\n")
    print(var, "gammas (valid-selected)", best, "->", f, flush=True)
print("DONE")
# ---------------- ranking-gap diagnostics (env DIAG=1) ----------------
# (1) within the base+pop top-50, does c1 put the gold above the other candidates? (AUC; 0.5 = no fine-grained information)
# (2) dynamic range: std inside top-50 / std over the whole pool, for c1 vs for base+pop
# (3) second-stage rerank of the top-50 with LOCALLY re-standardized c1 (gamma chosen on VALID HR@10), test once -> dump
if os.environ.get("DIAG"):
    gp = max([(g, 0) for g in GP], key=lambda g: crit(ranks("valid", scores("valid", *g))))[0]
    def comps(sp):
        B = zr(Q[sp] @ Vi.T); P = pred(Gw, Q[sp]); P -= P.mean(0); C1 = zr(P @ Yf.T); ZP = (POP - POP.mean()) / POP.std()
        return B + gp * ZP[None], C1
    DG = {}
    for sp in ("valid", "test"):
        BP, C1 = comps(sp); top = np.argsort(-BP, 1)[:, :50]; auc_c, auc_b, rc, rb, inn = [], [], [], [], 0
        for n, x in enumerate(S[sp]):
            j = pos.get(x["item"]); t = top[n]
            rc.append(C1[n, t].std() / (C1[n].std() + 1e-9)); rb.append(BP[n, t].std() / (BP[n].std() + 1e-9))
            if j is None or j not in set(t): continue
            inn += 1; o = t[t != j]
            auc_c.append(np.mean(C1[n, o] < C1[n, j]) + 0.5 * np.mean(C1[n, o] == C1[n, j])); auc_b.append(np.mean(BP[n, o] < BP[n, j]))
        DG[sp] = dict(gold_in_top50=round(inn / len(S[sp]), 4), auc_c1_within_top50=round(float(np.mean(auc_c)), 4), auc_basepop_within_top50=round(float(np.mean(auc_b)), 4),
                      std_ratio_c1=round(float(np.mean(rc)), 4), std_ratio_basepop=round(float(np.mean(rb)), 4))
        print("DIAG", sp, json.dumps(DG[sp]), flush=True)
    def local(sp, g):
        BP, C1 = comps(sp); top = np.argsort(-BP, 1)[:, :50]; R = []
        for n, x in enumerate(S[sp]):
            t = top[n]; j = pos.get(x["item"])
            if j is None: R.append(None); continue
            if j not in set(t): R.append(int((BP[n] > BP[n, j]).sum()) + 1); continue   # outside top-50: first-stage rank (unchanged)
            zl = lambda v: (v - v.mean()) / (v.std() + 1e-9); s2 = zl(BP[n, t]) + g * zl(C1[n, t]); k = list(t).index(j)
            R.append(int((s2 > s2[k]).sum()) + 1)
        return R
    hr10 = lambda r: np.mean([v <= 10 for v in r if v])
    gbest = max([0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], key=lambda g: hr10(local("valid", g)))
    f = f"{W}/results/dumps/ours_localc1_{'insp' if DS == 'inspired' else 'tg'}{TAG}.jsonl"
    with open(f, "w") as h:
        for sp, ep in (("valid", 0), ("test", "final")):
            for n, (x, r) in enumerate(zip(S[sp], local(sp, gbest))):
                if r is None: continue
                h.write(json.dumps([ep, sp, r, int(x["item"] in set(x["ctx_entities"])), x["item"], cid[sp][n]]) + "\n")
    print("DIAG local-rerank gamma (valid HR@10)", gbest, "->", f, flush=True)
