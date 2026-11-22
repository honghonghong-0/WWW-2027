"""Stage 1 encoding for Reddit-Movie large (same encoder/instruction as the ReDial-MS p0 backbone, Qwen3-Embedding-8B).
Encodes: (a) every thread text (opening post + the seeker's own later replies, used as current context [opening only] and as history),
(b) history-concatenated queries for units with prior threads (most recent 5, newest last), (c) item titles (no instruction).
Mojibake (UTF-8 read as cp1252/latin-1) is repaired and counted. Output: results/reddit_large_emb.npz"""
import json, sys, numpy as np, torch
P = "/path/to/base/www27/data/reddit_movie/prep_large"
W = "/path/to/base/www27"
fixed = 0
def fix(s):
    global fixed
    if "â" not in s and "Ã" not in s: return s
    for enc in ("cp1252", "latin-1"):
        try: t = s.encode(enc).decode("utf-8"); fixed += 1; return t
        except Exception: pass
    return s
TH = json.load(open(f"{P}/threads.json")); items = json.load(open(f"{P}/items.json"))["items"]
names = json.load(open(f"{W}/data/reddit_movie/large/id2name.json"))
U = {sp: [json.loads(l) for l in open(f"{P}/requests_{sp}.jsonl")] for sp in ("train", "valid", "test")}
clip = lambda s, n: s if len(s) <= n else s[:n]
tids = sorted(TH); opening = [fix(TH[t]["raw"]) for t in tids]
hist_txt = {t: clip(fix(TH[t]["raw"]) + "".join("\nUser later: " + fix(f["raw"]) for f in TH[t]["fb"]), 1500) for t in tids}
hq_keys, hq = [], []
for sp, L in U.items():
    for x in L:
        if x["hist"]:
            parts = [f"[Past request {i+1}]\n" + hist_txt[h] for i, h in enumerate(x["hist"][-5:])]
            hq_keys.append(x["req"]); hq.append("\n\n".join(parts) + "\n\n[Current request]\n" + clip(fix(x["raw"]), 3000))
it_txt = [names.get(i, i) for i in items]
print("threads", len(tids), "hist queries", len(hq), "items", len(items), "mojibake fixed", fixed, "items without name", sum(1 for i in items if i not in names), flush=True)
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("Qwen/Qwen3-Embedding-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda"); enc.max_seq_length = 2048
INST = ("Instruct: Given a movie-seeking conversation (with optional past sessions of the same user), "
        "retrieve the movie the user wants next\nQuery: ")
def E(texts, query=True, bs=32):
    order = np.argsort([-len(t) for t in texts]); out = np.zeros((len(texts), 4096), np.float16)
    out[order] = enc.encode([texts[i] for i in order], prompt=INST if query else None, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return out
Ei = E(it_txt, query=False, bs=256); print("items done", flush=True)
Et = E([clip(t, 6000) for t in opening]); print("threads done", flush=True)
Eh = E(hq, bs=16); print("hist done", flush=True)
np.savez(f"{W}/results/reddit_large_emb.npz", items=np.array(items), V=Ei, tids=np.array(tids), T=Et, hq_keys=np.array(hq_keys), H=Eh)
print("saved", flush=True)
