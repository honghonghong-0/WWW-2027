"""Zero-shot LLM CRS baseline on Reddit-Movie large (He et al. CIKM'23 protocol, open model instead of GPT).
The model reads the request (optionally the same seeker's 5 most recent earlier requests + their own follow-ups) and lists 20 movies;
titles are fuzzy-matched to the catalog (Phi_1: in-catalog titles only), context items removed, rank = list order.
Metrics match reddit_ladder.py (multi-gold Recall/NDCG/HIT@10; warm/cold; warm-unseen). Parse failures are counted.
usage: llm_zs_reddit.py <split> <none|hist> [limit]"""
import json, re, sys, os, collections, numpy as np
from rapidfuzz import process, fuzz
from vllm import LLM, SamplingParams
SP, MODE = sys.argv[1], sys.argv[2]; LIM = int(sys.argv[3]) if len(sys.argv) > 3 else None
B = "/path/to/base/www27"; P = f"{B}/data/reddit_movie/prep_large"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TH = json.load(open(f"{P}/threads.json")); names = json.load(open(f"{B}/data/reddit_movie/large/id2name.json"))
L = [json.loads(l) for l in open(f"{P}/requests_{SP}.jsonl")][:LIM]
def fix(s):
    if "â" not in s and "Ã" not in s: return s
    for e in ("cp1252", "latin-1"):
        try: return s.encode(e).decode("utf-8")
        except Exception: pass
    return s
clip = lambda s, n: s if len(s) <= n else s[:n] + " ..."
def prompt(x):
    h = ""
    if MODE == "hist" and x["hist"]:
        parts = []
        for i, t in enumerate(x["hist"][-5:]):
            d = TH[t]; fb = "".join("\n  (later, the user said) " + clip(fix(f["raw"]), 300) for f in d["fb"][:3])
            parts.append(f"Past request {i+1}: " + clip(fix(d["raw"]), 700) + fb)
        h = "Earlier requests from the same user:\n" + "\n".join(parts) + "\n\n"
    return (h + "Current request:\n" + clip(fix(x["raw"]), 2500) +
            "\n\nRecommend 20 movies for the current request. Answer only with a numbered list, one movie per line, in the format: Title (Year).")
norm = lambda s: re.sub(r"[^a-z0-9 ()]", "", s.lower()).strip()
cat_keys = {};
for i, n in names.items(): cat_keys.setdefault(norm(n), i)
ck = list(cat_keys)
def match(line):
    t = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip().strip("*").strip()
    if not t: return None
    k = norm(t)
    if k in cat_keys: return cat_keys[k]
    m = process.extractOne(k, ck, scorer=fuzz.ratio, score_cutoff=88)
    return cat_keys[m[0]] if m else None
llm = LLM(MODEL, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=float(os.environ.get("GMU", "0.85")), seed=0)
tok = llm.get_tokenizer()
msgs = [tok.apply_chat_template([{"role": "user", "content": prompt(x)}], tokenize=False, add_generation_prompt=True) for x in L]
outs = llm.generate(msgs, SamplingParams(temperature=0, max_tokens=400))
C = collections.Counter(); rows = []
for x, o in zip(L, outs):
    lines = [l for l in o.outputs[0].text.split("\n") if l.strip()]
    ids = []
    for l in lines:
        i = match(l); C["lines"] += 1
        if i is None: C["unmatched"] += 1
        elif i not in ids: ids.append(i)
    ctx = set(x["ctx_items"]); ids = [i for i in ids if i not in ctx]
    hseen = {j for t in x["hist"] for j in TH[t]["items"] + TH[t]["rec"] + [k for f in TH[t]["fb"] for k in f["items"]]}
    if not ids: C["empty_list"] += 1
    rows.append(dict(req=x["req"], user=x["user"], warm=bool(x["hist"]), ranked=ids[:20], gold=x["gold"], hseen=sorted(hseen)))
def met(ranked, gold):
    gs = set(gold)
    if not gs: return None
    top = ranked[:10]; rel = [t in gs for t in top]
    dcg = sum(1 / np.log2(k + 2) for k, r in enumerate(rel) if r); idcg = sum(1 / np.log2(k + 2) for k in range(min(10, len(gs))))
    return (sum(rel) / len(gs), dcg / idcg, float(any(rel)))
def agg(sel, unseen=False):
    v = []
    for r in sel:
        if unseen:
            hs = set(r["hseen"]); m = met([i for i in r["ranked"] if i not in hs], [g for g in r["gold"] if g not in hs])
        else: m = met(r["ranked"], r["gold"])
        if m: v.append(m)
    return [round(float(a), 4) for a in np.mean(v, 0)] + [len(v)] if v else None
res = dict(model=MODEL, mode=MODE, split=SP, n=len(rows), counts=dict(C), all=agg(rows), warm=agg([r for r in rows if r["warm"]]),
           cold=agg([r for r in rows if not r["warm"]]), warm_unseen=agg([r for r in rows if r["warm"]], unseen=True))
tag = f"{SP}_{MODE}" + (f"_lim{LIM}" if LIM else "")
json.dump(res, open(f"{B}/results/llm_zs_reddit_{tag}.json", "w"), indent=1)
with open(f"{B}/results/llm_zs_reddit_{tag}_ranked.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(res)); print("EXAMPLE OUTPUT:\n", outs[0].outputs[0].text[:400])
