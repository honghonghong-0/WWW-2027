"""Zero-shot LLM baseline (Qwen2.5-7B-Instruct, greedy) on CRSLab Inspired / TG-ReDial, on EXACTLY the baselines' samples (exported,
order verified). One generation per unique context (CRSLab repeats a context once per gold movie). The model lists 20 movies;
titles are matched to the item pool (names with parenthetical descriptions stripped; exact, then rapidfuzz ratio >= 88).
Rank of the label = its position in the matched list; labels not in the list get rank 10**6 (worse than any listed item).
Writes dump rows [epoch, mode, rank, ctx_repeat, label, conv_id] (valid epoch 0, test 'final') -> score_dumps.py / boot_compare.py.
Counts unmatched lines. usage: llm_zs_crslab.py <inspired|tgredial>"""
import sys, json, re, os, collections
from rapidfuzz import process, fuzz
from vllm import LLM, SamplingParams
DS = sys.argv[1]; W = "/path/to/base/www27"; X = f"{W}/data/crslab_export/{DS}"
C = "/path/to/base/projects/crs/baselines/crslab/data/dataset"
ITEMS = json.load(open(f"{C}/{DS}/{'nltk' if DS == 'inspired' else 'pkuseg'}/movie_ids.json")); i2e = json.load(open(f"{X}/vocab.json"))["id2entity"]
def clean(n):
    n = re.sub(r"^<http://dbpedia.org/resource/|>$", "", n).replace("_", " ")
    return re.sub(r"\s*[（(][^)）]*[)）]\s*", " ", n).strip()
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
cat = {}
for e in ITEMS:
    n = i2e.get(str(e), "")
    if n: cat.setdefault(norm(clean(n)), e)
ck = list(cat)
def match(line):
    t = re.sub(r"^\s*\d+[\.\)、．]\s*", "", line).strip().strip("*").strip()
    t = re.sub(r"\s*[（(]\d{4}[)）]\s*$", "", t)            # drop trailing (Year)
    t = re.sub(r"^[《\"]|[》\"]$", "", t).strip(); k = norm(t)
    if not k: return None
    if k in cat: return cat[k]
    m = process.extractOne(k, ck, scorer=fuzz.ratio, score_cutoff=88)
    return cat[m[0]] if m else None
S = {sp: [json.loads(l) for l in open(f"{X}/{sp}.jsonl")] for sp in ("valid", "test")}
dump = {"inspired": "kbrd_insp_s0", "tgredial": "kbrd_tg_s0"}[DS]
rows = [json.loads(l) for l in open(f"{W}/results/dumps/{dump}.jsonl")]
cid = {sp: [r[5] for r in rows if r[0] == ep and r[1] in md] for sp, ep, md in (("valid", 0, ("valid", "val")), ("test", "final", ("test",)))}
llm = LLM("Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GMU", "0.5")), seed=0)
tok = llm.get_tokenizer()
def prompt(ctx):
    conv = "\n".join(u for u in ctx if u.strip())[-3000:]
    if DS == "tgredial":
        q = f"以下是一段电影推荐对话：\n{conv}\n\n请为用户推荐20部电影。只输出编号列表，每行一部电影的中文片名，不要其他内容。"
    else:
        q = f"Here is a movie recommendation conversation:\n{conv}\n\nRecommend 20 movies for the user. Answer only with a numbered list, one movie per line, in the format: Title (Year)."
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
Cn = collections.Counter(); out = open(f"{W}/results/dumps/qwen_zs_{'insp' if DS == 'inspired' else 'tg'}.jsonl", "w")
for sp, ep in (("valid", 0), ("test", "final")):
    keys = list(dict.fromkeys("␞".join(x["ctx"]) for x in S[sp])); kpos = {k: n for n, k in enumerate(keys)}
    gens = llm.generate([prompt(k.split("␞")) for k in keys], SamplingParams(temperature=0, max_tokens=500))
    lists = []
    for g in gens:
        ids = []
        for l in g.outputs[0].text.split("\n"):
            if not l.strip(): continue
            m = match(l); Cn[f"{sp}_lines"] += 1
            if m is None: Cn[f"{sp}_unmatched"] += 1
            elif m not in ids: ids.append(m)
        lists.append(ids)
    for n, x in enumerate(S[sp]):
        ids = lists[kpos["␞".join(x["ctx"])]]; r = ids.index(x["item"]) + 1 if x["item"] in ids else 10 ** 6
        out.write(json.dumps([ep, sp, r, int(x["item"] in set(x["ctx_entities"])), x["item"], cid[sp][n]]) + "\n")
    Cn[f"{sp}_contexts"] = len(keys)
    if sp == "test": print("EXAMPLE:", gens[0].outputs[0].text[:300].replace("\n", " | "), flush=True)
out.close(); print(json.dumps(dict(Cn))); print("DONE")
