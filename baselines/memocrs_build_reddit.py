"""Build MemoCRS-dagger inputs for Reddit-Movie TEST (runs on H100; output shipped to the lab server).
Candidates: UCCR top-40 of the same seed (fixed .top dump, ctx items removed, in-catalog titles only).
Per request: current request text, the user's strictly-earlier threads (hist) with their own follow-ups (memory source).
Guideline-induction set: 40 TRAIN requests, candidates = gold (<=5) + top popular train items (fill to 40), shuffled."""
import json, sys, random, collections
B = "/path/to/base/www27"; P = f"{B}/data/reddit_movie/prep_large"
seed = sys.argv[1]
names = json.load(open(f"{B}/data/reddit_movie/large/id2name.json")); TH = json.load(open(f"{P}/threads.json"))
E = json.load(open("/path/to/base/projects/crs/baselines/crslab/data/dataset/reddit/nltk/entity2id.json")); i2e = {v: k for k, v in E.items()}
def fix(s):
    if "â" not in s and "Ã" not in s: return s
    for e in ("cp1252", "latin-1"):
        try: return s.encode(e).decode("utf-8")
        except Exception: pass
    return s
TOP = {}
for l in open(f"{B}/results/dumps/uccr_reddit_ep1_s{seed}.fixed.top"):
    cid, lab, top = json.loads(l); TOP.setdefault(cid, top)
inv = {v: k for k, v in json.load(open("/path/to/base/projects/crs/external/UCCR_infer/data/dataset/reddit/nltk/WWW27_id_maps.json"))["conv"].items()}
TOP = {inv.get(k, k): v for k, v in TOP.items()}
out = open(f"{B}/results/memocrs/reddit_test_s{seed}_inputs.jsonl", "w"); need = set(); short = 0; miss = 0
for l in open(f"{P}/requests_test.jsonl"):
    x = json.loads(l); t = TOP.get(x["req"])
    if t is None: miss += 1; t = []
    ctx = set(x["ctx_items"]); c = []
    for e in t:
        i = i2e.get(e, e)
        if i in ctx or i not in names or i in c: continue
        c.append(i)
        if len(c) == 40: break
    short += len(c) < 40; need.update(h for h in x["hist"] if h in TH)
    out.write(json.dumps({"req": x["req"], "user": x["user"], "utc": x["utc"], "text": fix(x["raw"]), "hist": [h for h in x["hist"] if h in TH],
                          "cands": [[i, fix(names[i])] for i in c]}) + "\n")
out.close()
json.dump({h: {"utc": TH[h]["utc"], "text": fix(TH[h]["raw"]), "fb": [fix(f["raw"]) for f in TH[h]["fb"][:3]]} for h in need},
          open(f"{B}/results/memocrs/reddit_hist_threads.json", "w"))
# guideline induction set (TRAIN only)
tr = [json.loads(l) for l in open(f"{P}/requests_train.jsonl")]; pop = collections.Counter(g for x in tr for g in x["gold"])
popl = [i for i, _ in pop.most_common(200) if i in names]; rng = random.Random(0); gi = []
for x in rng.sample([x for x in tr if any(g in names for g in x["gold"])], 40):
    g = [i for i in x["gold"] if i in names][:5]; c = g + [i for i in popl if i not in g and i not in x["ctx_items"]][:40 - len(g)]; rng.shuffle(c)
    gi.append({"text": fix(x["raw"]), "cands": [[i, fix(names[i])] for i in c], "gold": g})
json.dump(gi, open(f"{B}/results/memocrs/reddit_guideline_set.json", "w"))
print("seed", seed, "missing", miss, "short(<40)", short, "hist threads", len(need))
