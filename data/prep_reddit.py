"""Reddit-Movie large -> request-level evaluation units with per-seeker history (design fixed before experiments).
Unit = one request thread (He et al. CIKM'23 first recommender turn): context = seeker's opening post;
gold = items recommended in first-level replies (turn_order 1, non-seeker), over all paths of the thread,
minus items already in the opening post (repeat removal as in the paper's main protocol).
History of a request = the same seeker's earlier threads (utc strictly earlier, any split; splits are chronological):
opening text + its items, recommended items, and the seeker's own later replies in that thread (feedback text + items).
Text keeps both raw (titles) and processed (IMDb ids). Every drop is counted."""
import csv, ast, json, re, collections, sys
csv.field_size_limit(10 ** 9)
D = "/path/to/base/www27/data/reddit_movie"
OUT = f"{D}/prep_large"; import os; os.makedirs(OUT, exist_ok=True)
ML = "/path/to/base/datasets/movielens/ml-20m"
imdb2ml = {}
for r in csv.DictReader(open(f"{ML}/links.csv")): imdb2ml["tt" + r["imdbId"].zfill(7)] = int(r["movieId"])
gmov = set()
with open(f"{ML}/genome-scores.csv") as f:
    next(f); last = None
    for line in f:
        m = line.split(",", 1)[0]
        if m != last: gmov.add(int(m)); last = m
TT = re.compile(r"tt\d{7,8}")
C = collections.Counter()
th = {}  # thread -> dict
for sp in ("train", "valid", "test"):
    for r in csv.DictReader(open(f"{D}/large/{sp}.csv")):
        C["rows"] += 1
        try: role, ptxt = ast.literal_eval(r["processed"]); _, rtxt = ast.literal_eval(r["raw"])
        except Exception: C["parse_fail"] += 1; continue
        t = r["conv_id"].rsplit("_", 1)[0]
        d = th.setdefault(t, dict(thread=t, split=sp, seeker=None, utc=None, text=None, raw=None, items=[], rec=collections.Counter(), fb=[], seen_turns=set()))
        if r["turn_id"] in d["seen_turns"]: continue        # same turn repeated across reply paths
        d["seen_turns"].add(r["turn_id"]); o = int(r["turn_order"]); seek = r["is_seeker"] == "True"
        its = TT.findall(ptxt)
        if o == 0:
            d.update(seeker=r["user_id"] or None, utc=int(r["utc_time"]), text=ptxt, raw=rtxt, items=its)
        elif seek:
            d["fb"].append(dict(utc=int(r["utc_time"]), text=ptxt, raw=rtxt, items=its))
        elif o == 1:
            d["rec"].update(set(its))
units = collections.defaultdict(list); byseeker = collections.defaultdict(list)
for t, d in th.items():
    if d["text"] is None: C["thread_no_opening"] += 1; continue
    if not d["seeker"] or d["seeker"] in ("[deleted]", "None"): C["thread_no_seeker"] += 1; continue
    byseeker[d["seeker"]].append(t)
for u in byseeker: byseeker[u].sort(key=lambda t: th[t]["utc"])
genome = lambda i: imdb2ml.get(i) in gmov
hist_out = {}
for u, ts in byseeker.items():
    for k, t in enumerate(ts):
        d = th[t]; ctx = set(d["items"]); gold = sorted(i for i in d["rec"] if i not in ctx)
        hist_out[t] = dict(thread=t, utc=d["utc"], text=d["text"], raw=d["raw"], items=d["items"], rec=sorted(d["rec"]), fb=d["fb"])
        if not gold: C[f"unit_no_gold_{d['split']}"] += 1; continue
        prior = [p for p in ts[:k] if th[p]["utc"] < d["utc"]]
        units[d["split"]].append(dict(req=t, user=u, utc=d["utc"], text=d["text"], raw=d["raw"], ctx_items=sorted(ctx), gold=gold,
                                      gold_votes=[d["rec"][g] for g in gold], hist=prior))
for sp, L in units.items():
    with open(f"{OUT}/requests_{sp}.jsonl", "w") as f:
        for x in L: f.write(json.dumps(x) + "\n")
    g = [i for x in L for i in x["gold"]]
    C[f"units_{sp}"] = len(L); C[f"units_{sp}_with_hist"] = sum(1 for x in L if x["hist"]); C[f"gold_{sp}"] = len(g)
    C[f"gold_{sp}_genome_pct"] = round(100 * sum(map(genome, g)) / max(1, len(g)), 1)
    C[f"gold_per_unit_{sp}_median"] = sorted(len(x["gold"]) for x in L)[len(L) // 2]
    hg = [i for x in L if x["hist"] for i in x["gold"]]
    hitems = [set(i for p in x["hist"] for i in th[p]["items"] + list(th[p]["rec"])) for x in L if x["hist"]]
    C[f"gold_{sp}_in_own_history_pct"] = round(100 * sum(g in h for x, h in zip([x for x in L if x["hist"]], hitems) for g in x["gold"]) / max(1, len(hg)), 1)
json.dump(hist_out, open(f"{OUT}/threads.json", "w"))
items = sorted({i for d in th.values() for i in d["items"] + list(d["rec"]) + [j for fb in d["fb"] for j in fb["items"]]})
json.dump(dict(items=items, genome=[genome(i) for i in items], ml=[imdb2ml.get(i) for i in items]), open(f"{OUT}/items.json", "w"))
C["items"] = len(items); C["items_genome"] = sum(map(genome, items)); C["seekers"] = len(byseeker); C["threads"] = len(th)
json.dump(dict(C), open(f"{OUT}/stats.json", "w"), indent=1); print(json.dumps(dict(C), indent=1))
x = units["valid"][0]; print("FIRST valid unit:", {k: (v if k not in ("text", "raw") else v[:150]) for k, v in x.items()})
