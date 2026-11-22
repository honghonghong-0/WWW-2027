"""Reddit-Movie (large, our prep split) -> CRSLab ReDial-format dataset with a Wikidata KG, for KBRD/KGSF/UniCRS baselines.
Entities: every Reddit item (IMDb id, linked or not) + Wikidata objects of P57 director, P161 cast, P136 genre, P58 screenwriter,
P495 country, P364 original language (named 'wd:Qxxx'). KG edges movie -rel-> object. Non-movie entity mentions in the request
text: case-insensitive whole-word match of person (director/cast/screenwriter) labels (>=2 words) and genre labels.
Dialog per request unit: Seeker(opening text; movies = ctx items; entities = mentions) -> Recommender(movies = gold items).
Counts linking coverage and mentions. Output: crslab/data/dataset/reddit/nltk/"""
import json, re, os, glob, collections
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
O = "/path/to/base/projects/crs/baselines/crslab/data/dataset/reddit/nltk"; os.makedirs(O, exist_ok=True)
items = json.load(open(f"{P}/items.json"))["items"]
REL = {"P57": "director", "P161": "cast", "P136": "genre", "P58": "screenwriter", "P495": "country", "P364": "language"}
mv = {}; trip = set(); label = {}; kind = {}
for f in sorted(glob.glob(f"{W}/data/reddit_movie/wikidata/raw/*.json")):
    for b in json.load(open(f))["results"]["bindings"]:
        imdb = b["imdb"]["value"]; mv[imdb] = b["film"]["value"].rsplit("/", 1)[-1]
        if "p" not in b or "o" not in b: continue
        p = b["p"]["value"]; o = b["o"]["value"]
        if p not in REL or not o.startswith("http://www.wikidata.org/entity/Q"): continue
        q = "wd:" + o.rsplit("/", 1)[-1]; trip.add((imdb, REL[p], q))
        if "oLabel" in b: label[q] = b["oLabel"]["value"]
        kind.setdefault(q, set()).add(REL[p])
ent = ["<pad>"] + items + sorted({t for _, _, t in trip})      # id 0 reserved for pad_entity
e2i = {e: k for k, e in enumerate(ent)}; r2i = {r: k for k, r in enumerate(sorted({r for _, r, _ in trip}))}
kg = collections.defaultdict(list)
for h, r, t in trip:
    if h in e2i: kg[str(e2i[h])].append([r2i[r], e2i[t]])
json.dump(e2i, open(f"{O}/entity2id.json", "w")); json.dump(r2i, open(f"{O}/relation2id.json", "w")); json.dump(kg, open(f"{O}/dbpedia_subkg.json", "w"))
json.dump([e2i[i] for i in items], open(f"{O}/movie_ids.json", "w"))
json.dump({q: label.get(q, q) for q in e2i if q.startswith("wd:")}, open(f"{O}/wd_labels.json", "w"))
print("items", len(items), "wikidata-linked", sum(i in mv for i in items), "| objects", len(ent) - 1 - len(items), "| triples", len(trip),
      "| per relation", dict(collections.Counter(r for _, r, _ in trip)), flush=True)
# mention matcher: persons (>=2 words) and genres
pat = {}
for q, ks in kind.items():
    l = label.get(q, "")
    if not l or l.startswith("Q"): continue
    if (ks & {"director", "cast", "screenwriter"} and len(l.split()) >= 2) or ("genre" in ks and len(l) >= 4):
        pat.setdefault(l.lower(), q)
keys = sorted(pat, key=len, reverse=True); print("mention patterns", len(keys), flush=True)
RX = [re.compile(r"\b(" + "|".join(re.escape(k) for k in keys[i:i + 2000]) + r")\b") for i in range(0, len(keys), 2000)]
def fix(s):
    if "â" not in s and "Ã" not in s: return s
    for e in ("cp1252", "latin-1"):
        try: return s.encode(e).decode("utf-8")
        except Exception: pass
    return s
tokz = lambda s: re.findall(r"[a-z0-9']+|[^\sa-z0-9']", s.lower())
vocab = collections.Counter(); C = collections.Counter()
for sp in ("train", "valid", "test"):
    out = []
    for l in open(f"{P}/requests_{sp}.jsonl"):
        x = json.loads(l); raw = fix(x["raw"]); low = raw.lower()
        ments = sorted({pat[m.group(1)] for rx in RX for m in rx.finditer(low)}); C[f"{sp}_units"] += 1; C[f"{sp}_with_mention"] += bool(ments)
        toks = tokz(raw[:3000]); vocab.update(toks) if sp == "train" else None
        out.append({"conv_id": x["req"], "user_id": x["user"], "dialog": [
            {"utt_id": 0, "role": "Seeker", "text": toks, "movies": [i for i in x["ctx_items"] if i in e2i], "entity": ments, "word": []},
            {"utt_id": 1, "role": "Recommender", "text": ["here", "are", "some", "recommendations"], "movies": [g for g in x["gold"] if g in e2i], "entity": [], "word": []}]})
    json.dump(out, open(f"{O}/{sp}_data.json", "w"))
tok2id = {"__pad__": 0, "__start__": 1, "__end__": 2, "__unk__": 3}
for w, _ in vocab.most_common(30000): tok2id.setdefault(w, len(tok2id))
json.dump(tok2id, open(f"{O}/token2id.json", "w"))
open(f"{O}/.built", "w").write("built by build_reddit_crslab.py\n0.31\n")
print(json.dumps(dict(C)), "| vocab", len(tok2id))
