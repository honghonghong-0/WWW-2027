"""Convert a CRSLab dataset (Inspired nltk / TG-ReDial pkuseg) to UniCRS input format, so UniCRS trains on the SAME split and the
same KG as KBRD/KGSF. Entities are re-indexed 0..N-1 (UniCRS embedding index); map back via crslab_id.json.
Output dir: baselines/unicrs/src/data/<name>/{train,valid,test}_data_dbpedia.jsonl, entity2id.json, relation2id.json,
dbpedia_subkg.json, item_ids.json (= CRSLab movie_ids pool), crslab_id.json; plus process.py/process_mask.py copied from UniCRS inspired.
usage: crslab2unicrs.py <inspired|tgredial> <name>"""
import json, sys, os, re, shutil, collections
DS, NAME = sys.argv[1], sys.argv[2]
C = os.environ.get("CRSLAB_DIR") or f"/path/to/base/projects/crs/baselines/crslab/data/dataset/{DS}/{'pkuseg' if DS == 'tgredial' else 'nltk'}"
U = "/path/to/base/projects/crs/baselines/unicrs/src/data"; O = f"{U}/{NAME}"; os.makedirs(O, exist_ok=True)
ce2i = json.load(open(f"{C}/entity2id.json")); kgf = f"{C}/" + {'inspired': 'dbpedia_subkg.txt', 'tgredial': 'cn-dbpedia.txt', 'reddit': 'dbpedia_subkg.json', 'redial': 'dbpedia_subkg.json'}[DS]
names = sorted(ce2i, key=ce2i.get); e2i = {n: k for k, n in enumerate(names)}
json.dump(e2i, open(f"{O}/entity2id.json", "w"), ensure_ascii=False)
json.dump({str(k): ce2i[n] for n, k in e2i.items()}, open(f"{O}/crslab_id.json", "w"))
rel = {}; kg = collections.defaultdict(list); kept = drop = 0
if DS in ("reddit", "redial"):   # CRSLab json KG {head_id: [[rel_id, tail_id], ...]} (Wikidata; same graph as KBRD/KGSF)
    ci2n = {v: k for k, v in ce2i.items()}; r2n = {v: k for k, v in json.load(open(f"{C}/relation2id.json")).items()}
    _lines = [f"{ci2n[int(h)]}\t{r2n[r]}\t{ci2n[tl]}" for h, es in json.load(open(kgf)).items() for r, tl in es]
else: _lines = open(kgf, encoding="utf-8")
for l in _lines:
    p = l.rstrip("\n").split("\t")
    if len(p) != 3: drop += 1; continue
    h, r, t = p
    if h in e2i and t in e2i:
        kg[str(e2i[h])].append([rel.setdefault(r, len(rel)), e2i[t]]); kept += 1
    else: drop += 1
json.dump(rel, open(f"{O}/relation2id.json", "w"), ensure_ascii=False); json.dump(kg, open(f"{O}/dbpedia_subkg.json", "w"))
c2u = {ce2i[n]: e2i[n] for n in names}
items = [c2u[i] for i in json.load(open(f"{C}/movie_ids.json")) if i in c2u]
json.dump(items, open(f"{O}/item_ids.json", "w"))
sep = "" if DS == "tgredial" else " "
for sp in ("train", "valid", "test"):
    with open(f"{O}/{sp}_data_dbpedia.jsonl", "w", encoding="utf-8") as f:
        for c in json.load(open(f"{C}/{sp}_data.json")):
            ms = c["dialog"] if "dialog" in c else c["messages"]
            def mname(n): n = re.sub(r"^<http://dbpedia.org/resource/|>$", "", n).replace("_", " "); return re.sub(r"[（(].*?[)）]", "", n).strip()   # surface title for <movie> masking
            f.write(json.dumps([{"role": m["role"].upper(), "text": sep.join(m["text"]), "entity_link": list(m.get("entity", [])),
                                 "movie_link": list(m.get("movies") or m.get("movie") or []),
                                 "movie_name": [mname(n) for n in (m.get("movies") or m.get("movie") or [])]} for m in ms], ensure_ascii=False) + "\n")
for fn in ("process.py", "process_mask.py"): shutil.copy(f"{U}/inspired/{fn}", f"{O}/{fn}")
print(DS, "entities", len(e2i), "relations", len(rel), "kg triples kept", kept, "dropped", drop, "items", len(items))
