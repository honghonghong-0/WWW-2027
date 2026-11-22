"""UCCR on Reddit-Movie: copy the CRSLab Reddit (nltk) data into UCCR_infer with INTEGER ids.
conv_id -> int by request time (utc) so UCCR's history builder (orders by split, then conv_id) is chronological; user_id -> int.
Same dialogs/entities/KG/concepts as KBRD/KGSF. Writes id maps for back-conversion. Does not touch the CRSLab copy."""
import json, os, shutil
from collections import Counter
W = "/path/to/base/www27"; P = f"{W}/data/reddit_movie/prep_large"
SRC = "/path/to/base/projects/crs/baselines/crslab/data/dataset/reddit/nltk"
DST = "/path/to/base/projects/crs/external/UCCR_infer/data/dataset/reddit/nltk"
os.makedirs(DST, exist_ok=True)
utc = {}
for sp in ("train", "valid", "test"):
    for l in open(f"{P}/requests_{sp}.jsonl"):
        x = json.loads(l); utc[x["req"]] = x["utc"]
D = {sp: json.load(open(f"{SRC}/{sp}_data.json")) for sp in ("train", "valid", "test")}
allc = [c for sp in ("train", "valid", "test") for c in D[sp]]
print("convs", len(allc), "without utc", sum(c["conv_id"] not in utc for c in allc))
order = sorted(allc, key=lambda c: (utc.get(c["conv_id"], 0), c["conv_id"]))
cmap = {c["conv_id"]: k for k, c in enumerate(order)}
umap = {u: k for k, u in enumerate(sorted({c["user_id"] for c in allc}))}
for sp in D:
    for c in D[sp]:
        c["conv_id"] = cmap[c["conv_id"]]; c["user_id"] = umap[c["user_id"]]
    json.dump(D[sp], open(f"{DST}/{sp}_data.json", "w"))
for f in os.listdir(SRC):
    if f.endswith(".orig_www27") or f.endswith("_data.json"): continue
    shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
json.dump({"conv": cmap, "user": umap}, open(f"{DST}/WWW27_id_maps.json", "w"))
uc = Counter(c["user_id"] for c in allc)
print("users", len(umap), "users with >=2 convs", sum(v >= 2 for v in uc.values()), "convs of such users", sum(v for v in uc.values() if v >= 2))
open(f"{DST}/README_WWW27.txt", "w").write("Copied from CRSLab reddit/nltk by build_uccr_reddit.py; conv_id/user_id remapped to ints (conv by request utc). Maps: WWW27_id_maps.json\n")
print("DONE")
