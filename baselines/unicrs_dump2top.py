"""UniCRS WWW27 dump [processed_line, label, top100 unicrs entity ids] -> reddit_top_eval '.top' rows [conv_id(req), label, top100 imdb ids]."""
import json, sys
D = "/path/to/base/projects/crs/baselines/unicrs/src/data/reddit_wd"
C = "/path/to/base/projects/crs/baselines/crslab/data/dataset/reddit/nltk/test_data.json"
convs = [c["conv_id"] for c in json.load(open(C))]; line2conv = []
for k, l in enumerate(open(f"{D}/test_data_dbpedia.jsonl")): line2conv += [convs[k]] * len(json.loads(l))   # process.py: one processed line per turn
i2n = {v: k for k, v in json.load(open(f"{D}/entity2id.json")).items()}
n = 0
with open(sys.argv[2], "w") as f:
    for l in open(sys.argv[1]):
        ln, lab, top = json.loads(l); f.write(json.dumps([line2conv[ln], i2n[lab], [i2n[i] for i in top]]) + "\n"); n += 1
print("rows", n, "processed lines", len(line2conv), "convs", len(convs))
