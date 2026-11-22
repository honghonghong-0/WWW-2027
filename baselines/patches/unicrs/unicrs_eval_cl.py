"""Score a trained UniCRS (best-valid rec prompt) on the CRSLab baselines' samples.
UniCRS samples are built by its own CRSRecDataset from <gen dataset>/<split>_data_processed.jsonl (generated responses, --use_resp),
one per rec item. Each UniCRS sample is tagged (dialog index, label) by replaying the same line filter; CRSLab exported samples are
tagged the same way from the CRSLab raw split (consecutive same-role utterances merged, Recommender turns x movies).
Alignment = same dialog and label, in order of occurrence; unaligned samples are counted (rank 10**6 if a CRSLab sample has no
UniCRS counterpart). Ranking over the CRSLab item pool (movie_ids.json), so the pool matches KBRD/KGSF.
Writes dump rows [epoch, mode, rank, ctx_repeat, label(crslab id), conv_id] (valid epoch 0, test 'final').
usage (in unicrs/src): python unicrs_eval_cl.py <crslab ds: inspired|tgredial> <unicrs ds> <gen ds> <rec_dir> <tag>"""
import sys, json, collections, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
CDS, UDS, GEN, REC, TAG = sys.argv[1:6]
W = "/path/to/base/www27"; X = f"{W}/data/crslab_export/{CDS}"
C = f"/path/to/base/projects/crs/baselines/crslab/data/dataset/{CDS}/{'nltk' if CDS == 'inspired' else 'pkuseg'}"
import os
if os.path.exists(f"data/{UDS}/crslab_id.json"):
    u2c = {int(k): v for k, v in json.load(open(f"data/{UDS}/crslab_id.json")).items()}
else:   # official UniCRS data: entities are DBpedia URIs; map by name (CRSLab names are the URI suffix with spaces)
    ue = json.load(open(f"data/{GEN}/entity2id.json")); ce = json.load(open(f"{C}/entity2id.json"))
    u2c = {ue["<http://dbpedia.org/resource/" + n.replace(" ", "_") + ">"]: i for n, i in ce.items() if "<http://dbpedia.org/resource/" + n.replace(" ", "_") + ">" in ue}
c2u = {v: k for k, v in u2c.items()}
pool_c = json.load(open(f"{C}/movie_ids.json")); pool_u = [c2u[c] for c in pool_c if c in c2u]; pos = {u: k for k, u in enumerate(pool_u)}
print("CRSLab pool", len(pool_c), "mapped to UniCRS entities", len(pool_u), flush=True)
dev = "cuda"; kg = DBpedia(dataset=GEN).get_entity_kg_info()
tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small"); tok.add_special_tokens(gpt2_special_tokens_dict)
model = PromptGPT2forCRS.from_pretrained("microsoft/DialoGPT-small"); model.resize_token_embeddings(len(tok)); model.config.pad_token_id = tok.pad_token_id; model = model.to(dev).eval()
ttok = AutoTokenizer.from_pretrained("roberta-base"); ttok.add_special_tokens(prompt_special_tokens_dict)
tenc = AutoModel.from_pretrained("roberta-base"); tenc.resize_token_embeddings(len(ttok)); tenc = tenc.to(dev).eval()
pe = KGPrompt(model.config.n_embd, tenc.config.hidden_size, model.config.n_head, model.config.n_layer, 2, n_entity=kg['num_entities'],
              num_relations=kg['num_relations'], num_bases=8, edge_index=kg['edge_index'], edge_type=kg['edge_type'], n_prefix_rec=10)
pe.load(f"{REC}/best"); pe = pe.to(dev).eval()
coll = CRSRecDataCollator(tokenizer=tok, device=dev, context_max_length=200, entity_max_length=32, pad_entity_id=kg['pad_entity_id'],
                          prompt_tokenizer=ttok, prompt_max_length=200)
def unicrs_tags(split):
    tags, d = [], -1
    for line in open(f"data/{GEN}/{split}_data_processed.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if len(x["context"]) == 1 and x["context"][0] == "": d += 1
        if not x["rec"] or (len(x["context"]) == 1 and x["context"][0] == "") or all(u == "" for u in x["context"]): continue
        tags += [(d, u2c.get(i, -1)) for i in x["rec"]]
    return tags
def crslab_tags(split):
    e2i = json.load(open(f"{C}/entity2id.json")); tags = []
    for d, c in enumerate(json.load(open(f"{C}/{split}_data.json"))):
        ms = c["dialog"] if "dialog" in c else c["messages"]; merged = []
        for m in ms:
            mv = [e2i[x] for x in (m.get("movies") or m.get("movie") or []) if x in e2i]
            if merged and merged[-1][0] == m["role"]: merged[-1][1].extend(mv)
            else: merged.append([m["role"], list(mv)])
        for k, (r, mv) in enumerate(merged):
            if k > 0 and r == "Recommender": tags += [(d, x) for x in mv]
    return tags
dump = {"inspired": "kbrd_insp_s0", "tgredial": "kbrd_tg_s0"}[CDS]
rows = [json.loads(l) for l in open(f"{W}/results/dumps/{dump}.jsonl")]
out = open(f"{W}/results/dumps/unicrs_{TAG}.jsonl", "w"); Cn = collections.Counter()
for split, ep, md in (("valid", 0, ("valid", "val")), ("test", "final", ("test",))):
    ex = [json.loads(l) for l in open(f"{X}/{split}.jsonl")]; ct = crslab_tags(split)
    assert [x["item"] for x in ex] == [t[1] for t in ct], "CRSLab tag replay mismatch"
    cid = [r[5] for r in rows if r[0] == ep and r[1] in md]
    ds = CRSRecDataset(dataset=GEN, split=split, tokenizer=tok, context_max_length=200, use_resp=True, prompt_tokenizer=ttok, prompt_max_length=200, entity_max_length=32)
    ut = unicrs_tags(split); assert len(ut) == len(ds), (len(ut), len(ds))
    rk = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, collate_fn=coll):
            te = tenc(**b['prompt']).last_hidden_state
            b['context']['prompt_embeds'] = pe(entity_ids=b['entity'], token_embeds=te, output_entity=True, use_rec_prefix=True)
            b['context']['entity_embeds'] = pe.get_entity_embeds()
            lg = model(**b['context'], rec=True).rec_logits[:, pool_u].float()
            lab = b['context']['rec_labels'].tolist()
            for k, l in enumerate(lab):
                j = pos.get(l); rk.append(None if j is None else int((lg[k] > lg[k, j]).sum()) + 1)
    avail = collections.defaultdict(list)
    for t, r in zip(ut, rk): avail[t].append(r)
    for n, (x, t) in enumerate(zip(ex, ct)):
        if avail[t]: r = avail[t].pop(0); Cn[f"{split}_aligned"] += 1
        else: r = None; Cn[f"{split}_unaligned"] += 1
        if r is None: r = 10 ** 6
        out.write(json.dumps([ep, split, r, int(x["item"] in set(x["ctx_entities"])), x["item"], cid[n]]) + "\n")
    Cn[f"{split}_unicrs_samples"] = len(ut); Cn[f"{split}_crslab_samples"] = len(ex)
out.close(); print(json.dumps(dict(Cn))); print("DONE")
