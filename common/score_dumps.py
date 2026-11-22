"""Score per-sample rank dumps (TG-ReDial, UCCR last-two-sessions split).
Epoch is selected on VALID only (hit@1+hit@50, the CRSLab KGSF/UCCR early-stop criterion); test at that epoch is reported.
Views: all | ctx-norepeat (gold not among current-context entities; He et al. CIKM'23 shortcut, EyeCRS 4.1.4)
       | hist-norepeat (gold not mentioned in any of the user's earlier train-split sessions; our probe definition)
dump row: [epoch, mode, rank, ctx_repeat, label_entity_id, conv_id]
usage: score_dumps.py dump1.jsonl [...]   -> results/scores/<dump>.json (one file per dump)"""
import json, sys, collections, numpy as np
D = "/path/to/base/projects/crs/external/UCCR_infer/data/dataset/tgredial/pkuseg"
id2ent = {v: k for k, v in json.load(open(f"{D}/entity2id.json")).items()}
conv = {}
for sp in ["train", "valid", "test"]:
    for c in json.load(open(f"{D}/choiced_last_two_{sp}_data.json")):
        conv[c["conv_id"]] = (c["user_id"], sp, {m for u in c["messages"] for m in u["movie"]})
byuser = collections.defaultdict(list)
for cid, (u, sp, mv) in conv.items(): byuser[u].append((cid, sp, mv))
_h = {}
def hist_movies(cid):
    if cid not in _h:
        u = conv[cid][0]; _h[cid] = set().union(*[mv for c, sp, mv in byuser[u] if sp == "train" and c < cid] or [set()])
    return _h[cid]
def met(r):
    r = np.array(r)
    if len(r) == 0: return {"n": 0}
    return {f"HR@{k}": round(float((r <= k).mean()), 4) for k in (1, 5, 10, 20, 50)} | {
        "MRR@10": round(float(np.where(r <= 10, 1 / r, 0).mean()), 4), "NDCG@10": round(float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()), 4), "n": len(r)}
# UCCR's dataloader dropped conv_id (null). Eval sample order is deterministic across models (verified: identical
# label sequences in KBRD/KGSF/UCCR dumps), so null conv_ids are filled by position from a fully mapped reference
# dump, but only when the label at that position matches; otherwise the row stays unmapped and is counted.
REF = {}
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    if rows and all(x[5] is not None for x in rows):
        e0 = rows[0][0]
        for x in rows:
            if x[0] == e0: REF.setdefault(("valid" if x[1] in ("val", "valid") else x[1], f), []).append((x[4], x[5]))
REFM = {}
for (m, f), v in REF.items(): REFM.setdefault(m, v)
out = {}
for f in sys.argv[1:]:
    P = collections.defaultdict(list); miss = 0; pos = collections.Counter(); filled = 0
    for l in open(f):
        e, m, r, crep, lab, cid = json.loads(l); m = "valid" if m in ("val", "valid") else m
        k = pos[(e, m)]; pos[(e, m)] += 1
        if cid is None and m in REFM and k < len(REFM[m]) and REFM[m][k][0] == lab: cid = REFM[m][k][1]; filled += 1
        if cid is None or cid not in conv: miss += 1; hrep = None
        else: hrep = int(id2ent.get(lab) in hist_movies(cid))
        P[(e, m)].append((r, crep, hrep))
    eps = sorted({e for e, m in P if m == "valid" and e != "final"})
    crit = {e: (lambda r: (r <= 1).mean() + (r <= 50).mean())(np.array([x[0] for x in P[(e, "valid")]])) for e in eps}
    best = max(eps, key=lambda e: crit[e])
    # per-epoch test (offline selection) if present; otherwise the run restored the best-valid checkpoint and tested once ("final")
    if P.get((best, "test")): T = P[(best, "test")]; src = "test_at_best_valid_epoch"
    else: T = P[("final", "test")]; src = "final_test_of_restored_best_checkpoint"
    res = dict(test_source=src, best_epoch=best, n_epochs=len(eps), unmapped_rows=miss, filled_by_position=filled, valid=met([x[0] for x in P[(best, "valid")]]),
               test_all=met([x[0] for x in T]), test_ctx_norepeat=met([x[0] for x in T if not x[1]]),
               test_hist_norepeat=met([x[0] for x in T if x[2] == 0]), test_hist_repeat=met([x[0] for x in T if x[2] == 1]),
               ctx_repeat_frac=round(float(np.mean([x[1] for x in T])), 4),
               hist_repeat_frac=round(float(np.mean([x[2] for x in T if x[2] is not None])), 4),
               test_final_epoch=met([x[0] for x in P[("final", "test")]]))
    out[f.split("/")[-1]] = res
    print(f.split("/")[-1], "best_ep", best, "| test HR@10 all %.4f ctx-norep %.4f hist-norep %.4f hist-rep %s | hist_rep_frac %.3f unmapped %d filled %d" % (
        res["test_all"]["HR@10"], res["test_ctx_norepeat"]["HR@10"], res["test_hist_norepeat"].get("HR@10", float("nan")),
        res["test_hist_repeat"].get("HR@10"), res["hist_repeat_frac"], miss, filled))
# one summary file per dump (no cross-experiment overwrite): results/scores/<dump name>.json
import os
SD = "/path/to/base/www27/results/scores"; os.makedirs(SD, exist_ok=True)
for k, v in out.items(): json.dump(v, open(f"{SD}/{k.replace('.jsonl', '')}.json", "w"), indent=1, ensure_ascii=False)
print("wrote", len(out), "files to", SD)
