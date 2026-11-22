# WWW27 hook: per-sample gold rank + repeat flag dump (offline best-valid selection), optional seeding
import os, json, random
import numpy as np, torch
STATE = {"epoch": -1}
META = None
_f = None; _idx = None; _pos = None; _ft = None
def seed():
    s = os.environ.get("WWW27_SEED")
    if s is None: return
    s = int(s); random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def dump(mode, scores, labels, ctx, item_ids):
    global _f, _idx, _pos
    p = os.environ.get("WWW27_DUMP")
    if not p: return
    if _f is None:
        _f = open(p, "a"); _idx = torch.tensor(item_ids, dtype=torch.long); _pos = {it: i for i, it in enumerate(item_ids)}
    S = scores.detach().float().cpu()[:, _idx]
    if isinstance(ctx, torch.Tensor): ctx = ctx.tolist()
    for i, l in enumerate(labels):
        j = _pos[l]; r = int((S[i] > S[i, j]).sum()) + 1
        cid = META[i] if META is not None and len(META) == len(labels) else None
        _f.write(json.dumps([STATE["epoch"], mode, r, int(l in set(ctx[i])), int(l), cid]) + "\n")
    _f.flush()
    if os.environ.get("WWW27_TOPK") and STATE["epoch"] == "final":   # WWW27: top-100 item ids with context items masked (for the Reddit ladder evaluator)
        global _ft
        if _ft is None: _ft = open(p + ".top", "a")
        for i, l in enumerate(labels):
            v = S[i].clone(); cs = [_pos[e] for e in set(ctx[i]) if e in _pos]
            if cs: v[cs] = -1e9
            top = torch.topk(v, 100).indices.tolist()
            cid = META[i] if META is not None and len(META) == len(labels) else None
            _ft.write(json.dumps([cid, int(l), [item_ids[t] for t in top]]) + "\n")
        _ft.flush()
