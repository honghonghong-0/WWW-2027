"""Paired bootstrap of test-sample metrics between two dumps (same samples, same order). Resampling unit = dialog
(TG: conv_id from the dump; Inspired: dialog key = first two context utterances of the exported sample). 5,000 resamples.
usage: boot_compare.py <inspired|tgredial> <dumpA> <dumpB> [view=all|norep]   (reports B - A)"""
import json, sys, numpy as np
W = "/path/to/base/www27"
DS, A, B = sys.argv[1:4]; VIEW = sys.argv[4] if len(sys.argv) > 4 else "all"
ex = [json.loads(l) for l in open(f"{W}/data/crslab_export/{DS}/test.jsonl")]
def load(f):
    R = [json.loads(l) for l in open(f)]
    fin = [r for r in R if r[0] == "final" and r[1] == "test"]
    if not fin:  # per-epoch dumps: use best-valid epoch (HR@1+HR@50)
        eps = sorted({r[0] for r in R if r[1] in ("valid", "val")})
        crit = {e: np.mean([r[2] <= 1 for r in R if r[0] == e and r[1] in ("valid", "val")]) + np.mean([r[2] <= 50 for r in R if r[0] == e and r[1] in ("valid", "val")]) for e in eps}
        b = max(eps, key=crit.get); fin = [r for r in R if r[0] == b and r[1] == "test"]
    return fin
a, b = load(A), load(B); assert [r[4] for r in a] == [r[4] for r in b] == [x["item"] for x in ex][:len(a)], "order mismatch"
keep = [k for k in range(len(a)) if VIEW == "all" or not a[k][3]]
key = [str(a[k][5]) if a[k][5] is not None else "|".join(ex[k]["ctx"][:2]) for k in keep]
def m(rs): r = np.array(rs, float); return np.stack([(r <= 1), (r <= 10), (r <= 50), np.where(r <= 10, 1 / np.log2(r + 1), 0)], 1)
d = m([b[k][2] for k in keep]) - m([a[k][2] for k in keep])
u, inv = np.unique(key, return_inverse=True); S = np.zeros((len(u), 4)); np.add.at(S, inv, d); C = np.bincount(inv)
rng = np.random.default_rng(0); est = []
for _ in range(5000):
    i = rng.integers(0, len(u), len(u)); est.append(S[i].sum(0) / C[i].sum())
est = np.array(est); lo, hi = np.percentile(est, [2.5, 97.5], 0)
for j, nm in enumerate(["HR@1", "HR@10", "HR@50", "NDCG@10"]):
    print(f"{nm}: delta {d[:, j].mean():+.4f} [{lo[j]:+.4f}, {hi[j]:+.4f}]  p~{min(1.0, 2 * min((est[:, j] < 0).mean() + 0.5 * (est[:, j] == 0).mean(), (est[:, j] > 0).mean() + 0.5 * (est[:, j] == 0).mean())):.4f}")
print("n samples", len(keep), "dialogs", len(u), "| B - A,", B.split('/')[-1], "-", A.split('/')[-1])
