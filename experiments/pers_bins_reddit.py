import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Personal-episode effect by the number of the requester's own past episodes (analysis only, no selection; fixed 2026-11-17).
TEST, def A, lam in {0,1,10,100} (same main system). Bins by #usable own episodes: 0, 1, 2-3, 4-9, 10+.
Also bins by max similarity of the current request to the requester's own past requests (quartiles among warm)."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{REPO}/common/pers_pop_reddit.py").read().split("OUT = {}")[0])
R = {lam: run("test", "A", lam)[0] for lam in (0, 1, 10, 100)}
own_n = np.array([len(o) for o in OWN[("test", "A")]])
own_mx = np.array([max(float(np.dot(SPLIT["test"]["Q"][n], e)) for e, _ in o) if o else np.nan for n, o in enumerate(OWN[("test", "A")])])
allm = np.ones(len(own_n), bool)
OUT = {"by_n_own": {}, "by_own_maxsim": {}}
for name, m in (("0", own_n == 0), ("1", own_n == 1), ("2-3", (own_n >= 2) & (own_n <= 3)), ("4-9", (own_n >= 4) & (own_n <= 9)), ("10+", own_n >= 10)):
    OUT["by_n_own"][name] = {"n": int((m & ~np.isnan(R[0])).sum()), "lam0": round(float(np.nanmean(R[0][m])), 4) if m.any() else None,
        **{f"lam{l}-lam0": boot(R[l] - R[0], m) if m.sum() > 1 else None for l in (1, 10, 100)},
        "median_own_maxsim": round(float(np.nanmedian(own_mx[m])), 3) if m.any() and own_n[m].max() > 0 else None}
w = own_n > 0; qs = np.nanquantile(own_mx[w], [0, .25, .5, .75, 1])
for i in range(4):
    m = w & (own_mx >= qs[i]) & (own_mx <= qs[i + 1])
    OUT["by_own_maxsim"][f"q{i+1} [{qs[i]:.3f},{qs[i+1]:.3f}]"] = {"n": int(m.sum()), **{f"lam{l}-lam0": boot(R[l] - R[0], m) for l in (1, 10, 100)}}
json.dump(OUT, open(f"{W}/results/pers_bins_reddit.json", "w"), indent=1); print(json.dumps(OUT, indent=1)); print("BINS_DONE")
