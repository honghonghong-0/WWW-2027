import os as _os; REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Near-duplicate leakage check (2026-11-17). For each TEST request: max cosine to any TRAIN request (same embedding as EP_emb).
Report the distribution and the EP_emb gain (vs A1) and vs UniCRS after removing test requests with max-sim >= .95 / .90.
usage: python dup_check.py reddit|redial"""
import os, sys, json, numpy as np, torch
DS = sys.argv[1]; W = "/path/to/base/www27"
nd = lambda r: np.where(r <= 10, 1 / np.log2(r + 1), 0.0)
if DS == "redial":
    z = np.load(f"{W}/results/emb_cache_redialorig.npz"); Qt, Qtr = z["Q_test"], z["Q_train"]
    R = np.load(f"{W}/results/fullcat_redial_ranks.npz"); a1, ep = nd(R["A1"]), nd(R["A1__EP_emb"]); uni = np.mean([nd(r) for r in R["uni"]], 0)
else:
    os.environ["SPACES"] = "emb"
    src = open(f"{REPO}/coground/fullcat_episodic.py").read(); exec(src.split('D = "cuda"')[0])
    Qt, Qtr = SPLIT["test"]["Q"], Qtrain
    R = np.load(f"{W}/results/fullcat_base_rows.npz"); a1, ep, uni = R["A1"], R["A1__EP_emb"], R["uni"]
    assert list(R["req"]) == [x["req"] for x in SPLIT["test"]["L"]]
A = torch.as_tensor(np.ascontiguousarray(Qt), device="cuda").float(); B = torch.as_tensor(np.ascontiguousarray(Qtr), device="cuda").float()
mx = torch.cat([(A[b:b + 512] @ B.T).max(1).values for b in range(0, len(A), 512)]).cpu().numpy()
ok = ~np.isnan(a1)
out = {"n_test": int(ok.sum()), "maxsim_pct": {p: round(float(np.percentile(mx[ok], p)), 4) for p in (50, 90, 99, 99.9)},
       "frac>=.95": round(float((mx[ok] >= .95).mean()), 4), "frac>=.90": round(float((mx[ok] >= .90).mean()), 4)}
for th in (1.01, 0.95, 0.90, 0.85):
    k = ok & (mx < th)
    out[f"keep<{th}"] = dict(n=int(k.sum()), A1=round(float(a1[k].mean()), 4), EP=round(float(ep[k].mean()), 4), UniCRS=round(float(uni[k].mean()), 4),
                            EP_minus_A1=round(float((ep - a1)[k].mean()), 4), EP_minus_UniCRS=round(float((ep - uni)[k].mean()), 4))
q = np.quantile(mx[ok], [0, .25, .5, .75, 1.0])
out["gain_by_maxsim_quartile"] = [dict(lo=round(float(q[i]), 3), hi=round(float(q[i + 1]), 3), EP_minus_A1=round(float((ep - a1)[ok & (mx >= q[i]) & (mx <= q[i + 1])].mean()), 4)) for i in range(4)]
print(json.dumps(out, indent=1)); json.dump(out, open(f"{W}/results/dup_check_{DS}.json", "w"), indent=1)
