# TG (UCCR split): process.py writes only the 2,412 eval lines, in KBRD 'final' test order. Score dump rows directly.
import json,sys,numpy as np
D,dump,kbf=sys.argv[1:4]
cid={int(k):v for k,v in json.load(open(D+'/crslab_id.json')).items()}
rows=[json.loads(l) for l in open(dump)]; rows.sort(key=lambda r:r[0])
kl=[r[4] for r in map(json.loads,open(kbf)) if r[0]=='final' and r[1]=='test']
ul=[cid.get(r[1]) for r in rows]
rk=np.array([r[2].index(r[1])+1 if r[1] in r[2] else 10**6 for r in rows])
print(json.dumps({'n':len(rows),'n_kbrd':len(kl),'same_order':ul==kl,'pos_match':sum(a==b for a,b in zip(ul,kl)),
 **{f'HR@{k}':round(float((rk<=k).mean()),4) for k in (1,10,50)},'NDCG@10':round(float(np.where(rk<=10,1/np.log2(rk+1),0).mean()),4)}))
