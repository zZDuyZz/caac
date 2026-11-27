"""Usage: python bl_diag_legs.py results/bl_w2.jsonl.gz MAP_REL_JSONL POOLS_DIR
POST-HOC diagnostic (not in the design): when CLT+Bonferroni says GO and CAAC does not (W2, n=500),
which CAAC GO leg fails for the policy CLT+Bonferroni names."""
import json, numpy as np, sys
from collections import Counter
from multiprocessing import Pool
import bl_w2 as W, bl_methods as M
from bl_methods import R
def run(x):
    D,d,C,lo,hi,names,cfg=W.extract(x); P,N=D.shape; Df=D.astype(float); cid=W.M4.config_id(x); n=500
    idx=np.stack([np.random.default_rng(W.seed_of("BL",cid,n,r)).integers(0,N,n) for r in range(W.REPS)])
    Ds=np.transpose(Df[:,idx],(1,0,2)); ds=np.transpose(d[:,idx],(1,0,2)); mc=np.transpose(C[:,idx],(1,0,2)).mean(-1)
    _,chc=M.clt_bonf(Ds,ds,mc,lo,hi); _,cha,_=M.caac(Ds,ds,mc,lo,hi)
    c=Counter()
    for r in np.where((chc>=0)&(cha<0))[0]:
        k=chc[r]; pr=R.betting_p_rows((M.EPS-Ds[r,k])[None],M.EPS-1,M.EPS+1)[0]; ps=R.betting_p_rows(ds[r,k][None],lo,hi)[0]
        L=M.DELTA/P; c[("risk" if pr>L else "")+("+" if pr>L and ps>L else "")+("saving" if ps>L else "")]+=1
    return dict(c), (hi-lo)/max(1e-9,abs(d.mean()))
if __name__=="__main__":
    cf={json.loads(l)["config"]:json.loads(l) for l in __import__("gzip").open(sys.argv[1], "rt")}
    xs=[x for x in W.M4.select(sys.argv[2]) if cf[W.M4.config_id(x)]["any_feasible"]]
    tot=Counter()
    with Pool(2,initializer=W._init,initargs=(sys.argv[3],M.REPO+"/protocols/audit_lock_v3.json")) as p:
        for c,_ in p.imap_unordered(run,xs,chunksize=4): tot.update(c)
    print(len(xs), dict(tot))
