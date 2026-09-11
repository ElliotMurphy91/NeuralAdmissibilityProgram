"""Injectivity certificate: enumerate all commutative non-associative trees on n labelled leaves,
evaluate each law at generic (random) leaf functions, report the minimum pairwise separation.
Distinct real-analytic tree functions differ on a full-measure set, so a nonzero minimum over
random inputs certifies injectivity on all trees with <= n leaves (up to measure zero)."""
import numpy as np, itertools, sys
from functools import lru_cache
from gate_core import gate_norm, make_lexicon, compose_tree, NPT

def trees(leaves):
    """all unordered binary trees over a frozenset of leaves ((2n-3)!! of them)"""
    leaves=tuple(sorted(leaves))
    if len(leaves)==1: return [leaves[0]]
    out=[]; first=leaves[0]; rest=leaves[1:]
    # split: first goes left; choose the rest of the left set
    for k in range(0,len(rest)):
        for comb in itertools.combinations(rest,k):
            L=(first,)+comb; R=tuple(x for x in rest if x not in comb)
            if len(R)==0: continue
            for tl in trees(L):
                for tr in trees(R): out.append((tl,tr))
    return out

rng=np.random.default_rng(11)
W=np.linalg.qr(rng.normal(size=(NPT,NPT)))[0]
laws={"gate_norm b=1":lambda u,v:gate_norm(u,v,1.0),"mean":lambda u,v:0.5*(u+v),
      "tanh_mean g=2":lambda u,v:np.tanh(u+v)/np.tanh(2),"tied tanh":lambda u,v:np.tanh(0.75*(u+v)@W.T)}
for n in (4,5,6,7):
    T=trees(range(n)); print(f"n={n}: {len(T)} trees")
    for name,g in laws.items():
        mind=[]; ncoll=[]
        for rep in range(3):
            L=make_lexicon(n,np.random.default_rng(1000*n+rep))
            X=np.stack([compose_tree(t,L,g) for t in T])
            scale=np.linalg.norm(X,axis=1).mean()
            # nearest-neighbour distance via sorting on random projections + exact check of candidates
            from scipy.spatial import cKDTree
            d,_=cKDTree(X).query(X,k=2); nn=d[:,1]/scale
            mind.append(nn.min()); ncoll.append(int((nn<1e-9).sum()))
        print(f"   {name:16s} min rel. NN separation = {min(mind):.2e}   trees with a collision (<1e-9): {max(ncoll)}")
