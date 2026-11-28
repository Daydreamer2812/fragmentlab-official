"""Pixel-only continuation hypotheses for physical document seams.

All exemplar patches come from the submitted strips. No glyph decoding,
reference page, external training corpus, or synthetic pixel export.
"""
from __future__ import annotations

import math
import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from shardelix.reconstruction.strips.ordering import chain


def _unit_cost(c):
    """Scale positive score matrices without creating negative edge costs."""
    v=c[~np.eye(len(c),dtype=bool)]
    return c/max(1.e-6,float(np.median(v)))


def _patch_recurrence(images,half,gap,*,inset=None):
    """Classical nonparametric matching of half patches from observed pixels.

    Use a bounded index of interior patches of the same input page. For every
    seam-side patch, find geometrically similar half patches in intact local
    neighborhoods and compare their observed continuations with the opposing
    side. No glyph labels, OCR, training, language model, or generated pixels.
    """
    n=len(images);h=len(images[0]);step=2
    # Historical coarse geometry is retained as a complementary score.
    # Calibrated geometry measures center-to-center distances explicitly:
    # pixel center = inset + 0.5, with half the missing cut width on each side.
    if inset is None:
        yoffset=np.arange(-6,7,3)
        offset=np.arange(half-step//2,0,-step)-.5+gap
        xoffset=np.r_[-offset,offset[::-1]]
        interior=half+gap+3
        edge_inset=2
    else:
        yoffset=np.arange(-6,7,2)
        offset=np.arange(inset+.5,inset+half,step)+gap/2
        xoffset=np.r_[-offset[::-1],offset]
        interior=math.ceil(offset[-1])+1
        edge_inset=inset
    # Bound both query count and corpus size; same coordinates for every strip.
    stride=max(3,math.ceil(n*h/35_000))
    ycenter=np.arange(6,h-6,stride)
    min_width=max(half+3,edge_inset+step*(len(offset)-1)+1)
    if not len(ycenter) or any(im.shape[1]<min_width for im in images):return None
    corpus=[];edges=[]
    stride_x=max(3,math.ceil(len(ycenter)*sum(max(0,im.shape[1]-2*interior) for im in images)/300_000))
    for im in images:
        sm=gaussian_filter(im,.6)
        xs=np.arange(interior,im.shape[1]-interior,stride_x)
        if len(xs):
            Y,X=np.meshgrid(ycenter,xs,indexing="ij")
            v=map_coordinates(sm,[Y[...,None,None]+yoffset[:,None]+np.zeros((1,len(xoffset))),
                                  X[...,None,None]+xoffset[None,:]+np.zeros((len(yoffset),1))],
                              order=1,mode="nearest").reshape(-1,len(yoffset),len(xoffset))
            corpus.append(v[(v>.18).sum((1,2))>=2])
        xL=im.shape[1]-1-edge_inset-np.arange(len(offset))[::-1]*step
        xR=edge_inset+np.arange(len(offset))*step
        Y=ycenter[:,None,None]+yoffset[None,:,None]+np.zeros((1,1,len(offset)))
        edges.append([map_coordinates(sm,[Y,x[None,None,:]+np.zeros_like(Y)],order=1,mode="nearest")
                      .reshape(len(ycenter),-1) for x in (xL,xR)])
    if not corpus:return None
    c=np.concatenate(corpus)
    if len(c)<32:return None
    if len(c)>50_000:c=c[np.linspace(0,len(c)-1,50_000,dtype=int)]
    a,b=np.split(c,2,axis=2);a=a.reshape(len(c),-1);b=b.reshape(len(c),-1)
    e=np.asarray(edges);L,R=e[:,1],e[:,0]
    def predict(source,target,query):
        tree=cKDTree(source)
        # Independent KD-tree queries release the GIL. Use a bounded worker
        # pool on native CPUs, never threads in the browser/Wasm runtime.
        # Scheduling and chunk size do not alter search distances or eps.
        try:cores=len(os.sched_getaffinity(0))
        except (AttributeError,OSError):cores=os.cpu_count() or 1
        workers=1 if sys.platform=="emscripten" else min(4,cores)
        def search(q,eps):
            return tree.query(q,k=8,eps=eps,workers=workers if len(q)>=2048 else 1)
        q=query.reshape(-1,query.shape[-1])
        if inset is not None:
            # Deduplicate identical observed half patches, especially empty
            # paper. Chunk queries to bound neighbor memory on long pages.
            unique, inverse=np.unique(q,axis=0,return_inverse=True)
            result=np.empty_like(unique)
            for start in range(0,len(unique),8192):
                u=unique[start:start+8192]
                d,nn=search(u,.12)
                weight=np.exp(-d*d/(.06+d[:,0:1]**2))
                result[start:start+len(u)]=(weight[...,None]*target[nn]).sum(1)/weight.sum(1)[:,None]
            return result[inverse].reshape(query.shape)
        result=np.empty_like(q)
        # Full blank patches are identical queries. Evaluate once instead of
        # repeatedly traversing the high-dimensional tree over blank paper.
        nonzero=np.any(q!=0,axis=1);indices=np.flatnonzero(nonzero)
        if not nonzero.all():
            d,nn=search(np.zeros((1,q.shape[1]),dtype=q.dtype),.08)
            weight=np.exp(-d*d/(.06+d[:,0:1]**2))
            result[~nonzero]=(weight[...,None]*target[nn]).sum(1)/weight.sum(1)[:,None]
        # Bound working memory, including the neighbor tensor.
        for start in range(0,len(indices),8192):
            ii=indices[start:start+8192];d,nn=search(q[ii],.08)
            weight=np.exp(-d*d/(.06+d[:,0:1]**2))
            result[ii]=(weight[...,None]*target[nn]).sum(1)/weight.sum(1)[:,None]
        return result.reshape(query.shape)
    PL,PR=predict(a,b,R),predict(b,a,L)
    def pair(a,b):
        a=a.reshape(n,-1);b=b.reshape(n,-1)
        return cdist(a,b,"sqeuclidean")/(np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None,:]+1)
    return _unit_cost(.5*(pair(PL,L)+pair(R,PR)))


def _chain_distinctiveness(cost, order):
    """Lower is better: reciprocal rank and separation from rival seams.

    Different gap hypotheses have different absolute error scales. Compare
    ranks and dimensionless alternative margins instead of raw chain sums.
    This is a selection heuristic, not a calibrated accuracy probability.
    """
    a,b=order[:-1],order[1:]
    selected=cost[a,b]
    rowrank=np.sum(cost[a]<selected[:,None],axis=1)
    colrank=np.sum(cost[:,b]<selected[None,:],axis=0)
    alternatives=cost.copy()
    alternatives[a,b]=np.inf
    margin=np.minimum(alternatives[a].min(1)-selected,
                      alternatives[:,b].min(0)-selected)
    scale=max(1e-6,float(np.std(cost[np.isfinite(cost)])))
    return float(.5*np.mean(rowrank+colrank)-.5*np.mean(margin)/scale)


def select_cut_hypothesis(images, baseline, borders, update):
    """Evaluate a small bank of physical gap hypotheses, select without labels.

    An observed seam may lose several pixels of ink. Narrow, densely sampled
    patches complement the older wider patch evidence. The mixture bank also
    keeps the historical solution available if it is more distinctive.
    """
    def candidate(matrix, name):
        c=matrix.copy()
        np.fill_diagonal(c,np.inf)
        order=chain(c,.5,borders=borders)
        return _chain_distinctiveness(c,order),name,c,order

    base=baseline.copy()
    np.fill_diagonal(base,0)
    best=candidate(base,"coarse")
    diagnostics=[{"hypothesis":best[1],"distinctiveness_cost":best[0]}]
    # Two small missing-cut widths at the matching scale. A bounded bank,
    # evaluated once; only the chosen path needs the expensive global solver.
    for gap in (2,3):
        update(f"Checking {gap}-pixel cut-loss continuation")
        local=_patch_recurrence(images,6,gap,inset=1)
        if local is None:continue
        for weight in (.3,.6,1.):
            trial=candidate((1-weight)*base+weight*local,f"gap-{gap}-weight-{weight:g}")
            diagnostics.append({"hypothesis":trial[1],"distinctiveness_cost":trial[0]})
            if trial[0]<best[0]-1e-9:best=trial
    return best[2],best[3],{"selected":best[1],"candidates":diagnostics}
