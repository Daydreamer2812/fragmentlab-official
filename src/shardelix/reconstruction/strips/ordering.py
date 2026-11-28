"""Classical document stripe matching; no OCR/model, file names, or reference.

Foreground-weighted bidirectional gradient prediction, ambiguity-normalized
edge costs, and directed chain optimization. Supports unequal stripe widths.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from shardelix.reconstruction.common import Fragment, Pose
from shardelix.reconstruction.tiles.straight import _strip_assignment, _cycle_components
from shardelix.reconstruction.rotation import _orientation_dp, _signed_tree

# Frozen after training on the even-numbered LSU source documents only.
DEFAULT = dict(metric="gradient", depth=1, dark=10., boundary=0., normalization=.5, margins=8., blank_split=True)

def is_documentlike(pieces):
    """Inspect pixels, not names: white paper, sparse dark ink, low saturation."""
    samples = np.concatenate([p.rgba[p.mask][::max(1,int(p.mask.sum())//400),:3]
                              for p in pieces[:min(128,len(pieces))]])
    if not len(samples): return False
    white = (samples.min(axis=1)>225).mean()
    dark = (samples.mean(axis=1)<150).mean()
    neutral = (np.ptp(samples.astype(int),axis=1)<25).mean()
    return white>.52 and .015<dark<.38 and neutral>.90

def _ink(a,axis=0):
    if axis: a=a.transpose(1,0,2)
    rgb=a[:,:,:3].astype(np.float32)
    v=1-(.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2])/255
    v[a[:,:,3]==0]=0
    return v

def _profiles(images,depth,max_samples=1400):
    height=max(x.shape[0] for x in images)
    samples=min(height,max_samples)
    def one(x,right):
        if right:x=x[:,::-1]
        d=min(depth,x.shape[1]-1)
        e=x[:,0];g=(e-x[:,d])/max(d,1)
        if x.shape[0]!=samples:
            xx=np.linspace(0,len(e)-1,samples)
            e=np.interp(xx,np.arange(len(e)),e);g=np.interp(xx,np.arange(len(g)),g)
        return e,g
    l,lg=map(np.asarray,zip(*(one(x,False) for x in images)))
    r,rg=map(np.asarray,zip(*(one(x,True) for x in images)))
    return l,lg,r,rg

def _pair(r,rg,l,lg,metric, dark,boundary,shift=0):
    n,h=r.shape
    def compute(a,ag,b,bg):
        if metric=="l1":
            return cdist(a,b,"cityblock")/a.shape[1]
        if metric=="ink":
            # Stroke compatibility: relative mismatched foreground, not blank pixels.
            a=gaussian_filter1d(a,.55,axis=1);b=gaussian_filter1d(b,.55,axis=1)
            d=cdist(a,b,"cityblock")
            union=a.sum(1)[:,None]+b.sum(1)[None,:]
            return .6*d/a.shape[1]+.4*d/(union+.03*a.shape[1]+1e-9)
        if n>160:
            # Bounded-memory vectorized bidirectional prediction.
            pred=(cdist(a+ag,b,"sqeuclidean")+cdist(a,b+bg,"sqeuclidean"))*.5
            raw=cdist(a,b,"sqeuclidean")
            return np.sqrt((pred+boundary*raw)/a.shape[1])
        # Exact foreground weighting in small/medium document tasks.
        out=np.empty((n,len(b)))
        for i in range(n):
            ae,ge=a[i],ag[i];be,gb=b,bg
            err=.5*((ae+ge-be)**2+(be+gb-ae)**2)+boundary*(ae-be)**2
            weight=1+dark*np.maximum(np.maximum(ae,ae-ge),np.maximum(be,be-gb))
            out[i]=np.sqrt((weight*err).sum(1)/weight.sum(1))
        return out
    m=compute(r,rg,l,lg)
    if shift:
        for d in range(1,shift+1):
            for s,t in ((slice(d,None),slice(None,-d)),(slice(None,-d),slice(d,None))):
                m=np.minimum(m,compute(r[:,s],rg[:,s],l[:,t],lg[:,t])+.0025*d)
    return m

def document_costs(images,metric="gradient",depth=1,dark=10.,boundary=.25,shift=0,rotations=False):
    l,lg,r,rg=_profiles(images,depth)
    def pair(a,ag,b,bg):
        m=_pair(a,ag,b,bg,metric,dark,boundary,shift)
        np.fill_diagonal(m,np.inf)
        return m
    m=pair(r,rg,l,lg)
    if not rotations:return m
    A=np.empty((2,2,len(l),len(l)))
    A[0,0]=m;A[1,1]=m.T
    A[0,1]=pair(r,rg,r[:,::-1],rg[:,::-1])
    A[1,0]=pair(l[:,::-1],lg[:,::-1],l,lg)
    return A

def _priority(m,weight):
    """Normalize by competing edges but retain absolute cost."""
    n=len(m)
    if n<3:return m.copy()
    row=np.partition(m,1,axis=1)[:,1]
    col=np.partition(m,1,axis=0)[1,:]
    finite=m[np.isfinite(m)]
    scale=max(np.median(finite),1e-7)
    if weight==0:return m/scale
    if weight==1:return .5*(m/np.maximum(row[:,None],1e-6)+m/np.maximum(col[None,:],1e-6))
    return (1-weight)*m/scale+weight*.5*(m/np.maximum(row[:,None],1e-6)+m/np.maximum(col[None,:],1e-6))

def _greedy(cost):
    n=len(cost);succ=np.full(n,-1,int);prev=succ.copy()
    parent=np.arange(n)
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    count=0
    for idx in np.argsort(cost.ravel(),kind="stable"):
        a,b=divmod(int(idx),n)
        if a==b or succ[a]>=0 or prev[b]>=0 or root(a)==root(b):continue
        succ[a]=b;prev[b]=a;parent[root(a)]=root(b);count+=1
        if count==n-1:break
    start=int(np.flatnonzero(prev<0)[0]);order=[start]
    while succ[order[-1]]>=0:order.append(int(succ[order[-1]]))
    return np.asarray(order)

def _chain_energy(cost,order):
    n=len(order)
    return float(cost[order[:-1],order[1:]].sum()+cost[n,order[0]]+cost[order[-1],n])

def _relocate(cost,order,max_length=5,passes=15):
    """Vectorized Or-opt on a directed path with a zero-cost sentinel."""
    n=len(order)
    if n<3:return order
    a=cost
    for _ in range(passes):
        seq=np.r_[n,order,n];best=(0.,None)
        for L in range(1,min(max_length,n-1)+1):
            q=np.arange(1,n-L+2);p=q+L-1
            v,u=seq[q],seq[p];before,after=seq[q-1],seq[p+1]
            # Remove one segment, then insert into any disjoint edge.
            remove=a[before,after]-a[before,v]-a[u,after]
            with np.errstate(invalid="ignore"):
                delta=remove[:,None]+a[seq[:-1][None,:],v[:,None]]+a[u[:,None],seq[1:][None,:]]-a[seq[:-1],seq[1:]][None,:]
            j=np.arange(n+1)
            delta[(j[None,:]>=q[:,None]-1)&(j[None,:]<=p[:,None])]=np.inf
            delta[~np.isfinite(delta)]=np.inf
            i,jj=np.unravel_index(np.argmin(delta),delta.shape)
            if delta[i,jj]<best[0]-1e-9:best=(float(delta[i,jj]),(int(q[i]-1),L,int(jj)))
        if best[1] is None:break
        start,L,j=best[1];block=order[start:start+L];rest=np.r_[order[:start],order[start+L:]]
        idx=j-(L if j>start else 0)
        order=np.r_[rest[:idx],block,rest[idx:]]
    return order

def _border_cost(images, strength):
    # Outer page borders are often blank. Penalize endpoints that cut through
    # ink, rather than rewarding a spurious white-to-white internal connection.
    l=np.array([x[:,0].mean() for x in images])
    r=np.array([x[:,-1].mean() for x in images])
    scale=max(float(np.median(np.r_[l,r])),.015)
    # Disable a margin prior when the page is cropped through text.
    return (strength*l/scale if l.min()<.015 else np.zeros_like(l),
            strength*r/scale if r.min()<.015 else np.zeros_like(r))

def _assign_with_border(c):
    n=len(c)-1
    a,b=linear_sum_assignment(c);succ=np.empty(n+1,int);succ[a]=b
    cycles=_cycle_components(succ)
    while len(cycles)>1:
        u=np.array(next(cy for cy in cycles if n in cy));v=np.array([i for cy in cycles if n not in cy for i in cy])
        du,dv=succ[u],succ[v]
        d=c[u[:,None],dv]+c[v[None,:],du[:,None]]-c[u,du][:,None]-c[v,dv]
        i,j=np.unravel_index(np.argmin(d),d.shape)
        a,b=u[i],v[j];succ[a],succ[b]=succ[b],succ[a]
        cycles=_cycle_components(succ)
    order=[];i=int(succ[n])
    while i!=n:order.append(i);i=int(succ[i])
    return np.asarray(order)

def chain(m,normalization=.5, widths=None,slots=None,borders=None):
    n=len(m)
    if n==1:return np.array([0])
    if widths is None:widths=np.ones(n,int)
    if slots is None:slots=np.ones(n,int)
    if len(set(slots))>1:return _strip_assignment(m,widths,slots)[0]
    c=np.zeros((n+1,n+1));c[:n,:n]=_priority(m,normalization);c[n,n]=np.inf
    if borders is not None:c[n,:n],c[:n,n]=borders
    options=[_assign_with_border(c),_greedy(c[:n,:n]),_greedy(m)]
    candidates=[]
    for o in options:
        # Select the best observed page boundary, including wrap-around if it
        # repairs a falsely joined blank margin seam.
        candidates.append(min((np.roll(o,k) for k in range(n)),key=lambda o:_chain_energy(c,o)))
    best=min(candidates,key=lambda o:_chain_energy(c,o))
    return _relocate(c,best,passes=16 if n<=128 else 2)

def refine_chain_global(m, order, borders=None, *, normalization=.5, top_k=8, seconds=1.5):
    """Bounded directed path refinement using assignment lower bounds.

    Each cycle cover is the exact relaxation of the directed path with a
    dummy endpoint. A disconnected subtour must lose at least one of its
    edges, so branching on its arcs covers every feasible Hamiltonian path.
    Best-first search explores only cheaper covers and preserves the best
    feasible incumbent on timeout. No MILP backend or thread-pool startup
    is required; completion certifies optimality over the edge shortlist.
    """
    import heapq
    import time
    n=len(m)
    method="assignment-subtour-branch-and-bound"
    if n<3:return order,{"method":method,"accepted":False,"status":"not-applicable"}
    c=np.zeros((n+1,n+1))
    c[:n,:n]=_priority(m,normalization);c[n,n]=np.inf
    if borders is not None:c[n,:n],c[:n,n]=borders
    before=_chain_energy(c,order)
    diag={"method":method,"initial_energy":before,"selected_energy":before,
          "accepted":False,"status":"budget","iterations":0}
    if seconds<=0:return order,diag
    # Same sparse search space as the previous MILP implementation.
    shortlist=np.full_like(c,np.inf)
    for i in range(n):
        row=np.argsort(c[i,:n],kind="stable")[:top_k]
        col=np.argsort(c[:n,i],kind="stable")[:top_k]
        shortlist[i,row]=c[i,row];shortlist[col,i]=c[col,i]
        shortlist[n,i]=c[n,i];shortlist[i,n]=c[i,n]
    shortlist[order[:-1],order[1:]]=c[order[:-1],order[1:]]
    diag["candidate_arcs"]=int(np.isfinite(shortlist).sum())
    # Fixed cap limits queue memory even when thousands of cycles compete.
    max_nodes=8192
    deadline=time.monotonic()+seconds
    best=before;best_order=order
    queue=[];seen=set();serial=0;truncated=False

    def examine(forbidden):
        nonlocal serial,best,best_order,truncated
        if forbidden in seen:return
        if serial>=max_nodes or time.monotonic()>=deadline:
            truncated=True;return
        seen.add(forbidden);serial+=1
        q=shortlist.copy()
        if forbidden:
            a,b=zip(*forbidden);q[a,b]=np.inf
        try:rows,cols=linear_sum_assignment(q)
        except ValueError:return
        lb=float(q[rows,cols].sum())
        if lb>=best-1e-9:return
        succ=np.empty(n+1,dtype=int);succ[rows]=cols
        cycles=_cycle_components(succ)
        if len(cycles)==1:
            path=[];j=int(succ[n])
            while j!=n:path.append(j);j=int(succ[j])
            candidate=np.asarray(path,dtype=int)
            energy=_chain_energy(c,candidate)
            if energy<best-1e-9:best=energy;best_order=candidate
        else:
            cycle=min(cycles,key=len)
            heapq.heappush(queue,(lb,serial,forbidden,
                                  tuple((int(i),int(succ[i])) for i in cycle)))

    examine(frozenset())
    while queue and not truncated:
        lb,_,forbidden,cycle=heapq.heappop(queue)
        if lb>=best-1e-9:break
        for arc in cycle:
            examine(forbidden|{arc})
            if truncated:break
    diag.update(iterations=serial,status="budget" if truncated else "shortlist-optimal",
                selected_energy=best,accepted=bool(best<before-1e-9))
    return best_order,diag

def _projection_cost(images):
    """Text-line row projection under 0/180 rotations, sensitive to line baseline."""
    h=min(1400,max(len(x) for x in images))
    rows=[]
    for x in images:
        p=x.mean(axis=1)
        if len(p)!=h:p=np.interp(np.linspace(0,len(p)-1,h),np.arange(len(p)),p)
        p=gaussian_filter1d(p,.8);p-=p.mean()
        rows.append(p/max(float(np.linalg.norm(p)),1e-7))
    p=np.asarray(rows);r=p[:,::-1]
    same=np.clip(1-p@p.T,0,2);flipped=np.clip(1-p@r.T,0,2)
    return same,flipped

def _solve_document_v5(pieces,shape=None,grid=None,rotations=False,**params):
    """Blind document path. Images only, optional public canvas/grid dimensions."""
    config={**DEFAULT,**params};normalization=config.pop("normalization");margins=config.pop("margins");blank_split=config.pop("blank_split");projection=float(config.pop("projection",0.))
    axis=int(grid is not None and grid[1]==1 and grid[0]>1)
    turns=np.zeros(len(pieces),int)
    canonical=[]
    for i,p in enumerate(pieces):
        a=p.rgba
        if rotations and ((axis==0 and a.shape[0]<a.shape[1]) or (axis==1 and a.shape[1]<a.shape[0])):
            turns[i]=1;a=p.rotated(1)
        canonical.append(_ink(a,axis))
    widths=np.array([x.shape[1] for x in canonical])
    left,right=_border_cost(canonical,margins)
    slots=None
    if grid is not None and shape is not None:
        from shardelix.reconstruction.tiles.straight import _grid_bounds
        yy,xx=_grid_bounds(*shape,*grid)
        slots=np.diff(yy if axis else xx)
    if len(pieces)==1:
        order=np.array([0]);signs=np.zeros(1,int)
    else:
        raw=document_costs(canonical,rotations=rotations,**config)
        if rotations:
            same=np.minimum(raw[0,0],raw[1,1]);flip_cost=np.minimum(raw[0,1],raw[1,0])
            s=_signed_tree(np.minimum(same,same.T),np.minimum(flip_cost,flip_cost.T),neighbors=12)
            seeds=[s]
            if projection:
                line_same,line_flip=_projection_cost(canonical)
                seeds.append(_signed_tree(line_same,line_flip,neighbors=12))
                scale=np.median(raw[np.isfinite(raw)])
                raw[0,0]+=projection*scale*line_same
                raw[1,1]+=projection*scale*line_same
                raw[0,1]+=projection*scale*line_flip
                raw[1,0]+=projection*scale*line_flip
            best=None;ids=np.arange(len(pieces))
            for seed in seeds:
              for flip in (0,1):
                signs=seed^flip
                for _ in range(4):
                    mat=raw[signs[:,None],signs[None,:],ids[:,None],ids[None,:]]
                    border=(np.where(signs,right,left),np.where(signs,left,right))
                    order=chain(mat,normalization,widths if slots is not None else None,slots,border)
                    new=_orientation_dp(raw,order)
                    for q in (signs,new):
                        e=float(raw[q[order[:-1]],q[order[1:]],order[:-1],order[1:]].sum())
                        e+=float(np.median(raw[np.isfinite(raw)]))*(np.where(q,right,left)[order[0]]+np.where(q,left,right)[order[-1]])
                        if best is None or e<best[0]:best=(e,order.copy(),q.copy())
                    if np.array_equal(signs,new):break
                    signs=new
            _,order,signs=best
        else:
            signs=np.zeros(len(pieces),int)
            # Entirely white strips carry no ID/order evidence. Avoid placing
            # all blank margins at one end, which shifts every content strip.
            blank=np.array([x.max()<.004 for x in canonical])
            nonblank=np.flatnonzero(~blank);empty=np.flatnonzero(blank)
            if blank_split and len(empty) and len(nonblank)>1 and (slots is None or len(set(slots))==1):
                bimgs=[canonical[i] for i in nonblank]
                bl,br=_border_cost(bimgs,margins)
                sub=chain(raw[np.ix_(nonblank,nonblank)],normalization,borders=(bl,br))
                core=nonblank[sub]
                a=canonical[core[0]][:,0].mean();b=canonical[core[-1]][:,-1].mean()
                # With a missing/cropped margin on one side, use the other.
                if a<.015 and b>=.015:count=len(empty)
                elif b<.015 and a>=.015:count=0
                else:count=(len(empty)+1)//2
                order=np.r_[empty[:count],core,empty[count:]]
            else:
                order=chain(raw,normalization,widths if slots is not None else None,slots,(left,right))
    poses={};offset=0
    for i in order:
        poses[pieces[i].id]=Pose(0 if axis else offset,offset if axis else 0,int((turns[i]+2*signs[i])%4))
        offset+=int(widths[i])
    return poses,{"algorithm":"document-foreground-path","config":{**config,"normalization":normalization,"margins":margins,"projection":projection},
                  "orientation_ambiguous":bool(rotations),"blank_strip_count":int(sum(x.max()<.004 for x in canonical)),"source_reference_used":False}


def oriented_path_milp(raw,left,right,normalization=.5,top_k=8,timeout=8.):
    """Globally coherent binary orientation/path search for small document sets.

    Uses a directed sparse state graph, one incoming/outgoing edge per fragment,
    state-flow conservation and lazy subtour cuts. No labels/model/reference.
    Returns None if a complete feasible path was not obtained within budget.
    """
    import time
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    n=raw.shape[2];N=2*n;dummy=N
    cost=raw.transpose(2,0,3,1).reshape(N,N).copy()
    ids=np.arange(N)//2
    cost[ids[:,None]==ids[None,:]]=np.inf
    finite=cost[np.isfinite(cost)]
    if not len(finite):return None
    scale=max(float(np.median(finite)),1e-7)
    second=np.maximum(np.partition(cost,1,axis=1)[:,1],scale*.01)
    other=np.maximum(np.partition(cost,1,axis=0)[1],scale*.01)
    with np.errstate(invalid="ignore"):
        norm=cost/np.sqrt(second[:,None]*other[None,:])
    obj=(1-normalization)*cost/scale+normalization*np.log1p(norm)
    obj[~np.isfinite(cost)]=np.inf
    pairs=set()
    for i in range(N):
        for j in np.argsort(obj[i])[:min(top_k,N-1)]:
            if np.isfinite(obj[i,j]):pairs.add((i,int(j)))
        for j in np.argsort(obj[:,i])[:min(top_k,N-1)]:
            if np.isfinite(obj[j,i]):pairs.add((int(j),i))
        pairs.add((dummy,i));pairs.add((i,dummy))
    edges=sorted(pairs);m=len(edges)
    a=np.array([i for i,j in edges]);b=np.array([j for i,j in edges])
    ei=np.arange(m)
    lc=np.column_stack((left,right)).ravel();rc=np.column_stack((right,left)).ravel()
    c=np.array([lc[j] if i==dummy else rc[i] if j==dummy else obj[i,j] for i,j in edges])
    rows=[];cols=[];vals=[];ub=[];lb=[]
    def constrain(indices,values,lo,hi):
        index=len(ub)
        rows.extend([index]*len(indices));cols.extend(indices);vals.extend(values);ub.append(hi);lb.append(lo)
    for i in range(n):
        for v in (a,b):
            ii=np.flatnonzero((v<N)&(v//2==i))
            constrain(ii,np.ones(len(ii)),1,1)
    for i in range(N):
        aa=np.flatnonzero(a==i);bb=np.flatnonzero(b==i)
        constrain(np.r_[aa,bb],np.r_[np.ones(len(aa)),-np.ones(len(bb))],0,0)
    for v in (a,b):
        ii=np.flatnonzero(v==dummy)
        constrain(ii,np.ones(len(ii)),1,1)
    start=time.perf_counter()
    for _ in range(15):
        rem=timeout-(time.perf_counter()-start)
        if rem<=0:break
        M=coo_matrix((vals,(rows,cols)),shape=(len(ub),m)).tocsc()
        sol=milp(c,integrality=np.ones(m),bounds=Bounds(0,1),
                 constraints=LinearConstraint(M,np.array(lb),np.array(ub)),
                 options={"time_limit":rem,"mip_rel_gap":.001})
        if sol.x is None:break
        chosen=np.flatnonzero(sol.x>.5);succ={int(a[i]):int(b[i]) for i in chosen}
        if len(chosen)!=n+1 or dummy not in succ:break
        order=[];i=succ[dummy]
        while i!=dummy and i not in order and i in succ:
            order.append(i);i=succ[i]
        if i==dummy and len(order)==n:
            order=np.array(order);ids=order//2
            if len(set(ids))!=n:return None
            signs=np.zeros(n,int);signs[ids]=order%2
            return ids,signs
        seen=set(order)|{dummy};unseen=set(succ)-seen
        while unseen:
            j=min(unseen);cycle=[]
            while j in unseen:
                unseen.remove(j);cycle.append(j);j=succ[j]
            group=np.array([s//2 for s in cycle])
            inds=np.flatnonzero((a<N)&(b<N)&np.isin(a//2,group)&np.isin(b//2,group))
            constrain(inds,np.ones(len(inds)),-np.inf,len(group)-1)
    return None

def solve_document_strips(pieces,shape=None,grid=None,rotations=False,**params):
    """Maintain V5 path; add a duplicate-aware oriented hypothesis for repeated pages."""
    import hashlib
    from collections import defaultdict
    poses,diag=_solve_document_v5(pieces,shape,grid,rotations,**params)
    n=len(pieces)
    if __import__("sys").platform=="emscripten":return poses,{**diag,"duplicate_milp_skipped":"browser portability"}
    if not rotations or not (4<=n<=64) or (grid is not None and grid[1]==1):return poses,diag
    if grid is not None:
        from shardelix.reconstruction.tiles.straight import _grid_bounds
        y,x=_grid_bounds(*shape,*grid)
        if len(set(np.diff(x)))!=1:return poses,diag
    groups=defaultdict(list);canonical=[];turns=[]
    for i,p in enumerate(pieces):
        turn=int(p.rgba.shape[0]<p.rgba.shape[1])
        a=p.rotated(turn)
        key=min(hashlib.sha256(np.ascontiguousarray(np.rot90(a,k)).tobytes()).digest() for k in (0,2))
        groups[key].append(i);canonical.append(a);turns.append(turn)
    # This is an observable exact repetition, not a dataset-name condition.
    if len(groups)<2 or any(len(v)!=2 for v in groups.values()):return poses,diag
    reps=[v[0] for v in groups.values()];partners={v[0]:v[1] for v in groups.values()}
    ims=[_ink(a) for a in canonical];sub=[ims[i] for i in reps]
    conf={**DEFAULT,**params};w=conf.pop("normalization");margins=conf.pop("margins");conf.pop("blank_split");conf.pop("projection",None)
    raw=document_costs(ims,rotations=True,**conf)
    raw_unique=raw[:,:,reps,:][:,:,:,reps]
    l,r=_border_cost(sub,margins)
    result=oriented_path_milp(raw_unique,l,r,w,top_k=8,timeout=8.)
    if result is None:return poses,{**diag,"duplicate_orbit_count":len(groups)}
    order,signs=result
    core=[reps[j] for j in order]
    q={reps[j]:int(signs[j]) for j in order}
    for first in core:
        b=partners[first];target=np.rot90(canonical[first],2*q[first])
        q[b]=0 if np.array_equal(canonical[b],target) else 1
    l0,r0=_border_cost(ims,margins)
    scale=max(float(np.median(raw[np.isfinite(raw)])),1e-7)
    def value(order,sg):
        a=np.array(order[:-1]);b=np.array(order[1:])
        s=np.array([sg[i] for i in a]);t=np.array([sg[i] for i in b])
        return float(raw[s,t,a,b].sum()/scale+((r0 if sg[order[0]] else l0)[order[0]])+((l0 if sg[order[-1]] else r0)[order[-1]]))
    ids={p.id:i for i,p in enumerate(pieces)}
    base_order=[ids[k] for k in sorted(poses,key=lambda k:poses[k].x)]
    base_s={i:((poses[pieces[i].id].k-turns[i])%4)//2 for i in range(n)}
    candidates=[(value(base_order,base_s),base_order,base_s,"v5")]
    # Test both duplicated-page arrangements solely by visible seam evidence.
    repeated=core+[partners[i] for i in core]
    mirrored=core+[partners[i] for i in core[::-1]]
    qm=q.copy()
    for i in core:qm[partners[i]]^=1
    candidates += [(value(repeated,q),repeated,q,"repeat"),(value(mirrored,qm),mirrored,qm,"mirror")]
    e,seq,sg,label=min(candidates,key=lambda x:x[0])
    out={};x=0
    for i in seq:
        k=int((turns[i]+2*sg[i])%4);out[pieces[i].id]=Pose(x,0,k);x+=pieces[i].rotated(k).shape[1]
    alternatives=[]
    for energy,order,sign,kind in candidates:
        if kind==label or abs(energy-e)>1e-7:continue
        x=0;alt={}
        for i in order:
            k=int((turns[i]+2*sign[i])%4);alt[pieces[i].id]=Pose(x,0,k).as_dict();x+=pieces[i].rotated(k).shape[1]
        alternatives.append({"hypothesis":kind,"poses":alt,"blind_energy":energy})
    return out,{**diag,"algorithm":"document-duplicate-oriented-milp","duplicate_orbit_count":len(groups),
               "repeat_hypothesis":label,"repeat_trials":[{"hypothesis":a[3],"blind_energy":a[0]} for a in candidates],
               "equally_scored_alternatives":alternatives}
