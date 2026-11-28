"""Classical pictorial continuation for damaged planar fragments.

An oriented, local extrapolation of the observed pixels and line-exit voting.
Everything is deterministic, uses common working-pixel coordinates, and never
loads a reference image. Extrapolated pixels are used only for comparing poses.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from shardelix.reconstruction.irregular.scan import rotation, _overlap
from shardelix.reconstruction.irregular.eroded import _longest_run


@dataclass
class Field:
    rgb: np.ndarray
    gradient: np.ndarray
    strength: np.ndarray
    distance: np.ndarray
    pad: int
    points: np.ndarray
    colors: np.ndarray
    angles: np.ndarray
    features: np.ndarray
    # Boundary anchors associated with each oriented exit feature.
    arcs: np.ndarray


def _sample(a, xy, *, order=1, mode="nearest"):
    coords = [xy[..., 1]-.5, xy[..., 0]-.5]
    if a.ndim == 2:
        return ndi.map_coordinates(a, coords, order=order, mode=mode, prefilter=False)
    shape = a.shape[2:]
    return np.stack([ndi.map_coordinates(a.reshape(*a.shape[:2], -1)[..., k],
                                         coords, order=order, mode=mode, prefilter=False)
                     for k in range(int(np.prod(shape)))], axis=-1).reshape(*xy.shape[:-1], *shape)


def make_field(shape, max_gap):
    """Extend each boundary locally along the dominant isophote.

    A nearest-interior field is the fallback in flat or near-tangential areas.
    The direction is obtained from the multichannel 2x2 structure tensor;
    only rays landing in observed pixels can supply extrapolated color.
    """
    pad = int(math.ceil(max_gap))+16
    mask = np.pad(shape.fragment.mask, pad)
    rgb = np.pad(shape.fragment.rgba[..., :3].astype(np.float32),
                 ((pad,pad),(pad,pad),(0,0)))
    dist, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
    filled = rgb[nearest[0], nearest[1]]
    smooth = ndi.gaussian_filter(filled, (1.25,1.25,0))
    gx = ndi.gaussian_filter(filled, (1.4,1.4,0), order=(0,1,0))
    gy = ndi.gaussian_filter(filled, (1.4,1.4,0), order=(1,0,0))
    Jxx = ndi.gaussian_filter(np.mean(gx*gx, axis=2), 2.2)
    Jyy = ndi.gaussian_filter(np.mean(gy*gy, axis=2), 2.2)
    Jxy = ndi.gaussian_filter(np.mean(gx*gy, axis=2), 2.2)
    angle = .5*np.arctan2(2*Jxy,Jxx-Jyy)
    mag = np.sqrt(np.maximum(Jxx+Jyy,0.))
    coherence = np.hypot(Jxx-Jyy,2*Jxy)/(Jxx+Jyy+1.e-3)
    # Obtain tensor at least 3px into visible material to avoid boundary
    # artifacts caused by foreground/background normalization.
    core = ndi.binary_erosion(mask, iterations=3)
    if not core.any():
        core = mask
    dc, nc = ndi.distance_transform_edt(~core, return_indices=True)
    dcore = np.maximum(dc, 1.e-3)
    yy,xx = np.indices(mask.shape, dtype=np.float32)
    inx=(nc[1]-xx)/dcore; iny=(nc[0]-yy)/dcore
    th=angle[nc[0],nc[1]]
    dx=-np.sin(th);dy=np.cos(th)
    dot=dx*inx+dy*iny
    orient=np.where(dot<0,-1.,1.)
    dx*=orient;dy*=orient;dot=np.abs(dot)
    exterior=(~mask)&(dist<=max_gap+12.)
    ray_x=xx+dx*dc/np.maximum(dot,.35)
    ray_y=yy+dy*dc/np.maximum(dot,.35)
    # Use the observed interior only. For concave contours or an invalid
    # projected ray, fall back to the nearest-color prediction.
    hit=ndi.map_coordinates(mask.astype(np.float32),[ray_y,ray_x],order=1,mode="constant")>.95
    reliability=np.clip((coherence[nc[0],nc[1]]-.20)/.50,0.,1.)*np.clip(mag[nc[0],nc[1]]/4.,0.,1.)
    reliability*=exterior&hit&(dot>.35)
    pred=np.stack([ndi.map_coordinates(smooth[...,k],[ray_y,ray_x],order=1,mode="nearest") for k in range(3)],axis=-1)
    extended=smooth*(1-reliability[...,None])+pred*reliability[...,None]
    # Smoothed actual gradient is compared in the same rotated coordinate
    # frame; do not use the outside/zero alpha background as an edge.
    field_g=np.stack([ndi.gaussian_filter(extended,(1.,1.,0),order=(0,1,0)),
                      ndi.gaussian_filter(extended,(1.,1.,0),order=(1,0,0))],axis=-1)
    strength=np.sqrt(np.mean(np.sum(field_g**2,axis=-1),axis=-1))
    # Oriented line-exit descriptors. Sample actual color on both sides of
    # the local isophote in the interior. Rotating the "outward" ray of the
    # opposite fragment by pi must reverse the descriptor sample ordering.
    v=shape.points+pad
    inward=np.column_stack([-shape.tangent[:,1],shape.tangent[:,0]])
    inner=v+4*inward
    # Use the gradient tensor at inward point; theta describes the isophote.
    Jx=_sample(Jxx,inner);Jy=_sample(Jyy,inner);Jz=_sample(Jxy,inner)
    theta=.5*np.arctan2(2*Jz,Jx-Jy)
    ray=np.column_stack([-np.sin(theta),np.cos(theta)])
    ray*=np.where(np.sum(ray*inward,axis=1)>0,-1.,1.)[:,None]
    transverse=np.column_stack([-ray[:,1],ray[:,0]])
    outness=-np.sum(ray*inward,axis=1)
    coh=np.hypot(Jx-Jy,2*Jz)/(Jx+Jy+1.e-3)
    power=np.sqrt(Jx+Jy)
    offsets=np.array([-9.,-5.,-2.,0.,2.,5.,9.])
    # A cross-section in the extrapolated field is useful even if the
    # physical edge has rounded away from the pictorial line.
    coords=v[:,None,:]+offsets[None,:,None]*transverse[:,None,:]
    samples=_sample(extended,coords)
    contrast=np.linalg.norm(samples[:,0]-samples[:,-1],axis=1)/np.sqrt(3.)
    # Strong crossings only; local NMS avoids repeated votes for one edge.
    merit=power*np.sqrt(np.maximum(0.,coh*outness))*np.minimum(2.,contrast/12.)
    good=(power>2.2)&(coh>.30)&(outness>.32)&(contrast>12.)
    peaks=good&(merit>=ndi.maximum_filter1d(merit,size=5,mode="wrap")-1.e-6)
    ids=np.flatnonzero(peaks)
    ids=ids[np.argsort(-merit[ids],kind="stable")[:40]]
    p=samples[ids]/255.
    centered=p-p.mean(axis=1,keepdims=True)
    # Retain absolute appearance, contrast, and shape of the edge profile.
    feat=np.concatenate([p.reshape(len(ids),21),1.4*centered.reshape(len(ids),21)],axis=1)
    return Field(extended.astype(np.float32),field_g.astype(np.float32),strength.astype(np.float32),
                 dist.astype(np.float32),pad,shape.points[ids],samples[ids],
                 np.arctan2(ray[ids,1],ray[ids,0]),feat.astype(np.float32),ids)


def line_hypotheses(fields, max_gap, *, degree=None):
    """Vote in SE(2) from opposing pictorial line exits, not boundary chords."""
    if not any(len(f.points) for f in fields):
        return {}
    feat=np.concatenate([f.features for f in fields])
    reverse=np.concatenate([np.concatenate([f.colors[:,::-1].reshape(len(f.points),21)/255.,
                         1.4*((f.colors[:,::-1]-f.colors.mean(axis=1,keepdims=True))/255.).reshape(len(f.points),21)],axis=1)
                            for f in fields])
    owner=np.concatenate([np.full(len(f.points),i,np.int32) for i,f in enumerate(fields)])
    xy=np.concatenate([f.points for f in fields])
    theta=np.concatenate([f.angles for f in fields])
    k=min(48,len(feat))
    dist, nb=cKDTree(reverse).query(feat,k=k)
    if k==1: return {}
    u,v=np.nonzero((owner[:,None]!=owner[nb])&(dist<.85))
    vv=nb[u,v]
    best=defaultdict(list)
    for a,b,d in zip(u,vv,dist[u,v]):
        i,j=int(owner[a]),int(owner[b])
        if i>=j:continue
        # R maps b's outward direction to the negative outward ray of a.
        angle=float((theta[a]-theta[b]+2*np.pi)%(2*np.pi)-np.pi)
        best[i,j].append((int(a),int(b),angle,float(d)))
    if degree is not None:
        # Retain the most distinctive pair nominations from both endpoints;
        # dense repeated textures need not allocate a quadratic pose list.
        near=defaultdict(list)
        for (i,j),items in best.items():
            quality=min(q[3] for q in items)
            near[i].append((quality,j));near[j].append((quality,i))
        keep={(min(i,j),max(i,j)) for i,items in near.items()
              for _,j in sorted(items)[:degree]}
        best={pair:items for pair,items in best.items() if pair in keep}
    result={}
    for pair,items in best.items():
        items=sorted(items,key=lambda h:h[3])[:48]
        proposals=[]
        for a,b,angle,d in items:
            R=rotation(angle)
            ray=np.array([math.cos(theta[a]),math.sin(theta[a])])
            gaps=(1.,4.,8.,13.,20.,30.,40.) if max_gap<=40. else (1.,4.,8.,13.,20.,30.,40.,.75*max_gap,max_gap)
            for gap in gaps:
                if gap>max_gap+1:continue
                t=xy[a]-R@xy[b]+gap*ray
                proposals.append((angle,t,2,d))
        # Two nonparallel line exits fix translation in the plane, rather than
        # sampling gap length independently. Require distinct crossing points
        # and compatible rotations before solving this 2x2 system.
        for k,(a,b,angle,d) in enumerate(items):
            for c,e,other,v in items[k+1:]:
                da=math.atan2(math.sin(other-angle),math.cos(other-angle))
                if a==c or b==e or abs(da)>.21:continue
                if min(np.linalg.norm(xy[a]-xy[c]),np.linalg.norm(xy[b]-xy[e]))<8.:continue
                alpha=angle+da*.5
                R=rotation(alpha)
                rays=np.array([[math.cos(theta[a]),math.sin(theta[a])],
                               [math.cos(theta[c]),math.sin(theta[c])]])
                N=np.column_stack([-rays[:,1],rays[:,0]])
                det=float(np.linalg.det(N))
                if abs(det)<.28:continue
                base=xy[[a,c]]-xy[[b,e]]@R.T
                t=np.linalg.solve(N,np.sum(N*base,axis=1))
                gaps=np.sum((t-base)*rays,axis=1)
                if np.min(gaps)<-2.5 or np.max(gaps)>max_gap+3.:continue
                proposals.append((alpha,t,5,(d+v)*.5))
        out=[]
        for q in sorted(proposals,key=lambda h:h[3]-.10*(h[2]-2)):
            a,t,_,_=q
            if any(abs(math.atan2(math.sin(a-b),math.cos(a-b)))<.09 and np.linalg.norm(t-u)<4 for b,u,_,_ in out):
                continue
            out.append(q)
            if len(out)>=36:break
        if out:result[pair]=out
    return result


def _band_score(a,b,fa,fb, R,t, max_gap):
    """Score bidirectional observed/extrapolated bands, with loose geometry."""
    pos=b.points@R.T+t; d,idx=a.tree.query(pos)
    opp=-np.sum(a.tangent[idx]*(b.tangent@R.T),axis=1)
    inward=np.column_stack([-a.tangent[idx,1],a.tangent[idx,0]])
    normal=-np.sum(inward*(pos-a.points[idx]),axis=1)
    good=(d<=max_gap+3)&(opp>.15)&(normal>=-2.)&(normal>=.38*d-2.)
    run=_longest_run(good);run=run[good[run]]
    length=len(run)*b.step
    if length<max(12.,min(len(a.points)*a.step,len(b.points)*b.step)*.065):return None
    ia=idx[run]
    if len(np.unique(ia))*a.step<length*.40:return None
    overlap=_overlap(a,b,R,t)
    if overlap>.035:return None
    # Compare predictions into the opposite observed band at several depths;
    # evaluate both directions so a long edge cannot win against one pixel.
    def predict(src,dst,field_src,field_dst,ids,T,u):
        inward=np.column_stack([-dst.tangent[ids,1],dst.tangent[ids,0]])
        xy=dst.points[ids,None,:]+inward[:,None,:]*np.array([2.,5.,9.])[None,:,None]
        q=xy@T.T+u
        valid=_sample(dst.fragment.mask.astype(np.float32),xy)>.75
        dist=_sample(field_src.distance,q+field_src.pad)
        valid&=(dist<max_gap+10.)
        w=valid*np.exp(-dist/30.)
        if w.sum()<3:return None
        actual=_sample(field_dst.rgb,xy+field_dst.pad)
        predicted=_sample(field_src.rgb,q+field_src.pad)
        diff=np.mean(abs(actual-predicted),axis=-1)
        error=float((w*np.minimum(diff,75.)).sum()/w.sum())
        g1=_sample(field_src.gradient,q+field_src.pad)@T
        g2=_sample(field_dst.gradient,xy+field_dst.pad)
        g1=g1.reshape(*xy.shape[:2],6);g2=g2.reshape(*xy.shape[:2],6)
        m1=np.linalg.norm(g1,axis=-1);m2=np.linalg.norm(g2,axis=-1)
        gw=w*np.minimum(m1,m2)
        agree=float((gw*np.sum(g1*g2,axis=-1)/np.maximum(1.e-3,m1*m2)).sum()/max(1.e-3,gw.sum()))
        # Local variance rather than difference between average colors.
        tex=float(np.sqrt(np.sum(w[...,None]*(actual-(actual*w[...,None]).sum((0,1))/w.sum())**2)/(3*w.sum())))
        grad=float((w*m2).sum()/w.sum())
        return error,agree,tex,grad
    p=predict(a,b,fa,fb,run,R,t)
    q=predict(b,a,fb,fa,ia,R.T,-R.T@t)
    if p is None or q is None:return None
    err,agree,tex,grad=(np.array(p)+q)*.5
    median=float(np.median(d[run]));disp=float(np.median(abs(d[run]-median)))
    # Compensate for ubiquitous flat colors; still permit shape to
    # support a visually homogeneous, strongly bent complementary seam.
    arc=b.points[run];chord=arc[-1]-arc[0]
    bend=float(np.sqrt(np.mean(((arc-arc[0])@np.array([-chord[1],chord[0]])/max(np.linalg.norm(chord),1.e-6))**2)))
    if err>31. or (tex<5. and bend<3.):return None
    reliability=.16+.84*min(1.,tex/25.)
    fit=math.exp(-err/18.-median/26.-disp/12.-max(0.,.6-agree)*min(1.6,grad/5.))
    score=length*fit*reliability*(.65+.35*float(np.mean(opp[run])))
    if score<3.:return None
    return float(score),dict(contact_length=float(length),median_gap=median,gap_dispersion=disp,
          interior_overlap=float(overlap),color_difference=float(err),texture=float(tex),
          profile_residual=float(err),seam_bend_rms=bend,continuation_error=float(err),
          gradient_agreement=float(agree),gradient_strength=float(grad),
          arc_a=int(ia[len(ia)//2]),arc_b=int(run[len(run)//2]),
          arc_fraction_a=float(length/(len(a.points)*a.step)),arc_fraction_b=float(length/(len(b.points)*b.step)),
          evidence="isophote-band")


def line_support(fa,fb,R,t,max_gap):
    """Independent matches of pictorial crossings under an already known pose."""
    if not len(fa.points) or not len(fb.points):
        return 0,0.
    a=fa.points[:,None,:]; b=(fb.points@R.T+t)[None,:,:]
    delta=b-a
    ra=np.column_stack([np.cos(fa.angles),np.sin(fa.angles)])[:,None,:]
    rb=(np.column_stack([np.cos(fb.angles),np.sin(fb.angles)])@R.T)[None,:,:]
    forward=np.sum(delta*ra,axis=-1);back=-np.sum(delta*rb,axis=-1)
    cross_a=abs(delta[:,:,0]*ra[:,:,1]-delta[:,:,1]*ra[:,:,0])
    cross_b=abs(delta[:,:,0]*rb[:,:,1]-delta[:,:,1]*rb[:,:,0])
    opposed=-np.sum(ra*rb,axis=-1)
    desc=np.sqrt(np.mean((fa.colors[:,None]-fb.colors[None,:,::-1])**2,axis=(-2,-1)))
    valid=(forward>=-2.)&(back>=-2.)&(forward<=max_gap+3)&(back<=max_gap+3)&(opposed>.84)&(cross_a<4.5)&(cross_b<4.5)&(desc<36.)
    score=np.exp(-desc/24.-(cross_a+cross_b)/7.-(forward+back)/55.)*np.maximum(0.,opposed)
    score[~valid]=0.
    pairs=[]
    for q in np.argsort(score.ravel())[::-1]:
        i,j=np.unravel_index(q,score.shape)
        if score[i,j]<.13:break
        if any(i==a or j==b or np.linalg.norm(fa.points[i]-fa.points[a])<8 or
               np.linalg.norm(fb.points[j]-fb.points[b])<8 for a,b in pairs):continue
        pairs.append((i,j))
    return len(pairs),float(sum(score[i,j] for i,j in pairs))


def population_weights(shapes):
    """Soft robust outlier penalty, only for a coherent rounded population.

    In mixed/ordinary torn populations the prior is disabled. Angular shards
    remain possible even in a rounded population; shape never deletes input.
    """
    if len(shapes)<20:
        return np.ones(len(shapes)),False
    compact=[];curves=[]
    for sh in shapes:
        perimeter=len(sh.points)*sh.step
        compact.append(4*np.pi*sh.fragment.mask.sum()/max(1.,perimeter**2))
        t=sh.tangent
        # Scale the angular measurement to a common working arc distance.
        hop=max(1,round(4/sh.step))
        turn=np.arccos(np.clip(np.sum(t*np.roll(t,hop,axis=0),axis=1),-1.,1.))
        curves.append([np.percentile(turn,90),np.percentile(turn,99)])
    curves=np.asarray(curves)
    med=np.median(curves,axis=0)
    mad=np.maximum(1.4826*np.median(abs(curves-med),axis=0),[.06,.08])
    # This prior only applies when the majority is smooth AND compact.
    active=bool(np.median(compact)>.80 and med[0]<.40 and mad[0]<.13)
    if not active:return np.ones(len(shapes)),False
    z=np.maximum(0.,(curves-med)/mad-[1.5,2.])
    weight=1./(1.+(z[:,0]/1.8)**2+(z[:,1]/2.5)**2)
    return np.maximum(.10,weight),True


def pose_difference(shapes, j, R,t,Q,u):
    """Compare relative poses at the moving piece center, not crop origin."""
    center=shapes[j].points.mean(0)
    d=float(np.linalg.norm((R-Q)@center+t-u))
    angle=math.acos(float(np.clip(np.trace(R.T@Q)*.5,-1.,1.)))
    return d,angle


def score_candidates(shapes, fields, candidates, max_gap, previous=()):
    """Independent cue combination and conservative confidence calibration."""
    prior,active=population_weights(shapes)
    old=defaultdict(list)
    for s,i,j,R,t,info in previous:
        old[i,j].append((s,R,t))
    scored=[]
    for s,i,j,R,t,info in candidates:
        count,strength=line_support(fields[i],fields[j],R,t,max_gap)
        agree=False
        for _,Q,u in old[i,j]:
            d,da=pose_difference(shapes,j,R,t,Q,u)
            agree=agree or (d<8. and da<.15)
        # Multiple line exits, different appearance model and robust shape
        # population are independent cues. A single line does not fix gap.
        weight=s*(1.+.18*min(3,count)+.45*min(1.5,strength)+.15*max(0,min(3,count)-1)**2)*(1.18 if agree else 1.)
        weight*=math.sqrt(prior[i]*prior[j])
        scored.append((float(weight),i,j,R,t,{**info,"band_score":float(s),
                                               "line_count":count,"line_strength":strength,
                                               "model_agreement":bool(agree),
                                               "population_weight":float(math.sqrt(prior[i]*prior[j]))}))
    return scored,active


def bridge_evidence(a,b,fa,fb,R,t,max_gap):
    """Compare two independent predictions halfway across the missing seam.

    Correlation of *variations* along the seam removes the advantage of a
    common flat paint colour. Mid-gap comparison halves the extrapolation
    horizon relative to predicting all the way into the opposite fragment.
    Original pixels and output alpha are never changed.
    """
    pos=b.points@R.T+t;d,idx=a.tree.query(pos)
    opp=-np.sum(a.tangent[idx]*(b.tangent@R.T),axis=1)
    inward=np.column_stack([-a.tangent[idx,1],a.tangent[idx,0]])
    normal=-np.sum(inward*(pos-a.points[idx]),axis=1)
    good=(d<=max_gap+3)&(opp>.15)&(normal>=-2.)&(normal>=.38*d-2.)
    run=_longest_run(good);run=run[good[run]]
    if len(run)<6:return None
    # Compare a narrow ribbon, not a lone centreline, using two predictions
    # into the gap. No midpoint is obtained from ground truth or filenames.
    p=a.points[idx[run]];q=pos[run]
    xy=p[:,None,:]+(q-p)[:,None,:]*np.array([.25,.5,.75])[None,:,None]
    A=_sample(fa.rgb,xy+fa.pad);B=_sample(fb.rgb,(xy-t)@R+fb.pad)
    A=A.mean(axis=1);B=B.mean(axis=1)
    x=A-A.mean(axis=0);y=B-B.mean(axis=0)
    n1=float(np.sum(x*x));n2=float(np.sum(y*y))
    variance=math.sqrt(min(n1,n2)/max(1,3*len(run)))
    corr=float(np.sum(x*y)/max(1.e-3,math.sqrt(n1*n2)))
    error=float(np.mean(abs(A-B)))
    # Coherence of contrast across the gap, not just agreement of mean RGB.
    da=np.diff(A,axis=0);db=np.diff(B,axis=0)
    cross=float(np.sum(da*db)/max(1.e-3,math.sqrt(float(np.sum(da*da)*np.sum(db*db)))))
    return corr,cross,error,variance


def _closed_triangles(shapes,modes):
    """Find corroborating SE(2) cycles with distinct boundary arcs.

    Each mode maps j into i. A closed triangle must agree at the actual
    moving-piece center and use different arcs at each piece. Bound neighbor
    and pose counts so the search cannot expand to all cubic combinations.
    """
    neighbors=defaultdict(dict)
    for z,(score,i,j,R,t,info) in enumerate(modes):
        if score<5. or info.get("population_weight",1.)<.5:continue
        neighbors[i].setdefault(j,[]).append((score,z,R,t))
        neighbors[j].setdefault(i,[]).append((score,z,R.T,-R.T@t))
    for i,links in neighbors.items():
        for j,items in links.items():
            links[j]=sorted(items,key=lambda v:-v[0])[:3]
    centers=[q.points.mean(0) for q in shapes]
    def different(z,w,i):
        a,b=modes[z],modes[w];A,B=a[-1],b[-1]
        ka="a" if a[1]==i else "b";kb="a" if b[1]==i else "b"
        p,q=A["arc_"+ka],B["arc_"+kb];n=len(shapes[i].points)
        return min(abs(p-q),n-abs(p-q))>n*.045
    support=np.zeros(len(modes),np.int16)
    for z,(s,i,j,R,t,info) in enumerate(modes):
        if s<5. or j not in neighbors[i]:continue
        common=set(neighbors[i])&set(neighbors[j])
        common=sorted(common,key=lambda k:-min(neighbors[i][k][0][0],neighbors[j][k][0][0]))[:16]
        for k in common:
            found=False
            for _,a,A,u in neighbors[i][k]:
                if not different(z,a,i):continue
                for _,b,B,v in neighbors[k][j]:
                    if not different(z,b,j) or not different(a,b,k):continue
                    Q=A@B;e=A@v+u
                    if np.linalg.norm((R-Q)@centers[j]+t-e)>7.:continue
                    da=math.acos(float(np.clip(np.trace(R.T@Q)*.5,-1.,1.)))
                    if da<.13:found=True;break
                if found:break
            if found:support[z]+=1
    return support


def independent_consensus(shapes,candidates,*,sparse=False):
    """Merge SE(2) modes supported by distinct proposal families.

    Preserve local alternatives for confidence calculations. A mode must beat
    competitors using either endpoint of its matched arc before it can join.
    """
    by_pair=defaultdict(list)
    for c in sorted(candidates,key=lambda q:-q[0]):
        s,i,j,R,t,info=c
        matched=None
        for mode in by_pair[i,j]:
            a,da=pose_difference(shapes,j,R,t,mode[0][3],mode[0][4])
            if a<13. and da<.25:
                matched=mode;break
        if matched is None:
            if len(by_pair[i,j])>=6:continue
            by_pair[i,j].append([c])
        else:matched.append(c)
    modes=[];incident=defaultdict(list)
    for pair,items in by_pair.items():
        for item in items:
            s,i,j,R,t,info=item[0]
            families=sorted(set(q[-1].get("source","line-contour") for q in item))
            weight=s*(1.+.26*min(2,len(families)-1))
            # The peak is often biased by a particular grid orientation or a
            # single extrapolated texture patch. Estimate a pose at the actual
            # moving centre from all locally compatible, high-quality peaks.
            near=[q for q in item if q[0]>=.65*s]
            center=shapes[j].points.mean(0)
            if len(near)>1:
                roots=[]
                for family in families:
                    part=[q for q in near if q[-1].get("source","line-contour")==family]
                    if not part:continue
                    ww=np.array([q[0] for q in part]);ww/=ww.sum()
                    angles=[math.atan2(q[3][1,0],q[3][0,0]) for q in part]
                    roots.append((np.sum(ww*np.exp(1j*np.asarray(angles))),
                                  np.sum(ww[:,None]*np.array([q[3]@center+q[4] for q in part]),axis=0)))
                P=rotation(float(np.angle(sum(q[0] for q in roots))))
                u=np.mean([q[1] for q in roots],axis=0)-P@center
                d,da=pose_difference(shapes,j,R,t,P,u)
                if d<=5. and da<=.14 and _overlap(shapes[i],shapes[j],P,u)<=.035:
                    R,t=P,u
                    info={**info,"pose_fused":True}
            c=(weight,i,j,R,t,{**info,"base_weight":float(s),"families":families})
            z=len(modes);modes.append(c)
            incident[i].append(z);incident[j].append(z)
    loops=_closed_triangles(shapes,modes)
    accepted=[]
    for z,(s,i,j,R,t,info) in enumerate(modes):
        # All distinct poses compete, but only when they use the same
        # physical region of the boundary on one of the mating fragments.
        ratios=[]
        for u,arc,frac in ((i,info["arc_a"],info["arc_fraction_a"]),(j,info["arc_b"],info["arc_fraction_b"])):
            n=len(shapes[u].points)
            comp=0.
            for k in incident[u]:
                if k==z:continue
                ss,a,b,_,_,inf=modes[k]
                pos=inf["arc_a"] if a==u else inf["arc_b"]
                f=inf["arc_fraction_a"] if a==u else inf["arc_fraction_b"]
                delta=abs(arc-pos);delta=min(delta,n-delta)
                if delta<=n*max(.05,min(.22,(frac+f)*.32)):
                    comp=max(comp,ss)
            ratios.append(s/max(1.,comp))
        uniqueness=min(ratios)
        f=info["families"];lines=info.get("line_count",0);strength=info.get("line_strength",0.)
        multi=len(f)>=2
        legacy="legacy" in f
        # Two generators alone are not independent photometric evidence;
        # still require a relative confidence margin on BOTH fragments.
        # FFT and point retrieval use overlapping photometric evidence, so
        # two proposal families alone must not certify a weak single exit.
        # Legacy geometry, multiple exits, or a large uniqueness margin
        # provide the extra evidence to seed a component.
        independent=(legacy and multi) or (lines>=2 and strength>=.50)
        strong=(s>=7. and uniqueness>=1.08 and (independent or (multi and uniqueness>=1.35)))
        multi_line=(lines>=3 and strength>=.9 and s>=12. and uniqueness>=.98)
        clear=(s>=9.5 and uniqueness>=1.25 and info.get("texture",0)>10.)
        confirmed=(legacy and multi and s>=6. and uniqueness>=1.02)
        # A verified loop can rescue an otherwise borderline seam, but never
        # a single weak/ambiguous color match.
        loop=loops[z]>0 and s>=7. and uniqueness>=1.02 and (multi or lines>=2)
        if not (strong or clear or confirmed or multi_line or loop):continue
        # A piece geometrically atypical for a coherent population can still
        # join, but must win by a larger margin rather than receiving a free
        # pass from a single weak colour match. No fragments are removed.
        prior=info.get("population_weight",1.)
        if uniqueness<1.08/math.sqrt(max(.1,prior)) and not (multi_line or loop):continue
        # Sparse nomination has not seen many competitors. Do not interpret
        # their absence as high confidence. Require corroborated cycles,
        # a distinctive legacy shape seam, or several separate line exits.
        if sparse and not (loop or (confirmed and uniqueness>=1.25) or
                           (multi and multi_line and uniqueness>=1.20)):
            continue
        w=s*min(1.6,max(.8,uniqueness))
        accepted.append((float(w),i,j,R,t,{**info,"seam_uniqueness":float(uniqueness),
                    "loop_support":int(loops[z]),"consensus_supported":bool(multi or loop)}))
    return accepted,modes


def confidence_forest(shapes, seeds, modes, fields=None, max_gap=36.):
    """Grow conservative components, rechecking ambiguity after other pieces
    occupy part of the canvas. No single ambiguous bridge merges two clusters.
    """
    incident=defaultdict(list)
    by_pair=defaultdict(list)
    for c in modes:
        s,i,j,R,t,info=c
        incident[i].append(c);incident[j].append(c);by_pair[i,j].append(c)
    for i in incident:incident[i].sort(key=lambda q:-q[0])
    def orient(c,a):
        _,i,j,R,t,_=c
        return (j,R,t) if i==a else (i,R.T,-R.T@t)
    def arc(c,a):
        inf=c[-1]
        return (inf["arc_a"],inf["arc_fraction_a"]) if c[1]==a else (inf["arc_b"],inf["arc_fraction_b"])
    def competes(c,d,a):
        n=len(shapes[a].points);i,f=arc(c,a);j,g=arc(d,a)
        return min(abs(i-j),n-abs(i-j))<=n*max(.05,min(.22,(f+g)*.32))
    def can_place(group,a,c):
        j,R,t=orient(c,a);A,u=group[a]
        Q,v=A@R,A@t+u
        if j in group:return None
        for k,(P,w) in group.items():
            if k==a:continue
            if _overlap(shapes[k],shapes[j],P.T@Q,P.T@(v-w))>.025:return None
        return j,Q,v
    used=set();groups=[];joins=[]
    # Evaluate new implied seams only on the bounded frontier.
    direct_trials=0
    # Include confidence in the ranking of each join. Pair hypotheses near
    # the same pose have already been collapsed into modes.
    for seed in sorted(seeds,key=lambda q:-q[0]):
        s,i,j,R,t,info=seed
        if i in used or j in used:continue
        group={i:(np.eye(2),np.zeros(2)),j:(R,t)}
        used.update((i,j));local=[seed]
        # Limit expensive contextual search; components can remain separate.
        while len(group)<len(shapes):
            options=[]
            for a in list(group):
                # Most incident alternatives are weak. Use a bounded frontier.
                for c in incident[a][:36]:
                    score,x,y,_,_,inf=c
                    k=y if x==a else x
                    if k in used or score<6.:continue
                    families=inf.get("families",[])
                    if inf.get("population_weight",1.)<.50:continue
                    pose=can_place(group,a,c)
                    if pose is None:continue
                    k,Q,v=pose
                    # Competition on the incoming fragment is evaluated
                    # globally. On the occupied component, colliding partners
                    # cannot be plausible alternatives anymore.
                    other=max([d[0] for d in incident[k] if d is not c and competes(c,d,k)
                               and (d[1],d[2])!=(c[1],c[2])]+[1.])
                    incoming=score/other
                    best=1.
                    for d in incident[a]:
                        if d is c or not competes(c,d,a):continue
                        if (d[1],d[2])==(c[1],c[2]):continue
                        if d[0]<=best:continue
                        if can_place(group,a,d) is not None:best=d[0]
                    conditional=score/best
                    # Check independently retrieved relations to other
                    # members of the component at the implied pose.
                    support=0
                    direct_support=0
                    for b,(P,w) in group.items():
                        if b==a:continue
                        key=(min(b,k),max(b,k))
                        found=False
                        for d in by_pair[key]:
                            if d[0]<6. or d[-1].get("population_weight",1.)<.5:continue
                            if competes(c,d,k):continue
                            if b<k:E,e=P.T@Q,P.T@(v-w);n=k
                            else:E,e=Q.T@P,Q.T@(w-v);n=b
                            dd,da=pose_difference(shapes,n,E,e,d[3],d[4])
                            if dd<7. and da<.13:
                                support+=1;found=True;break
                        # Retrieval is lossy: a second correct seam may not
                        # have made any pair's short list. Verify the implied
                        # geometry and appearance directly against OBSERVED
                        # pixels, never against the output canvas.
                        if found or fields is None or score<6. or direct_trials>=32*len(shapes):continue
                        if incoming<.65 or conditional<.65:continue
                        E,e=P.T@Q,P.T@(v-w)
                        # Fast centre/radius rejection before full band work.
                        A,B=shapes[b],shapes[k]
                        ca,cb=A.points.mean(0),B.points.mean(0)
                        if np.linalg.norm(ca-E@cb-e)>np.linalg.norm(A.points-ca,axis=1).max()+np.linalg.norm(B.points-cb,axis=1).max()+max_gap:continue
                        direct_trials+=1
                        check=_band_score(A,B,fields[b],fields[k],E,e,max_gap)
                        if check is None:continue
                        value,metrics=check
                        if value<4. or metrics["texture"]<8.:continue
                        # Do not let two anchors certify the same incoming arc.
                        extra=(value,b,k,E,e,metrics)
                        if competes(c,extra,k):continue
                        evidence=bridge_evidence(A,B,fields[b],fields[k],E,e,max_gap)
                        if evidence is None:continue
                        corr,cross,error,variance=evidence
                        if corr<.40 or error>25. or variance<5.:continue
                        # Compare to independently retrieved alternatives
                        # on both arcs, including unrelated fragment IDs.
                        rival=max([d[0] for node in (b,k) for d in incident[node]
                                   if competes(extra,d,node) and d[1:3]!=key]+[1.])
                        lc,ls=line_support(fields[b],fields[k],E,e,max_gap)
                        strength=value*(1.+.18*min(3,lc)+.45*min(1.5,ls)+.15*max(0,min(3,lc)-1)**2)
                        if strength<.60*rival:continue
                        support+=1;direct_support+=1
                    # Two separate anchors on different incoming arcs can
                    # resolve an ambiguous single seam. Without them keep the
                    # original strict, mutually distinctive one-anchor gate.
                    multi_anchor=(support>=1 and score>=6. and incoming>=.70 and conditional>=.70)
                    single_anchor=(incoming>=1.12 and conditional>=.92 and
                                   (len(families)>=2 or inf.get("line_count",0)>=2) and
                                   (inf.get("line_count",0)>=2 or support>0))
                    if not (single_anchor or multi_anchor):continue
                    weight=score*math.sqrt(min(2.,incoming)*min(1.5,conditional))*(1+.40*min(2,support))
                    options.append((weight,a,k,Q,v,c,incoming,conditional,support,direct_support))
            if not options:break
            weight,a,k,Q,v,c,u1,u2,support,direct_support=max(options,key=lambda q:q[0])
            group[k]=(Q,v);used.add(k)
            s,x,y,E,e,inf=c
            local.append((float(weight),x,y,E,e,{**inf,"seam_uniqueness":float(min(u1,u2)),
                         "context_support":int(support),"direct_bridge_support":int(direct_support),"contextual_growth":True,"loop_support":int(support)}))
        groups.append(group)
        for s,i,j,_,_,inf in local:
            joins.append({"a":shapes[i].fragment.id,"b":shapes[j].fragment.id,"score":float(s),**inf})
    for i in range(len(shapes)):
        if i not in used:groups.append({i:(np.eye(2),np.zeros(2))})
    def quality(g):
        ids={shapes[i].fragment.id for i in g}
        e=[j["score"] for j in joins if j["a"] in ids and j["b"] in ids]
        if not e:return (0.,len(g))
        return (float(sum(e))*(min(e)/max(e))**.25,len(g))
    groups.sort(key=quality,reverse=True)
    return groups,joins


def verify_rounded_seeds(shapes, fields, seeds, max_gap):
    """Reject seam seeds lacking contrast evidence on smooth eroded fragments.

    Distinct line crossings can still establish a seam despite weak texture
    correlation. Rejected seeds stay among the competing pose modes.
    """
    accepted=[]
    for s,i,j,R,t,info in seeds:
        evidence=bridge_evidence(shapes[i],shapes[j],fields[i],fields[j],R,t,max_gap)
        if evidence is not None:
            corr,derivative,error,variance=evidence
            if info.get("line_count",0)<2 and (
                (corr<.50 and derivative<.25) or derivative<.07):
                continue
            info={**info,"seed_bridge_correlation":corr,
                  "seed_bridge_derivative_correlation":derivative}
        accepted.append((s,i,j,R,t,info))
    return accepted


def recover(shapes,previous,update=lambda _:None,*,memory_budget=None,profile="balanced",offset_search=False):
    """Bounded high-quality classical search; useful when contact is sparse."""
    from shardelix.reconstruction.irregular.eroded import _hypotheses,_profiles
    from shardelix.reconstruction.irregular.spectral_gap import fft_proposals,spectral_workspace
    n=len(shapes)
    if n<2:return None,[],{"attempted":False,"reason":"insufficient fragments"}
    from shardelix.reconstruction.resource_budget import resource_budget
    budget=memory_budget or resource_budget()
    median=float(np.median([min(s.fragment.mask.shape) for s in shapes]))
    # Scale the gap with the measured masks (larger scan fragments can have
    # physically wider abrasion). Reserve headroom for descriptors and FFTs.
    max_gap=min(max(36.,median*.35),max(10.,median*.70))
    field_limit=budget.memory_bytes//96
    requested_gap=max_gap
    def field_cost(gap):
        pad=math.ceil(gap)+16
        areas=[(s.fragment.mask.shape[0]+2*pad)*(s.fragment.mask.shape[1]+2*pad) for s in shapes]
        pixels=sum(areas)
        return pixels,44*pixels+160*max(areas)+192_000*n
    pixels,workspace=field_cost(max_gap)
    # If padding, rather than the observed fragments, is the bottleneck,
    # use the widest smaller gap that fits instead of skipping continuation.
    if pixels>field_limit or workspace>budget.memory_bytes:
        lo,hi=10.,max_gap
        for _ in range(12):
            mid=(lo+hi)/2
            p,w=field_cost(mid)
            if p<=field_limit and w<=budget.memory_bytes:lo=mid
            else:hi=mid
        max_gap=lo
        pixels,workspace=field_cost(max_gap)
    if pixels>field_limit or workspace>budget.memory_bytes:
        return None,[],{"attempted":False,"reason":"estimated continuation fields exceed memory budget",
                        "estimated_field_pixels":pixels,"field_pixel_budget":field_limit,
                        "estimated_workspace_bytes":workspace}
    # FFT spectra can require close to 1 MiB per piece. Sparse nomination
    # bounds work on larger sets; do not impose a fixed piece-count ceiling.
    sparse=n>128
    degree=min(8 if profile=="accurate" else 6,budget.gap_pair_budget//max(1,n))
    if sparse and degree<2:
        return None,[],{"attempted":False,"reason":"estimated continuation pair budget"}
    profiles=_profiles(shapes)
    if sum(float(np.percentile(p.texture,80))>=4. for p in profiles)<2:
        return None,[],{"attempted":False,"reason":"insufficient pictorial texture"}
    update("Classical deep recovery: extrapolating observed pictorial structure")
    fields=[make_field(s,max_gap) for s in shapes]
    hyps=line_hypotheses(fields,max_gap,degree=degree if sparse else None)
    line_count=sum(map(len,hyps.values()))
    for i,j,a,t,v,d in _hypotheses(profiles,min(max(30.,median*.30),max_gap),update,
                                   sparse_degree=degree if sparse else None):
        hyps.setdefault((i,j),[]).append((a,t,v,d))
    # Bound search on large sets, preserving the strongest distinct proposals
    # per pair. The old and new methods can disagree; retain alternatives.
    def verify(hypotheses,source,limit):
        out=[]
        for q,((i,j),poses) in enumerate(hypotheses.items()):
            local=[]
            for angle,t,votes,dist in poses:
                R=rotation(angle)
                z=_band_score(shapes[i],shapes[j],fields[i],fields[j],R,t,max_gap)
                if z is not None:
                    s,info=z
                    local.append((s,i,j,R,t,{**info,"votes":votes,"descriptor_distance":dist,"source":source}))
            out.extend(sorted(local,key=lambda q:-q[0])[:limit])
            if q%max(1,len(hypotheses)//12)==0:
                update(f"Classical {source} verification {q+1}/{len(hypotheses)}")
        return out
    n_hyp=sum(map(len,hyps.values()))
    raw=verify(hyps,"line-contour",5)
    ranked,prior_on=score_candidates(shapes,fields,raw,max_gap,previous)
    best={}
    for s,i,j,*_ in ranked:best[i,j]=max(s,best.get((i,j),0.))
    # Pair nomination from two directions; old reliable matches also enter.
    near=defaultdict(list)
    for (i,j),s in best.items():
        near[i].append((s,j));near[j].append((s,i))
    spectral_degree=degree if sparse else 16
    pairs={(min(i,j),max(i,j)) for i,cs in near.items()for _,j in sorted(cs,reverse=True)[:spectral_degree]}
    pairs.update((i,j) for _,i,j,*_ in previous)
    # For small sets, search all pairs, including ones missed by point retrieval.
    if n<=24:pairs={(i,j) for i in range(n) for j in range(i+1,n)}
    spectral_bytes=spectral_workspace(shapes,pairs,max_gap)+pixels*44+192_000*n
    spectral_skipped=spectral_bytes>budget.memory_bytes
    if spectral_skipped:pairs=set()
    spectral=fft_proposals(shapes,fields,sorted(pairs),max_gap,update)
    sr=verify(spectral,"spectral",4)
    sf,_=score_candidates(shapes,fields,sr,max_gap,previous)
    weights,_=population_weights(shapes)
    legacy=[(float(s*math.sqrt(weights[i]*weights[j])),i,j,R,t,{**info,"source":"legacy",
             "population_weight":float(math.sqrt(weights[i]*weights[j]))})for s,i,j,R,t,info in previous]
    seeds,modes=independent_consensus(shapes,ranked+sf+legacy,sparse=sparse)
    seed_count=len(seeds)
    if prior_on:
        seeds=verify_rounded_seeds(shapes,fields,seeds,max_gap)
    groups,joins=confidence_forest(shapes,seeds,modes,fields,max_gap)
    from shardelix.reconstruction.irregular.component_graph import merge_supported_components
    groups,joins,graph_stats=merge_supported_components(shapes,groups,joins,seeds,_overlap)
    if graph_stats["component_graph_merges"]:
        from shardelix.reconstruction.irregular.eroded import rank_components
        # Preserve the internal gauges of already reconstructed components.
        groups=rank_components(shapes,groups,joins)
    offset_stats={"attempted":False}
    # An eroded but still curved cut can preserve an offset of its original
    # contour. Compact rounded fragments lack enough such shape evidence.
    if offset_search and not prior_on and profile!="fast":
        from shardelix.reconstruction.irregular.offset_contours import offset_recovery
        from shardelix.reconstruction.irregular.scan import _merge
        offset_seeds,offset_stats=offset_recovery(shapes,fields,max_gap,budget,update)
        if offset_seeds:
            # The conservative offset seeds have a separate physical
            # contour constraint. Merge them into a single graph, retaining
            # only collision-free placements. Do not invent global locations.
            og,oj=_merge(shapes,offset_seeds)
            if oj:
                groups,joins=og,oj
                offset_stats["selected"]=True
    stats={"attempted":True,"gap_limit_px":max_gap,"requested_gap_limit_px":requested_gap,"field_pixels":int(pixels),"field_pixel_budget":int(field_limit),"estimated_workspace_bytes":int(workspace),
           "spectral_estimated_workspace_bytes":int(spectral_bytes),
           "sparse_degree":degree if sparse else None,"spectral_skipped_for_memory":bool(spectral_skipped),"line_tokens":sum(len(f.points)for f in fields),
           "line_proposals":line_count,"point_proposals":n_hyp,"point_verified":len(raw),
           "spectral_pairs":len(pairs),"spectral_proposals":sum(map(len,spectral.values())),
           "spectral_verified":len(sr),"population_prior":prior_on,
           "population_downweighted":int(np.count_nonzero(weights<.5)),
           "pose_modes":len(modes),"seed_hypotheses":len(seeds),"seed_bridge_rejections":seed_count-len(seeds),
           "offset_recovery":offset_stats,"component_graph":graph_stats,"accepted_joins":len(joins),
           "contextual_growth_joins":sum(bool(j.get("contextual_growth"))for j in joins),
           "closed_loop_joins":sum(j.get("loop_support",0)>0 for j in joins),
           "direct_bridge_joins":sum(j.get("direct_bridge_support",0)>0 for j in joins),
           "component_sizes":[len(g)for g in groups]}
    return groups,joins,stats
