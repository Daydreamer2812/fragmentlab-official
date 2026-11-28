"""Conservative assembly of independently reconstructed rigid components.

A local component has an arbitrary SE(2) gauge. Two disjoint components are
merged only when at least two different, verified seams agree on that gauge
and the combined observed masks pass the collision test. Neither canvas size
nor target image is assumed.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def merge_supported_components(shapes, groups, joins, seeds, overlap, *,
                               max_trials=256, center_tol=6., angle_tol=.12):
    """Merge separate nontrivial groups on distinct corroborating constraints.

    `seeds` contains verified (score, i, j, R, t, metrics) hypotheses. A single
    seam, two modes of the same seam, and collision are insufficient. Keep
    |groups| and all original joins unchanged if no merge is justified.
    """
    groups=[dict(g) for g in groups]
    joins=list(joins)
    centers=[s.points.mean(0) for s in shapes]
    trials=merged=0
    while trials<max_trials:
        owner={i:k for k,g in enumerate(groups) for i in g}
        links=defaultdict(list)
        for s,i,j,R,t,info in seeds:
            a,b=owner[i],owner[j]
            if a==b or min(len(groups[a]),len(groups[b]))<2:continue
            if a>b:
                a,b=b,a;i,j=j,i;R,t=R.T,-R.T@t
                info=dict(info)
                for left,right in (("arc_a","arc_b"),("arc_fraction_a","arc_fraction_b")):
                    if left in info and right in info:
                        info[left],info[right]=info[right],info[left]
            P,u=groups[a][i];Q,v=groups[b][j]
            B=P@R@Q.T;w=P@t+u-B@v
            links[a,b].append((float(s),i,j,B,w,info))
        candidates=[]
        for (a,b),edges in sorted(links.items(),key=lambda item:-max(e[0] for e in item[1])):
            if len(edges)<2:continue
            edges=sorted(edges,key=lambda e:-e[0])
            # Compare the effect of relative component transforms at the
            # incoming piece centres, not at the arbitrary local origin.
            for k,e in enumerate(edges):
                s,i,j,R,t,info=e
                for f in edges[k+1:]:
                    if trials>=max_trials:break
                    z,u,v,Q,w,_=f
                    trials+=1
                    if i==u or j==v:continue  # independent anchors on both sides
                    angle=math.acos(float(np.clip(np.trace(R.T@Q)/2,-1,1)))
                    if angle>angle_tol:continue
                    xy=[groups[b][p][0]@centers[p]+groups[b][p][1] for p in (j,v)]
                    errors=[np.linalg.norm((R-Q)@x+t-w) for x in xy]
                    if max(errors)>center_tol:continue
                    # Prefer the stronger measured transform; a subsequent
                    # robust pose-graph refinement may use both constraints.
                    primary,secondary=(e,f) if s>=z else (f,e)
                    candidates.append((min(s,z),-max(errors),a,b,primary,secondary))
                if trials>=max_trials:break
            if trials>=max_trials:break
        updated=False
        for _,_,a,b,e,f in sorted(candidates,key=lambda c:(c[0],c[1]),reverse=True):
            _,_,_,R,t,_=e
            mapped={i:(R@P,R@u+t) for i,(P,u) in groups[b].items()}
            if any(overlap(shapes[i],shapes[j],P.T@Q,P.T@(v-u))>.025
                   for i,(P,u) in groups[a].items() for j,(Q,v) in mapped.items()):
                continue
            groups[a].update(mapped);del groups[b]
            for score,i,j,_,_,info in (e,f):
                joins.append({"a":shapes[i].fragment.id,"b":shapes[j].fragment.id,
                    "score":score,**info,"component_bridge":True,"component_bridge_support":2})
            merged+=1;updated=True;break
        if not updated:break
    return groups,joins,{"component_graph_merges":merged,"component_graph_trials":trials,
                         "component_graph_trial_budget":max_trials}
