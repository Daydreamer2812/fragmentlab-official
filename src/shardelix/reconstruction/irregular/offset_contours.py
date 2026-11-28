"""Hypothesis-only morphological offset matching for abraded curved seams.

Expanding a mask estimates a possible lost boundary; expanded pixels never
become fragments or enter rendering. Independent observed-mask and pictorial
bridge checks are required before a pose can seed the component graph.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
from scipy import ndimage as ndi

from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.irregular.scan import _shape, _descriptor, _pair_vectorized, _overlap
from shardelix.reconstruction.irregular.eroded import _longest_run


def offset_hypotheses(shapes, radius):
    """Return rigid hypotheses in original, unexpanded fragment coordinates."""
    pad = radius + 3
    work = []
    for shape in shapes:
        mask = np.pad(shape.fragment.mask, pad)
        distance, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
        rgba = np.pad(shape.fragment.rgba, ((pad, pad), (pad, pad), (0, 0)))
        rgba = rgba[nearest[0], nearest[1]].copy()
        rgba[..., 3] = np.uint8(distance <= radius) * 255
        work.append(_shape(Fragment(shape.fragment.id, rgba), 2.))
    scales = (10, 16, 24, 32)
    forward = [[_descriptor(s.points, k) for k in scales] for s in work]
    reverse = [[_descriptor(s.points, k, True) for k in scales] for s in work]
    out = []
    for i, a in enumerate(work):
        for j in range(i + 1, len(work)):
            for score, R, t, info in _pair_vectorized(a, work[j], forward[i], reverse[j], 2.7):
                # Convert from the common padded coordinate origin to the
                # measured raster coordinate origin; no scale change.
                t = t + (R - np.eye(2)) @ np.full(2, pad)
                if _overlap(shapes[i], shapes[j], R, t) <= .025:
                    out.append((score, i, j, R, t, {**info, "offset_radius_px": radius}))
    return out


def verify_offset(a, b, fa, fb, R, t, max_gap, info):
    """Verify the pose against real tangents, masks and independent gap fields."""
    from shardelix.reconstruction.irregular.continuation import bridge_evidence

    pos = b.points @ R.T + t
    distance, idx = a.tree.query(pos)
    opposed = -np.sum(a.tangent[idx] * (b.tangent @ R.T), axis=1)
    inward = np.column_stack([-a.tangent[idx, 1], a.tangent[idx, 0]])
    normal = -np.sum(inward * (pos - a.points[idx]), axis=1)
    good = (distance <= max_gap + 3) & (opposed > .5) & (normal >= -2) & (normal >= .55 * distance - 2)
    run = _longest_run(good)
    run = run[good[run]]
    length = len(run) * b.step
    if length < max(25., min(len(a.points)*a.step, len(b.points)*b.step) * .12):
        return None
    ia = idx[run]
    if len(np.unique(ia)) * a.step < length * .5:
        return None
    median = float(np.median(distance[run]))
    dispersion = float(np.median(abs(distance[run] - median)))
    bend = info["seam_bend_rms"]
    if median < 2 or dispersion > max(3., median * .23) or bend < 3 or info["profile_similarity"] < .4:
        return None
    evidence = bridge_evidence(a, b, fa, fb, R, t, max_gap)
    if evidence is None:
        return None
    corr, derivative, error, variance = evidence
    if corr < .55 or derivative < .25 or variance < 8 or error > 35:
        return None
    score = (length * corr**2 * derivative**.8 * math.exp(-error/30-dispersion/5)
             * min(1., variance/20) * min(1., bend/7))
    if score < 4.:
        return None
    return score, {
        "contact_length": float(length), "median_gap": median, "gap_dispersion": dispersion,
        "texture": variance, "continuation_error": error, "bridge_correlation": corr,
        "bridge_derivative_correlation": derivative, "arc_a": int(ia[len(ia)//2]),
        "arc_b": int(run[len(run)//2]), "arc_fraction_a": float(length/(len(a.points)*a.step)),
        "arc_fraction_b": float(length/(len(b.points)*b.step)), "offset_radius_px": info["offset_radius_px"],
        "seam_bend_rms": bend, "evidence": "offset-contour-and-observed-bridge",
        "source": "offset-contour", "families": ["offset-contour", "observed-bridge"],
        "consensus_supported": True,
    }


def refine_offset(a, b, fa, fb, R, t, max_gap, info, value):
    """Refine a verified rigid hypothesis at the moving fragment centre.

    Offset contours nominate poses, but have discretization and rounding
    error. Re-evaluate small SE(2) perturbations against the original masks
    and observed bridge; never modify the measured raster. Keep the incumbent
    unless a candidate strictly improves the same verification objective.
    """
    from shardelix.reconstruction.irregular.scan import rotation
    center = b.points.mean(0)
    best = (value[0], R, t, value[1])
    for distance, angle_step in ((2., .035), (1., .0175)):
        score, Q, u, detail = best
        target = Q @ center + u
        trials = [(Q, u + np.asarray(d)*distance) for d in
                  ((1,0),(-1,0),(0,1),(0,-1))]
        for angle in (-angle_step, angle_step):
            P = rotation(angle) @ Q
            trials.append((P, target-P@center))
        for P, v in trials:
            if _overlap(a,b,P,v) > .025:
                continue
            q = verify_offset(a,b,fa,fb,P,v,max_gap,info)
            if q is None or q[0] <= best[0]*1.005:
                continue
            # An expanded contour encodes a physical two-sided gap estimate.
            # Do not chase a higher photometric score by drifting AWAY from
            # that geometry (e.g. pulling nearby texture into a false seam).
            radius=info.get("offset_radius_px")
            if radius is not None and "median_gap" in value[1]:
                target=2.*radius
                if abs(q[1]["median_gap"]-target)>abs(value[1]["median_gap"]-target)+.2:
                    continue
            best = (q[0],P,v,q[1])
    return best


def offset_recovery(shapes, fields, max_gap, budget, update=lambda _: None):
    """Budgeted multi-radius search, with global competition on each real arc."""
    from shardelix.reconstruction.irregular.continuation import pose_difference

    n = len(shapes)
    median = float(np.median([min(s.fragment.mask.shape) for s in shapes]))
    base = max(2, min(4, round(median*.035)))
    radii = [r for r in range(base, base*5, base) if 2*r <= max_gap+4]
    pair_work = n*(n-1)//2*len(radii)
    # This exhaustive offset preflight limits *work*, not the input archive.
    # The main sparse continuation route remains available on larger cases.
    pair_cap = min(1200, budget.gap_pair_budget)
    pixels = sum((s.fragment.mask.shape[0]+2*(base*4+3)) *
                 (s.fragment.mask.shape[1]+2*(base*4+3)) for s in shapes)
    diag = {"attempted": False, "pair_work": pair_work, "pair_budget": pair_cap, "radii_px": radii}
    if pair_work > pair_cap or pixels*96 > budget.memory_bytes//3:
        return [], {**diag, "reason": "offset search budget"}
    if not radii:
        return [], {**diag, "reason": "insufficient gap range"}
    verified = []
    hypotheses = 0
    for radius in radii:
        update(f"Testing lost-boundary offset {radius} px")
        candidates = offset_hypotheses(shapes, radius)
        hypotheses += len(candidates)
        for _, i, j, R, t, info in candidates:
            q = verify_offset(shapes[i], shapes[j], fields[i], fields[j], R, t, max_gap, info)
            if q:
                # Only already verified hypotheses receive bounded local refinement.
                score, R, t, detail = refine_offset(
                    shapes[i], shapes[j], fields[i], fields[j], R, t, max_gap, info, q
                )
                verified.append((score, i, j, R, t, {**detail, "offset_pose_refined": score > q[0]}))
    # Nearby poses at different trial radii are not independent matches.
    modes = []
    by_pair = defaultdict(list)
    incident = defaultdict(list)
    for c in sorted(verified, key=lambda q: -q[0]):
        _, i, j, R, t, _ = c
        if any(d < 9 and a < .16 for d, a in
               (pose_difference(shapes, j, R, t, z[3], z[4]) for z in by_pair[i,j])):
            continue
        modes.append(c)
        by_pair[i,j].append(c)
        incident[i].append(c)
        incident[j].append(c)
    seeds = []
    for c in modes:
        s, i, j, R, t, info = c
        ratios = []
        for node, side in ((i, "a"), (j, "b")):
            npts = len(shapes[node].points)
            arc, frac = info["arc_"+side], info["arc_fraction_"+side]
            rival = 1.
            for z in incident[node]:
                if z is c:
                    continue
                k = "a" if z[1] == node else "b"
                p, f = z[-1]["arc_"+k], z[-1]["arc_fraction_"+k]
                delta = min(abs(p-arc), npts-abs(p-arc))
                if delta <= npts*max(.05, min(.22, (f+frac)*.32)):
                    rival = max(rival, z[0])
            ratios.append(s/rival)
        uniqueness = min(ratios)
        if uniqueness >= 1.08 and s >= 6.:
            seeds.append((s, i, j, R, t, {**info, "seam_uniqueness": float(uniqueness)}))
    return seeds, {**diag, "attempted": True, "hypotheses": hypotheses,
                   "verified_modes": len(modes), "verified_seeds": len(seeds)}
