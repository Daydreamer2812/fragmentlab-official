"""Blind scanned strip reconstruction using pixel continuity and patch recurrence.

No filename order, OCR, original image or supplied order.txt is used.
The rectified pieces are renderable raster strips, not rigid SE(2) poses.
"""
from __future__ import annotations

import time
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter1d, gaussian_filter, grey_opening
from scipy.spatial.distance import cdist

from shardelix.reconstruction.common import Fragment, Pose, render, save_rgba, json_write
from shardelix.reconstruction.strips.ordering import chain, _projection_cost, refine_chain_global
from shardelix.reconstruction.strips.document_patches import _unit_cost, _patch_recurrence, select_cut_hypothesis
from shardelix.reconstruction.rotation import _signed_tree, solve_unknown_color_strips
from shardelix.reconstruction.overlay import write_cut_outputs


class NotDocumentStrips(ValueError):
    """Elongated image shape alone is insufficient to identify a single strip."""


def _rectify(path):
    with Image.open(path) as im:
        a=np.array(ImageOps.exif_transpose(im).convert("RGBA"))
    original_size=[a.shape[1],a.shape[0]]
    if not np.any(a[:,:,3]<128):
        from shardelix.reconstruction.input_detection import solid_background,scan_evidence
        from shardelix.reconstruction.irregular.scan import chroma_mask
        keyed=chroma_mask(a)
        if keyed is None:
            solid=solid_background(a,exact=True)
            keyed=solid[0] if solid is not None else None
        if keyed is not None:
            a[:,:,3]=keyed.astype(np.uint8)*255
        elif scan_evidence(Image.fromarray(a))=="matte-background":
            raise NotDocumentStrips("Unmasked scan with variable illumination requires foreground segmentation")
    had_mask=bool(np.any(a[:,:,3]<128))
    # Trim empty scan margins before deciding the strip axis. A padded
    # photograph with a valid mask should not be rejected as a short strip.
    yy,xx=np.nonzero(a[:,:,3]>=128)
    if not len(xx):raise NotDocumentStrips("Empty foreground mask")
    x0,y0=int(xx.min()),int(yy.min())
    a=a[y0:int(yy.max())+1,x0:int(xx.max())+1].copy()
    turn=int(a.shape[0]<a.shape[1])
    if turn:a=np.rot90(a,-1)
    mask=a[:,:,3]>=128
    valid=mask.any(axis=1)
    if valid.sum()<.96*len(valid):
        raise NotDocumentStrips("Strip silhouette must span nearly all scan rows")
    y=np.arange(len(mask));left=mask.argmax(1);right=mask.shape[1]-1-mask[:,::-1].argmax(1)
    left=np.interp(y,y[valid],left[valid]);right=np.interp(y,y[valid],right[valid])
    width=right-left+1
    W=int(round(np.median(width)))
    # Long, contiguous machine-cut strips can have tapered or worn rims.
    # Allow extra width variation only for very elongated silhouettes;
    # the interior-hole guard below still rejects multi-piece masks.
    tolerance=.38 if len(mask)>=15*max(1,W) else .22
    width_deviation=float(np.percentile(abs(width-W),95))
    # A localized outward rim or folded scan section can inflate just a
    # minority of rows. Keep the dominant width, with a bounded excursion
    # and no substantial inward narrowing. Do not relax compact fragments.
    local_rim=(len(mask)>=15*max(1,W) and
               np.mean(abs(width-W)<=max(3,.22*W))>=.80 and
               width.min()>=.65*W and width.max()<=1.60*W)
    if len(mask)<4*W or W<5 or (width_deviation>max(3,tolerance*W) and not local_rim):
        raise NotDocumentStrips("Scan silhouette is not a near-uniform strip")
    if mask.sum()/max(1,width.sum())<.96:
        raise NotDocumentStrips("Strip mask has substantial interior holes or multiple pieces")
    h=len(mask)
    yy=np.broadcast_to(y[:,None],(h,W))
    xx=left[:,None]+(right-left)[:,None]*np.linspace(0,1,W)
    rgb=np.stack([map_coordinates(a[:,:,k],(yy,xx),order=1,prefilter=False) for k in range(3)],axis=2)
    # Preserve invalid rows as transparent rather than fabricating pixels.
    observed=map_coordinates(mask.astype(np.uint8),(yy,xx),order=0,prefilter=False)>0
    rgba=np.dstack([rgb,observed.astype(np.uint8)*255]);rgba[~observed]=0
    return Fragment(path.stem,rgba),{"file":path.name,"id":path.stem,
        "original_wh":original_size,"rectified_wh":[W,h],"initial_clockwise_turns":turn,
        "masked":had_mask,"crop_xy":[x0,y0],"median_width":W,"row_width_range":[int(width.min()),int(width.max())],
        "localized_rim":bool(local_rim and width_deviation>max(3,tolerance*W))}


def _ink(p):
    a=p.rgba
    v=1-(.299*a[:,:,0].astype(np.float32)+.587*a[:,:,1]+.114*a[:,:,2])/255
    v[a[:,:,3]==0]=0
    return v


def _clean_ink(im):
    # Estimate paper/shadow intensity separately in every scan column.
    # No pixels are changed in the exported image.
    base=np.percentile(im,20,axis=0)
    return np.clip((im-base)/.65,0,1).astype(np.float32)


def _local_ink(im):
    """Remove spatially varying paper shadow in matching evidence only.

    A vertical morphological opening spans a typical text stroke while
    preserving column-wise illumination. Smooth the estimate, then normalize
    ink contrast against the local paper level. Exported pixels are untouched.
    """
    window=max(15,min(61,int(round(len(im)/44))|1))
    bg=gaussian_filter(grey_opening(im,size=(window,1)),(window//4,.5))
    bg=np.maximum(bg,.8*np.percentile(im,20,axis=0)[None,:])
    return np.clip((im-bg)/np.maximum(.35,.65*(1-bg)),0,1).astype(np.float32)

def _stroke_overlap(left,right):
    """Normalized ink mismatch under small vertical seam misregistration."""
    L=gaussian_filter1d(left,2.5,axis=1);R=gaussian_filter1d(right,2.5,axis=1)
    def cost(a,b):
        return cdist(a,b,"cityblock")/(a.sum(1)[:,None]+b.sum(1)[None,:]+1)
    score=cost(R,L)
    for d in (2,4,6):
        if d>=L.shape[1]//4:break
        score=np.minimum(score,cost(R[:,d:],L[:,:-d])+.009*d)
        score=np.minimum(score,cost(R[:,:-d],L[:,d:])+.009*d)
    return _unit_cost(score)


def _shift_rows(im,offset):
    if not offset:return im
    out=np.zeros_like(im)
    if offset>0:out[offset:]=im[:-offset]
    else:out[:offset]=im[-offset:]
    return out


def _align_text_rows(images):
    """Estimate bounded vertical registration from shared printed line rows.

    Only matching evidence is shifted. A near-zero or weakly supported
    solution leaves the original profiles unchanged; no raster is invented.
    """
    n=len(images);h=len(images[0])
    offsets=np.zeros(n,dtype=int)
    diag={"applied":False,"shifted_count":0,"max_abs_shift_px":0}
    if n<8 or h<100:return offsets,diag
    limit=min(12,max(6,round(.007*h)))
    profiles=np.stack([gaussian_filter1d(im.mean(axis=1),1.5) for im in images])
    profiles-=gaussian_filter1d(profiles,8,axis=1)
    norms=np.linalg.norm(profiles,axis=1)
    valid=norms>.05
    if valid.sum()<5:return offsets,diag
    profiles/=np.maximum(norms[:,None],1e-6)
    consensus=np.median(profiles[valid],axis=0)
    a=consensus[limit:h-limit].copy()
    a/=max(float(np.linalg.norm(a)),1e-9)
    shifts=np.arange(-limit,limit+1)
    quality=np.zeros(n)
    for k,q in enumerate(profiles):
        scores=np.array([a@q[limit-d:h-limit-d]/max(float(np.linalg.norm(q[limit-d:h-limit-d])),1e-9)
                         for d in shifts])
        at=int(np.argmax(scores-.002*np.abs(shifts)))
        quality[k]=scores[at]
        if scores[at]>.28 and scores[at]-scores[limit]>.035:offsets[k]=shifts[at]
    if np.count_nonzero(quality>.28)<max(5,n//2):offsets[:]=0
    offsets-=int(np.median(offsets))
    # A few one-pixel disagreements may be different character shapes, not
    # registration. Apply only if a material fraction needs a real shift.
    if np.count_nonzero(np.abs(offsets)>=3)<max(2,math.ceil(.15*n)):offsets[:]=0
    diag.update(applied=bool(offsets.any()),shifted_count=int(np.count_nonzero(offsets)),
                max_abs_shift_px=int(np.max(np.abs(offsets))),row_consensus=float(np.median(quality)))
    return offsets,diag


def _textlike(pieces,images):
    a=np.concatenate([p.rgba[p.mask][::max(1,int(p.mask.sum())//800),:3] for p in pieces])
    dark=np.concatenate([im.ravel()[::max(1,im.size//800)] for im in images])
    return bool(len(a) and (np.ptp(a.astype(np.int16),axis=1)<35).mean()>.85
                and np.median(dark)<.30 and .012<(dark>.40).mean()<.40)


def _line_cost(left,right):
    """Remove common row layout; compare local ink rather than blank paper."""
    def compare(l,r):
        l=l-l.mean(0);r=r-r.mean(0)
        cosine=cdist(r+1.e-5,l+1.e-5,"cosine")
        absolute=cdist(r,l,"cityblock")/(np.abs(r).sum(1)[:,None]+np.abs(l).sum(1)[None,:]+1)
        return cosine,absolute
    a,b=compare(left,right)
    for d in (1,2):
        for l,r in ((left[:,d:],right[:,:-d]),(left[:,:-d],right[:,d:])):
            u,v=compare(l,r)
            a=np.minimum(a,u+.007*d);b=np.minimum(b,v+.007*d)
    return .5*_unit_cost(a)+.5*_unit_cost(b)


def _matching_scale(im,h,x_scale):
    """Resize only the matching raster; preserve the observed export.

    A 300 dpi scan often has twice the working height and width. Scaling
    only y distorts printed strokes and the assumed missing-cut widths.
    Use a common x scale for strips of varying rectified widths.
    For equal-width cuts retain the native horizontal detail.
    """
    w=max(2,int(round(im.shape[1]*x_scale)))
    if (h,w)==im.shape:return im
    if x_scale<.99:
        sigma=.5*math.sqrt(max(0.,1/x_scale**2-1))
        im=gaussian_filter(im,sigma)
    y=np.broadcast_to(np.linspace(0,len(im)-1,h)[:,None],(h,w))
    x=np.broadcast_to(np.linspace(0,im.shape[1]-1,w),(h,w))
    return map_coordinates(im,[y,x],order=1,prefilter=False)


def _stroke_order(images,profile,update):
    # Common-height normalized evidence; only estimation, no resizing of output.
    height=max(map(len,images));h=min(1800,height);match_scale=h/height
    x_scale=match_scale if len({im.shape[1] for im in images})>1 else 1.0
    cleaned=[_matching_scale(_clean_ink(im),h,x_scale) for im in images]
    offsets,alignment=_align_text_rows(cleaned) if profile!="fast" else (
        np.zeros(len(cleaned),dtype=int),{"applied":False,"shifted_count":0,"max_abs_shift_px":0})
    if offsets.any():cleaned=[_shift_rows(im,int(d)) for im,d in zip(cleaned,offsets)]
    background=float(np.median([np.percentile(im,20) for im in images]))
    # The artifact band on gray/shadowed scans is less reliable than deeper
    # ink. A near-white scan retains more high-frequency boundary detail.
    inset,depth=(1,2) if background<.035 else (3,4)
    def side(im,right):
        if right:im=im[:,::-1]
        a=min(inset,max(0,im.shape[1]-2))
        return gaussian_filter1d(im[:,a:a+depth].mean(1),.9)
    L,R=[np.asarray([side(im,right) for im in cleaned]) for right in (False,True)]
    c=_line_cost(L,R)
    patch=False;optimization={"status":"not-run"};margin_used=True
    cut_hypotheses={"selected":"not-run","candidates":[]}
    order=None
    l,r=L.mean(1),R.mean(1);scale=max(.02,float(np.median(np.r_[l,r])))
    borders=(3*l/scale,3*r/scale)
    if profile!="fast" and len(images)>=8 and h>=48:
        update("Checking stroke continuation with locally shadow-corrected patches")
        half,gap=(6,0) if inset==1 else (8,2)
        # Preserve the original column-normalized ink for direct matching:
        # the local shadow estimate is most useful for interior-patch search.
        local=[_matching_scale(_local_ink(im),h,x_scale) for im in images]
        if offsets.any():local=[_shift_rows(im,int(d)) for im,d in zip(local,offsets)]
        # The original recurrence remains available. Reuse its predictions
        # for robust residuals only on tall varying-width physical scans.
        alt={} if x_scale<.99 and h>=1500 and min(im.shape[1] for im in local)>=20 else None
        q=_patch_recurrence(local,half,gap,alternatives=alt)
        if q is not None:
            # Match a narrow stroke profile, tolerating scan registration
            # errors; broader means can erase actual interrupted strokes.
            overlap_depth=1 if inset==1 else 3
            def narrow(im,right):
                if right:im=im[:,::-1]
                a=min(inset,max(0,im.shape[1]-2))
                return gaussian_filter1d(im[:,a:a+overlap_depth].mean(1),.9)
            NL,NR=[np.asarray([narrow(im,right) for im in cleaned]) for right in (False,True)]
            overlap=_stroke_overlap(NL,NR)
            c=.30*overlap+.70*q;patch=True
            # Do not force a blank margin for a cropped printed page, a
            # shadowed edge or an ambiguous interior gutter. Require clear
            # near-white evidence on both outer sides before adding a prior.
            margin_used=bool(l.min()<.02 and r.min()<.02)
            if not margin_used:borders=(np.zeros_like(l),np.zeros_like(r))
            # The extra bank is designed for high-resolution physical scans;
            # a 2–3 pixel loss model is not scale-invariant on thumbnails.
            if h>=1500 and min(im.shape[1] for im in local)>=20:
                mixtures={name:.30*overlap+.70*matrix for name,matrix in (alt or {}).items()}
                c,order,cut_hypotheses=select_cut_hypothesis(local,c,borders,update,alternatives=mixtures)
    np.fill_diagonal(c,np.inf)
    if order is None:order=chain(c,.5,borders=borders)
    if patch:
        update("Verifying a single globally consistent strip chain")
        order,optimization=refine_chain_global(c,order,borders,top_k=12 if profile=="accurate" else 8,
                                               seconds=3.0 if profile=="accurate" else 1.5)
    # Reciprocal ranking is a qualitative warning, not a probability or GT.
    rowrank=np.argsort(np.argsort(c,axis=1),axis=1)
    colrank=np.argsort(np.argsort(c,axis=0),axis=0)
    a,b=order[:-1],order[1:]
    return order,{"patch_recurrence":patch,"edge_inset_px":inset,"profile_depth_px":depth,
                  "reciprocal_top3_joins":int(np.count_nonzero((rowrank[a,b]<3)&(colrank[a,b]<3))),
                  "total_joins":len(a),"local_shadow_normalization":patch,
                  "margin_prior_used":margin_used,"global_path":optimization,"vertical_registration":alignment,
                  "cut_loss_hypotheses":cut_hypotheses,"matching_scale":float(match_scale),"matching_x_scale":float(x_scale)}


def reconstruct_strips(paths,target:Path,out:Path,update=lambda _:None,*,pixel_cap=8_000_000,profile="balanced"):
    if not 2<=len(paths)<=128:raise ValueError("Physical document mode supports 2–128 scanned strips")
    started=time.perf_counter();pieces=[];records=[]
    for path in paths:
        p,r=_rectify(path);pieces.append(p);records.append(r)
    if len({p.id for p in pieces})!=len(pieces):raise ValueError("Strip filenames must have unique stems")
    heights=[p.rgba.shape[0] for p in pieces]
    h=max(heights);w=sum(p.rgba.shape[1] for p in pieces)
    if h-min(heights)>max(3,.035*h):raise NotDocumentStrips("Rectified strips lack comparable long dimensions")
    if h*w>pixel_cap:raise ValueError("Rectified document exceeds canvas pixel budget")
    target.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for p in pieces:save_rgba(target/(p.id+".png"),p.rgba)
    images=[_ink(p) for p in pieces]
    update("Matching continuity along scanned strip edges")
    if _textlike(pieces,images):
        # Retain the full cleaned row projection. Cropping to the middle of
        # each narrow strip discarded the few signals linking page columns and
        # sometimes reversed an entire region independently.
        same,flipped=_projection_cost([_clean_ink(im) for im in images])
        signs=_signed_tree(same,flipped,neighbors=12)
        ims=[np.rot90(im,2) if signs[i] else im for i,im in enumerate(images)]
        order,matching=_stroke_order(ims,profile,update)
        poses={};x=0
        for i in order:
            p=pieces[i];poses[p.id]=Pose(x,0,2*int(signs[i]));x+=p.rgba.shape[1]
        matching["relative_orientation_flips"]=int(np.count_nonzero(signs))
        algorithm="physical-document-foreground-multiscale"
    else:
        # A long photo strip is not an ink document. Retain chromatic evidence.
        poses,color=solve_unknown_color_strips(pieces)
        matching={"patch_recurrence":False,"color_aware":True}
        algorithm="physical-color-strip-chain"
    update("Rendering rectified scanned document")
    rgba,labels,collision=render(pieces,poses,(h,w))
    save_rgba(out/"assembly.png",rgba);write_cut_outputs(out,rgba,labels)
    warnings=["Physical strip order is experimental. Full placement does not establish correct adjacency.",
              "Global orientation is ambiguous in blind pixel-only matching; minor scan alignment and edge damage may remain."]
    if any(not r["masked"] for r in records):warnings.append("Some inputs had no foreground mask; their entire raster was treated as paper.")
    diag={"algorithm":algorithm,"status":"unverified",
          "placed":len(poses),"piece_count":len(pieces),"coverage":float((labels>=0).mean()),
          "render_collision_pixels":collision,"runtime_seconds":time.perf_counter()-started,
          "accuracy_verified":False,"orientation_ambiguous":True,"source_reference_used":False,
          "canvas_height":h,"canvas_width":w,
          "scan":{"kind":"document-strips","input_images":len(paths),"detected_pieces":len(pieces),
                  "components":1,"matching":matching,"warnings":warnings,"common_scale":1.0,"accuracy_verified":False}}
    json_write(out/"placements.json",{"mode":"blind-rectified-strips","poses":{k:v.as_dict() for k,v in poses.items()},
              "diagnostics":diag})
    json_write(out/"scan_report.json",{"input_images":len(paths),"detected_pieces":len(pieces),
        "mode":"document-strips","rectification":"per-row linear sampling between foreground mask edges",
        "pieces":records,"matching":matching,"orientation_ambiguous":True,"accuracy_verified":False,"warnings":warnings})
    tile=140;cols=min(8,len(pieces));rows=math.ceil(len(pieces)/cols)
    preview=Image.new("RGB",(cols*tile,rows*tile),(225,231,228));draw=ImageDraw.Draw(preview)
    for i,p in enumerate(pieces):
        im=Image.fromarray(p.rgba);im.thumbnail((tile-12,tile-24))
        dx=(i%cols)*tile+(tile-im.width)//2;dy=(i//cols)*tile+(tile-20-im.height)//2
        preview.paste(im,(dx,dy),im);draw.text(((i%cols)*tile+6,(i//cols+1)*tile-17),p.id,fill=(40,55,45))
    preview.save(out/"fragments_preview.png")
    json_write(out/"diagnostics.json",diag)
    return diag
