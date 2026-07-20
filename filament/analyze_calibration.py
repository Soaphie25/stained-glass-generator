#!/usr/bin/env python3
"""Analyse photos of a single-filament calibration pad -> per-channel transmittance.

Companion to ``make_calibration_pad.py``.  You print the pad once in a transparent
filament, lay it on a phone/tablet showing a full-screen solid colour, and take one
photo per screen colour (white / red / green / blue).  This tool recovers, for that
filament, how strongly it absorbs light per unit THICKNESS in each channel -- the
Beer-Lambert coefficients the mixture solver needs to predict stacked-filament colour.

Pipeline (per photo):
  1. DETECT the four opaque black corner markers (+ the orientation dot) so we know
     where the pad is, at any rotation / mild perspective tilt.
  2. HOMOGRAPHY from marker positions in pad-millimetres (read from ``layout.json``)
     to their pixels in the photo -- lets us project any pad feature into the image.
  3. SAMPLE every thickness cell (median colour) and every bare-screen reference
     window.
  4. NORMALISE each cell by a plane fitted through the reference windows (divides out
     screen brightness gradient / vignetting / exposure) -> transmittance T in [0,1].
  5. FIT  ln T = b - a*t  per channel: ``a`` = absorption per mm, ``exp(b)`` = the
     thickness-independent surface (Fresnel) transmittance.

Only channels the screen actually lights are fitted (a red screen calibrates the red
channel, etc.); the WHITE screen calibrates all three.

Because we don't have a real printed pad yet, the primary entry point is a SELF-TEST
that renders synthetic photos from a known ground-truth filament and checks the
analyser recovers it:

    python3 filament/analyze_calibration.py selftest --out-dir /tmp/cal

Real use, once you have photos:

    python3 filament/analyze_calibration.py analyze --layout filament/pad/layout.json \
        --name "PolyTerra Teal" \
        --white white.jpg --red red.jpg --green green.jpg --blue blue.jpg \
        --out-dir filament/cal_polyterra_teal
"""
import argparse
import itertools
import json
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CHANNELS = ("R", "G", "B")


# --------------------------------------------------------------------------- #
# sRGB <-> linear light.  Phone JPEGs are gamma-encoded; Beer-Lambert and
# filament stacking are LINEAR-light laws, so we linearise before measuring
# transmittance (median commutes with this monotonic map, so it's fine to apply
# it after sampling the median patch).
# --------------------------------------------------------------------------- #
def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


# --------------------------------------------------------------------------- #
# Geometry: homography (DLT) and point projection
# --------------------------------------------------------------------------- #
def homography(src, dst):
    """3x3 H mapping src points -> dst points (>=4 correspondences), via SVD."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def project(H, pts):
    """Apply homography H to Nx2 points -> Nx2."""
    pts = np.atleast_2d(np.asarray(pts, float))
    P = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return P[:, :2] / P[:, 2:3]


# --------------------------------------------------------------------------- #
# Marker detection: connected components on a "very dark" mask
# --------------------------------------------------------------------------- #
def _components(mask, min_area):
    """4/8-connected components of a boolean mask (BFS over the True pixels).

    Returns list of dicts {area, cx, cy, w, h}.  The mask is expected to be
    sparse (only opaque black markers), so iterating the True pixels is cheap.
    """
    ys, xs = np.where(mask)
    coords = set(zip(xs.tolist(), ys.tolist()))
    seen = set()
    comps = []
    for start in coords:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        pix = []
        while q:
            cx, cy = q.popleft()
            pix.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (cx + dx, cy + dy)
                    if n in coords and n not in seen:
                        seen.add(n)
                        q.append(n)
        if len(pix) >= min_area:
            a = np.asarray(pix, float)
            comps.append({
                "area": len(pix),
                "cx": float(a[:, 0].mean()), "cy": float(a[:, 1].mean()),
                "w": float(np.ptp(a[:, 0]) + 1), "h": float(np.ptp(a[:, 1]) + 1),
            })
    return comps


def _quad_area(p):
    """Area of the quadrilateral through 4 points (ordered by angle)."""
    c = p.mean(axis=0)
    q = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return 0.5 * abs(sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1]
                         for i in range(4)))


def detect_markers(rgb, dark_frac=0.10):
    """Find the 4 corner markers + orientation dot in an HxWx3 uint8 image.

    Uses ``V = max(R,G,B)`` so it works over any screen colour (a pure-blue
    screen has low luminance but high V).  The 4 corner markers are the
    OUTERMOST dark blobs -- picked by extreme position, not by area -- because a
    thick, strongly-absorbing cell can go nearly as dark as a marker but is
    always interior to them.  Returns (corners, dot).
    """
    v = rgb.max(axis=2).astype(np.float32)
    p95 = float(np.percentile(v, 95))
    h, w = v.shape
    min_area = max(9, int(0.00002 * h * w))
    # escalate the darkness threshold until >=4 square blobs appear: matte-black
    # markers read ~0 in a contrasty JPEG but ~30/255 in a linear RAW (lifted
    # blacks).  Extra dark cells that get caught are rejected below.
    square = []
    for frac in (dark_frac, 0.18, 0.28, 0.40, 0.55):
        comps = _components(v < max(8.0, frac * p95), min_area)
        square = [c for c in comps if 0.55 <= c["w"] / max(c["h"], 1) <= 1.8]
        if len(square) >= 4:
            break
    if len(square) < 4:
        raise SystemExit("error: found only %d marker-like blobs (need 4). "
                         "Try --dark-frac or check the photo." % len(square))
    # the 4 corner markers ENCLOSE the largest quadrilateral (cells are interior);
    # robust to perspective + stray dark cells, unlike axis-aligned extremes.
    pts = np.array([[c["cx"], c["cy"]] for c in square], float)
    cand = list(range(len(square)))
    if len(cand) > 14:                              # keep the most outer candidates
        d = ((pts - pts.mean(axis=0)) ** 2).sum(axis=1)
        cand = list(np.argsort(-d)[:14])
    best = None
    for combo in itertools.combinations(cand, 4):
        a = _quad_area(pts[list(combo)])
        if best is None or a > best[0]:
            best = (a, combo)
    idx = list(best[1])
    corners = [square[i] for i in idx]
    rest = [square[i] for i in range(len(square)) if i not in idx]
    # the dot is a SMALL blob (<< marker) sitting next to one corner
    corner_area = np.median([c["area"] for c in corners])
    dot = None
    dot_cands = [c for c in rest if c["area"] < 0.6 * corner_area]
    if dot_cands:
        def nearest_corner_dist(c):
            return min((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2
                       for k in corners)
        dot = min(dot_cands, key=nearest_corner_dist)
    return corners, dot


def order_and_fit(corners, dot, layout):
    """Label the 4 detected corners as TL/TR/BR/BL and return the homography
    (pad-mm -> image-px).

    A quad can be labelled 24 ways; a photo can be rotated OR mirrored, so we
    can't assume a fixed winding.  We simply try every assignment and keep the
    homography that best reprojects the orientation dot (which sits next to the
    top-left corner) -- the correct labelling lands the dot within ~1px, well
    clear of every wrong one.
    """
    reg = layout["register_markers"]["corners"]
    names = ["top_left", "top_right", "bottom_right", "bottom_left"]
    mm = np.array([[reg[n]["cx"], reg[n]["cy"]] for n in names], float)
    dot_mm = [reg["orientation_dot"]["cx"], reg["orientation_dot"]["cy"]]
    pts = np.array([[c["cx"], c["cy"]] for c in corners], float)

    if dot is None:
        # no orientation cue: assume an upright, un-mirrored photo (TL=min x+y,
        # TR=max x-y, BR=max x+y, BL=min x-y in image pixels)
        s, diff = pts.sum(1), pts[:, 0] - pts[:, 1]
        img4 = pts[[np.argmin(s), np.argmax(diff), np.argmax(s), np.argmin(diff)]]
        return homography(mm, img4)

    d = np.array([dot["cx"], dot["cy"]])
    best = None
    for perm in itertools.permutations(range(4)):
        H = homography(mm, pts[list(perm)])
        err = float(np.hypot(*(project(H, [dot_mm])[0] - d)))
        if best is None or err < best[0]:
            best = (err, H)
    return best[1]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_patch(rgb, H, cx, cy, w, h, frac=0.55):
    """Median RGB over the central ``frac`` of a pad feature at (cx,cy) size wxh.

    Projects the feature's shrunk corners into the image and medians the pixels
    in their axis-aligned bounding box (mild perspective -> negligible error).
    """
    hw, hh = w * frac / 2, h * frac / 2
    box_mm = [(cx - hw, cy - hh), (cx + hw, cy - hh),
              (cx + hw, cy + hh), (cx - hw, cy + hh)]
    px = project(H, box_mm)
    H_, W_ = rgb.shape[:2]
    x0 = int(np.clip(np.floor(px[:, 0].min()), 0, W_ - 1))
    x1 = int(np.clip(np.ceil(px[:, 0].max()), 1, W_))
    y0 = int(np.clip(np.floor(px[:, 1].min()), 0, H_ - 1))
    y1 = int(np.clip(np.ceil(px[:, 1].max()), 1, H_))
    if x1 <= x0 or y1 <= y0:
        return np.array([np.nan, np.nan, np.nan])
    region = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    return np.median(region, axis=0)


def fit_reference_plane(win_xy, win_rgb):
    """Least-squares plane R_c(x,y)=p0+p1*x+p2*y per channel over the windows."""
    A = np.column_stack([np.ones(len(win_xy)), win_xy[:, 0], win_xy[:, 1]])
    planes = []
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, win_rgb[:, c], rcond=None)
        planes.append(coef)
    return np.array(planes)                              # 3x3 (channel x coef)


def eval_plane(planes, xy):
    xy = np.atleast_2d(xy)
    A = np.column_stack([np.ones(len(xy)), xy[:, 0], xy[:, 1]])
    return (A @ planes.T)                                # N x 3


# --------------------------------------------------------------------------- #
# Fit  ln T = b - a t   (weighted, returns a, b=intercept, T0, r2, n)
# --------------------------------------------------------------------------- #
def fit_absorption(thick, trans):
    """thick, trans: 1D arrays (transmittance per thickness). Fit ln T=b-a t."""
    t_all = np.asarray(thick, float)
    T_all = np.asarray(trans, float)
    t = t_all
    T = T_all
    ok = np.isfinite(T) & (T > 1e-3) & (T <= 1.5)
    t, T = t[ok], T[ok]
    if len(t) < 3:
        return None
    y = np.log(np.clip(T, 1e-3, None))
    # weight by T (bright, low-noise points count more), one robust reweight pass
    w = np.clip(T, 0.05, 1.0)
    for _ in range(2):
        W = np.diag(w)
        A = np.column_stack([np.ones(len(t)), -t])       # [b, a]
        coef, *_ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
        b, a = coef
        resid = y - (b - a * t)
        s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        w = np.clip(T, 0.05, 1.0) / (1.0 + (resid / (2 * s)) ** 2)
    pred = b - a * t
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    if a <= 0.0:
        # Absorption can't be negative for a passive filter -- a<0 is a degenerate
        # fit.  Two cases, told apart by how bright the thinnest cell is:
        Tmax = float(T.max())
        tmin = float(t[int(np.argmax(T))])            # thickness of brightest cell
        if Tmax < 0.15 and tmin > 1e-6:
            # channel FULLY ABSORBED (floored to black, e.g. red through an
            # intense blue): the log-fit is fitting black-floor noise.  The
            # tightest honest lower bound comes from the THINNEST cell -- if even
            # that reads ~black, a must exceed (ln T0 - ln T_thin)/t_thin.  Noise
            # only ADDS light, so measured T over-estimates true T -> this is a
            # valid lower bound.  Flag it; the exact value is unmeasurable.
            i0 = int(np.argmin(t_all))
            t_thin = max(float(t_all[i0]), 1e-6)
            T_thin = min(max(float(T_all[i0]), 5e-3), 0.15)
            a_lb = (np.log(0.9) - np.log(T_thin)) / t_thin
            return {"a": float(a_lb), "b": float(np.log(0.9)), "T0": 0.9,
                    "r2": 0.0, "n": int(len(t)), "floored": True}
        # else: channel barely absorbs (a wobbled slightly negative on noise) ->
        # clamp to transparent rather than trusting a spurious negative slope.
        b, a = float(np.log(min(Tmax, 1.0))), 0.0
    return {"a": float(a), "b": float(b), "T0": float(np.exp(b)),
            "r2": float(1 - ss_res / ss_tot), "n": int(len(t))}


# --------------------------------------------------------------------------- #
# Core analysis of one filament (multiple screen photos)
# --------------------------------------------------------------------------- #
def _locate(rgb, layout, dark_frac=0.10):
    """Find the pad; return (homography mm->px, chosen 4 corner blobs).

    Robust to a SMALL pad on a BIG screen (the iPad bezel is dark too), dark
    cells, perspective and lifted-black RAW.  Rather than trusting the outermost
    dark blobs, it enumerates candidate 4-marker sets and keeps the homography
    that best EXPLAINS the image: reference windows must land on the bright bare
    screen, cells must be dimmer, and the orientation dot must land on a real
    dark blob.
    """
    v = rgb.max(axis=2).astype(np.float32)
    h, w = v.shape
    p95 = float(np.percentile(v, 95))
    min_area = max(9, int(1.2e-5 * h * w))
    seen = {}                                        # union of square blobs
    for frac in (dark_frac, 0.2, 0.32, 0.48):
        for c in _components(v < max(8.0, frac * p95), min_area):
            if 0.5 <= c["w"] / max(c["h"], 1) <= 2.0:
                k = (round(c["cx"] / 10), round(c["cy"] / 10))
                if k not in seen or c["area"] > seen[k]["area"]:
                    seen[k] = c
    square = list(seen.values())
    if len(square) < 4:
        raise SystemExit("error: found only %d marker-like blobs (need 4)."
                         % len(square))
    pts = np.array([[c["cx"], c["cy"]] for c in square], float)
    reg = layout["register_markers"]["corners"]
    mm = np.array([[reg[n]["cx"], reg[n]["cy"]] for n in
                   ("top_left", "top_right", "bottom_right", "bottom_left")], float)
    dot_mm = [reg["orientation_dot"]["cx"], reg["orientation_dot"]["cy"]]
    # expected marker-rectangle aspect (short/long) -- a strong shape prior that
    # rejects the iPad bezel (~0.75) and stray cell quads.
    ew = np.hypot(*(mm[1] - mm[0]))
    eh = np.hypot(*(mm[3] - mm[0]))
    exp_aspect = min(ew, eh) / max(ew, eh)

    cand = list(range(len(square)))
    if len(cand) > 24:                               # cap combos (markers are
        d = ((pts - pts.mean(0)) ** 2).sum(1)        # among the outermost)
        cand = list(np.argsort(-d)[:24])

    def shape(q):
        c = q.mean(0)
        r = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
        s = [np.hypot(*(r[i] - r[(i + 1) % 4])) for i in range(4)]
        wd, ht = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        asp = min(wd, ht) / max(wd, ht, 1e-6)
        return asp, _quad_area(r), r

    # pick the 4 blobs forming the largest quad whose aspect matches the markers
    best = None
    for combo in itertools.combinations(cand, 4):
        asp, area, ring = shape(pts[list(combo)])
        if area < 0.003 * h * w:
            continue
        score = area * np.exp(-((asp - exp_aspect) / 0.2) ** 2)
        if best is None or score > best[0]:
            best = (score, ring, combo)
    if best is None:
        raise SystemExit("error: could not locate the pad (no plausible marker quad)")
    ring, combo = best[1], best[2]
    other = np.array([pts[i] for i in range(len(square)) if i not in combo])

    # orient: try the 8 labellings (4 rotations x 2 flips); keep the one whose
    # projected orientation dot lands on a real (non-corner) blob.
    bestH = None
    for flip in (ring, ring[::-1]):
        for roll in range(4):
            H = homography(mm, np.roll(flip, roll, axis=0))
            dpx = project(H, [dot_mm])[0]
            ddist = float(np.hypot(*(other - dpx).T).min()) if len(other) else 0.0
            if bestH is None or ddist < bestH[0]:
                bestH = (ddist, H)
    return bestH[1], [square[i] for i in combo]


def _prep(rgb, max_dim, blur_frac, H_probe=None):
    """Downscale a big phone photo (speed + mild anti-alias) and return it."""
    im = Image.fromarray(rgb)
    if max(im.size) > max_dim:
        s = max_dim / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im)


def _sample_cells_linear(layout, rgb0, max_dim, blur_frac):
    """Detect + warp + sample: returns per-cell and per-window LINEAR RGB (0..1),
    already normalised by the reference-window plane -> per-cell transmittance.
    Works on a thickness pad ('cells') or a mixture pad ('pads')."""
    cells = layout.get("cells") or layout["pads"]
    cell_xy = np.array([[c["cx"], c["cy"]] for c in cells], float)
    win = layout["reference_windows"]
    win_xy = np.array([[w["cx"], w["cy"]] for w in win], float)

    rgb = _prep(rgb0, max_dim, blur_frac)
    H, corners = _locate(rgb, layout)
    c0 = cells[0]
    cpx = float(np.hypot(*(project(H, [(c0["cx"] + c0["w"] / 2, c0["cy"])])[0]
                           - project(H, [(c0["cx"] - c0["w"] / 2, c0["cy"])])[0])))
    smooth = np.asarray(Image.fromarray(rgb).filter(
        ImageFilter.GaussianBlur(max(1.0, cpx * blur_frac))))

    win_rgb = srgb_to_linear(np.array(
        [sample_patch(smooth, H, w["cx"], w["cy"], 2 * w["r"], 2 * w["r"], frac=0.9)
         for w in win], float) / 255.0)
    good = np.isfinite(win_rgb).all(axis=1)
    planes = fit_reference_plane(win_xy[good], win_rgb[good])
    ref = eval_plane(planes, cell_xy)
    cell_rgb = srgb_to_linear(np.array(
        [sample_patch(smooth, H, c["cx"], c["cy"], c["w"], c["h"]) for c in cells],
        float) / 255.0)
    T = np.clip(cell_rgb / np.clip(ref, 1e-6, None), 0, 1.5)
    return T, H, corners


def measure(layout, photos, cals=None, max_dim=1600, blur_frac=0.03):
    """Measure each cell's transmittance/colour (no ramp fit).  If ``cals`` (a
    {name: Filament}) is given, also predict each cell from its stack composition
    and report Delta-E.  Returns a list of per-cell result dicts."""
    from solve_recipe import predict_linear, linear_to_hex, linear_to_lab, delta_e
    cells = layout["cells"]
    Ts = {screen: _sample_cells_linear(layout, rgb, max_dim, blur_frac)[0]
          for screen, rgb in photos.items()}
    out = []
    for i, c in enumerate(cells):
        comp = c.get("composition_mm", {})
        row = {"index": c["index"], "composition_mm": comp,
               "thickness_mm": c.get("thickness_mm"),
               "transmittance": {s: [round(float(x), 4) for x in Ts[s][i]]
                                 for s in Ts}}
        if "white" in Ts:
            lin = np.clip(Ts["white"][i], 0, 1)
            row["measured_hex"] = linear_to_hex(lin)
            row["measured_lab"] = [round(float(x), 1) for x in linear_to_lab(lin)]
            if cals and comp and all(n in cals for n in comp):
                models = [cals[n] for n in comp]
                thicks = [comp[n] for n in comp]
                pred = predict_linear(models, thicks)
                row["predicted_hex"] = linear_to_hex(pred)
                row["predicted_lab"] = [round(float(x), 1) for x in linear_to_lab(pred)]
                row["delta_e"] = round(delta_e(lin, pred), 2)
        out.append(row)
    return out


CAPTURE_TIPS = (
    "Capture tips for a clean calibration:\n"
    "  * A single well-exposed WHITE shot is enough -- white is the primary basis\n"
    "    (correct for backlit-white panes); red/green/blue screens are optional.\n"
    "  * RAW (DNG/ARW/...) is PREFERRED -- it's linear and skips the phone's tone\n"
    "    curve, so the absorption numbers are trustworthy (JPEG can inflate them).\n"
    "  * Shoot in a DARK room to kill reflections off the pad's top surface, but\n"
    "    set the SCREEN to MAX, FIXED brightness -- turn OFF auto/adaptive\n"
    "    brightness, True Tone and Night Shift (a dim adaptive screen ruins SNR).\n"
    "  * Keep ISO LOW and lock the camera's exposure + white balance so all four\n"
    "    shots share one exposure; avoid clipping the bare screen to pure white.\n"
    "  * Reference starting point (manual/pro mode): ISO 100, then set the shutter\n"
    "    so the BARE SCREEN reads ~85-95%% (bright but NOT clipped) -- if the cells\n"
    "    look too dark, LENGTHEN the shutter rather than raising ISO. Lock ISO +\n"
    "    shutter + WB and use the same for all four screen colours.\n"
    "  * Fill the frame with the pad, keep it flat and roughly square-on."
)


def _quality_warnings(screens):
    """Flag suspicious shots with actionable re-shoot tips."""
    out = []
    for s, d in screens.items():
        mr, clip = d.get("max_ref", 1.0), d.get("clip_frac", 0.0)
        if mr < 0.25:
            out.append("[%s] UNDER-EXPOSED: brightest reference is only %.0f%% of "
                       "full -- screen too dim (adaptive brightness?) or exposure "
                       "too low. Max the screen brightness / raise exposure."
                       % (s, mr * 100))
        if clip > 0.12:
            out.append("[%s] OVER-EXPOSED: %.0f%% of the bare screen is clipped to "
                       "white -- the reference is saturated. Lower exposure or "
                       "screen brightness so the windows aren't maxed out."
                       % (s, clip * 100))
        for ch, f in d.get("per_channel", {}).items():
            if f.get("floored"):
                out.append("[%s/%s] FULLY ABSORBED: this channel is black through "
                           "even the thinnest cell -- absorption is too strong to "
                           "measure, so a=%.1f/mm is a LOWER BOUND (fine: the "
                           "channel reads ~0 regardless). Expected for an intense "
                           "filament (e.g. red through deep blue)." % (s, ch, f["a"]))
                continue
            # only flag channels that ABSORB meaningfully but fit poorly (a
            # near-zero channel, e.g. red through a red filament, is low-r2 by
            # nature -- nothing to fit -- and must not be flagged as noisy).
            if abs(f.get("a", 0.0)) > 0.25 and f.get("r2", 1.0) < 0.5:
                out.append("[%s/%s] NOISY fit (r2=%.2f) despite strong absorption "
                           "-- likely high ISO / motion blur / reflections. Dark "
                           "room, low ISO, lock focus & exposure, hold steady."
                           % (s, ch, f["r2"]))
    return out


def _reliability(cal):
    """Classify the filament (normal vs intense) and spell out the CAPTURE
    REQUIREMENTS: which screen colours are needed, and the exposure target.

    A NORMAL transparent filament is readable from one white shot -- white carries
    all three channels and each fits cleanly.  An INTENSE filament absorbs one or
    more channels so hard that white can't expose them (they floor to black or fit
    noisily); those channels need their OWN full-brightness primary screen, where
    that channel can be over-exposed without clipping the others.
    """
    screens = cal["screens"]
    wpc = screens.get("white", {}).get("per_channel", {})
    max_ref = screens.get("white", {}).get("max_ref")
    diag_of = {"R": "red", "G": "green", "B": "blue"}
    scr = {"R": "RED", "G": "GREEN", "B": "BLUE"}
    per, need, opaque = {}, [], []
    for c in ("R", "G", "B"):
        provided = diag_of[c] in screens
        f = wpc.get(c) or screens.get(diag_of[c], {}).get("per_channel", {}).get(c)
        if not f:
            per[c] = "not measured"
            continue
        if f.get("floored"):
            per[c] = "fully absorbed -> a=%.1f is a LOWER BOUND" % f["a"]
            (opaque if provided else need).append(c)
        elif abs(f.get("a", 0.0)) > 0.25 and f.get("r2", 1.0) < 0.5:
            per[c] = "noisy: a=%.2f but r2=%.2f" % (f["a"], f["r2"])
            if not provided:
                need.append(c)
        elif abs(f.get("a", 0.0)) < 0.12 or (abs(f.get("a", 0.0)) < 0.25
                                             and f.get("r2", 1.0) < 0.6):
            per[c] = ("~weakly absorbing: a=%.2f (little to fit, low r2 is "
                      "expected -- fine)" % f["a"])
        else:
            per[c] = "reliable: a=%.2f, r2=%.2f" % (f["a"], f.get("r2", 1.0))

    if need:
        color_req = ("shoot WHITE + %s screen%s -- this filament absorbs %s too "
                     "strongly to read from white alone; photograph it over each "
                     "full-brightness single-colour screen so that channel can be "
                     "over-exposed without clipping the others."
                     % (" + ".join(scr[c] for c in need),
                        "s" if len(need) > 1 else "", "/".join(need)))
    else:
        color_req = "WHITE screen only -- a normal transparent filament."
    if opaque:
        color_req += ("  (%s stays fully absorbed even on its own screen: "
                      "genuinely opaque there, the lower bound is fine -- it reads "
                      "~0 in any mix.)" % "/".join(opaque))

    if max_ref is None:
        exp_req = ("bare screen (the reference windows between cells) must read "
                   "~85-95% of full -- bright but NOT clipped. Shoot RAW, dark "
                   "room, ISO 100, screen at max FIXED brightness, lengthen the "
                   "shutter (not ISO) to reach it, lock exposure + WB.")
    else:
        if max_ref < 0.75:
            state = "TOO DIM -- lengthen the shutter (not ISO)"
        elif max_ref > 0.97:
            state = "TOO BRIGHT -- clipping; shorten the shutter"
        else:
            state = "OK"
        exp_req = ("bare screen (reference windows) must read ~85-95%%; yours = "
                   "%.0f%% [%s]. Shoot RAW, dark room, ISO 100, screen at max "
                   "FIXED brightness, lock exposure + WB." % (max_ref * 100, state))

    return {"filament_class": "intense" if need else "normal-transparent",
            "per_channel": per, "color_requirement": color_req,
            "exposure_requirement": exp_req,
            "needs_extra_screens": [scr[c] for c in need]}


def analyze(layout, photos, name="filament", ref_floor_frac=0.18, dark_frac=0.10,
            max_dim=1600, blur_frac=0.03, layer_mm=None, diag_dir=None):
    """photos: dict screen_colour -> HxWx3 uint8 array.  Returns calibration dict.

    layer_mm records the slicer layer height the pad was printed at -- absorption
    of translucent FDM is layer-height dependent, so a calibration only applies to
    prints at the SAME layer height."""
    cells = layout["cells"]
    thick = np.array([c["thickness_mm"] for c in cells], float)
    cell_xy = np.array([[c["cx"], c["cy"]] for c in cells], float)
    cell_wh = np.array([[c["w"], c["h"]] for c in cells], float)
    win = layout["reference_windows"]
    win_xy = np.array([[w["cx"], w["cy"]] for w in win], float)

    screens = {}
    samples = []
    for screen, rgb0 in photos.items():
        rgb = _prep(rgb0, max_dim, blur_frac)
        H, corners = _locate(rgb, layout, dark_frac=dark_frac)

        # blur to smooth print texture (layer lines) before SAMPLING, with a
        # radius tied to the projected cell size so it averages several layer
        # lines but never bleeds across a cell edge.  Detection stays on `rgb`.
        c0 = cells[0]
        cpx = float(np.hypot(*(project(H, [(c0["cx"] + c0["w"] / 2, c0["cy"])])[0]
                               - project(H, [(c0["cx"] - c0["w"] / 2, c0["cy"])])[0])))
        rad = max(1.0, cpx * blur_frac)
        smooth = np.asarray(Image.fromarray(rgb).filter(
            ImageFilter.GaussianBlur(rad)))

        # bare-screen reference windows -> brightness plane per channel
        # (sampled in sRGB, then LINEARISED to real light before any ratios)
        win_rgb = np.array([sample_patch(smooth, H, w["cx"], w["cy"],
                                         2 * w["r"], 2 * w["r"], frac=0.9)
                            for w in win], float)
        win_rgb = srgb_to_linear(win_rgb / 255.0)
        good = np.isfinite(win_rgb).all(axis=1)
        planes = fit_reference_plane(win_xy[good], win_rgb[good])

        # which channels does this screen actually light?
        ref_at_cells = eval_plane(planes, cell_xy)           # N x 3
        max_ref = float(np.nanmax(win_rgb))
        floor = ref_floor_frac * max_ref
        lit = eval_plane(planes, win_xy[good]).mean(axis=0) > floor
        clip_frac = (float((win_rgb[good][:, lit] > 0.97).mean())
                     if good.any() and lit.any() else 0.0)

        per_channel = {}
        cell_rgb = np.array([sample_patch(smooth, H, c["cx"], c["cy"],
                                          c["w"], c["h"]) for c in cells], float)
        cell_rgb = srgb_to_linear(cell_rgb / 255.0)
        for ci, cname in enumerate(CHANNELS):
            if not lit[ci]:
                continue
            ref = ref_at_cells[:, ci]
            T = np.where(ref > floor, cell_rgb[:, ci] / np.clip(ref, 1e-6, None),
                         np.nan)
            fit = fit_absorption(thick, T)
            if fit:
                fit["range"] = round(float(np.nanmax(T) - np.nanmin(T)), 3)
                per_channel[cname] = fit
            for k in range(len(cells)):
                if np.isfinite(T[k]):
                    samples.append({"screen": screen, "channel": cname,
                                    "thickness_mm": float(thick[k]),
                                    "transmittance": float(T[k])})
        screens[screen] = {
            "per_channel": per_channel,
            "markers_px": [{"cx": round(c["cx"], 1), "cy": round(c["cy"], 1)}
                           for c in corners],
            "max_ref": round(max_ref, 3),
            "clip_frac": round(clip_frac, 3),
        }
        if diag_dir is not None:
            _draw_diag(rgb, H, cells, win, corners, None,
                       os.path.join(diag_dir, "detect_%s.png" % screen))

    # headline: per-channel absorption from the WHITE screen -- the physically
    # correct basis for backlit-WHITE panes (white light through the filament ->
    # camera RGB is exactly what a pane does) and the easiest to expose cleanly.
    # A matching colour screen is a fallback only if white wasn't shot; the R/G/B
    # screens are optional extra spectral detail, not required.
    primary = {}
    diag_of = {"R": "red", "G": "green", "B": "blue"}
    for cname in CHANNELS:
        wfit = screens.get("white", {}).get("per_channel", {}).get(cname)
        dfit = screens.get(diag_of[cname], {}).get("per_channel", {}).get(cname)
        src = wfit or dfit
        if src:
            primary[cname] = round(src["a"], 5)

    cal = {
        "filament": name,
        "model": "ln T = b - a*t  (a = absorption per mm, T0 = exp(b) surface term)",
        "layer_height_mm": layer_mm,        # a only valid at THIS print layer height
        "step_mm": layout["step_mm"], "max_mm": layout["max_mm"],
        "primary_absorption_per_mm": primary,
        "screens": screens,
        "samples": samples,
        "warnings": _quality_warnings(screens),
    }
    cal["reliability"] = _reliability(cal)
    if diag_dir is not None:
        _draw_curves(screens, os.path.join(diag_dir, "curves.png"),
                     layout["max_mm"])
        _draw_absorption(screens, samples,
                         os.path.join(diag_dir, "absorption.png"))
    return cal


# --------------------------------------------------------------------------- #
# Diagnostics (pure-PIL, no matplotlib)
# --------------------------------------------------------------------------- #
def _draw_diag(rgb, H, cells, win, corners, dot, path):
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    for c in cells:                                     # sampled cell boxes
        hw, hh = c["w"] * 0.55 / 2, c["h"] * 0.55 / 2
        box = project(H, [(c["cx"] - hw, c["cy"] - hh), (c["cx"] + hw, c["cy"] - hh),
                          (c["cx"] + hw, c["cy"] + hh), (c["cx"] - hw, c["cy"] + hh)])
        d.polygon([tuple(p) for p in box], outline=(255, 255, 0))
    for w in win:                                       # reference windows
        p = project(H, [(w["cx"], w["cy"])])[0]
        d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], outline=(0, 255, 255))
    for c in corners:
        d.ellipse([c["cx"] - 6, c["cy"] - 6, c["cx"] + 6, c["cy"] + 6],
                  outline=(255, 0, 0), width=2)
    if dot is not None:
        d.ellipse([dot["cx"] - 4, dot["cy"] - 4, dot["cx"] + 4, dot["cy"] + 4],
                  outline=(255, 0, 255), width=2)
    img.save(path)


def _draw_curves(screens, path, max_mm):
    cols = {"R": (220, 40, 40), "G": (30, 170, 60), "B": (50, 80, 230)}
    pad, W, H = 46, 300, 210
    order = [s for s in ("white", "red", "green", "blue") if s in screens]
    img = Image.new("RGB", (W * len(order) + 10, H + 20), (250, 250, 252))
    d = ImageDraw.Draw(img)
    for si, screen in enumerate(order):
        ox = si * W + pad
        oy = 10
        pw, ph = W - pad - 12, H - 24
        d.rectangle([ox, oy, ox + pw, oy + ph], outline=(180, 180, 190))
        d.text((ox, oy - 0), screen, fill=(60, 60, 70))

        def X(t):
            return ox + pw * (t / max_mm)

        def Y(T):
            return oy + ph * (1 - np.clip(T, 0, 1))

        for lab in (0.0, 0.5, 1.0):                     # y grid
            d.line([ox, Y(lab), ox + pw, Y(lab)], fill=(230, 230, 235))
            d.text((ox - 26, Y(lab) - 5), "%.1f" % lab, fill=(150, 150, 160))
        for cname, fit in screens[screen]["per_channel"].items():
            col = cols[cname]
            ts = np.linspace(0, max_mm, 40)
            Ts = np.exp(fit["b"] - fit["a"] * ts)
            d.line([(X(t), Y(T)) for t, T in zip(ts, Ts)], fill=col, width=2)
            d.text((ox + 4 + 40 * ("RGB".index(cname)), oy + ph + 4),
                   "%s a=%.2f" % (cname, fit["a"]), fill=col)
    img.save(path)


def _draw_absorption(screens, samples, path):
    """One panel: transmittance vs thickness for the 4 backlights overlaid --
    R/G/B channels over their matching screens (red/green/blue lines) and the
    white screen's luminance (black line).  Data points + fitted curves."""
    W, H = 780, 540
    px0, py0, px1, py1 = 72, 28, W - 130, H - 58
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ts = [s["thickness_mm"] for s in samples]
    tmax = max(ts) if ts else 2.4

    def X(t):
        return px0 + (px1 - px0) * (t / tmax)

    def Y(v):
        return py1 - (py1 - py0) * float(np.clip(v, 0, 1))

    d.rectangle([px0, py0, px1, py1], outline=(170, 170, 180))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        d.line([px0, Y(v), px1, Y(v)], fill=(236, 236, 240))
        d.text((px0 - 34, Y(v) - 6), "%.2f" % v, fill=(120, 120, 130))
    for i in range(int(tmax / 0.5) + 1):
        t = i * 0.5
        d.line([X(t), py0, X(t), py1], fill=(236, 236, 240))
        d.text((X(t) - 8, py1 + 6), "%.1f" % t, fill=(120, 120, 130))
    d.text(((px0 + px1) // 2 - 42, py1 + 30), "thickness (mm)", fill=(60, 60, 70))
    d.text((8, py0 - 2), "transmittance", fill=(60, 60, 70))

    series = [("red R", "red", "R", (210, 40, 40)),
              ("green G", "green", "G", (30, 160, 60)),
              ("blue B", "blue", "B", (50, 80, 230)),
              ("white lum", "white", None, (0, 0, 0))]
    ly = py0 + 6
    for label, screen, ch, col in series:
        if ch is not None:
            pts = sorted((s["thickness_mm"], s["transmittance"]) for s in samples
                         if s["screen"] == screen and s["channel"] == ch)
            fit = screens.get(screen, {}).get("per_channel", {}).get(ch)
        else:                                            # white -> luminance
            wm = {}
            for s in samples:
                if s["screen"] == "white":
                    wm.setdefault(s["thickness_mm"], {})[s["channel"]] = \
                        s["transmittance"]
            pts = sorted((t, 0.2126 * v.get("R", 0) + 0.7152 * v.get("G", 0)
                          + 0.0722 * v.get("B", 0))
                         for t, v in wm.items() if len(v) == 3)
            fit = None
        if not pts:
            continue
        if fit:                                          # smooth fitted curve
            xs = np.linspace(0, tmax, 60)
            ys = np.exp(fit["b"] - fit["a"] * xs)
            d.line([(X(t), Y(v)) for t, v in zip(xs, ys)], fill=col, width=2)
        else:
            d.line([(X(t), Y(v)) for t, v in pts], fill=col, width=2)
        for t, v in pts:
            d.ellipse([X(t) - 2, Y(v) - 2, X(t) + 2, Y(v) + 2], fill=col)
        d.line([px1 + 14, ly + 5, px1 + 34, ly + 5], fill=col, width=3)
        suffix = "" if fit is None else "  a=%.2f" % fit["a"]
        d.text((px1 + 40, ly), label + suffix, fill=(50, 50, 60))
        ly += 20
    img.save(path)


# --------------------------------------------------------------------------- #
# Synthetic photo renderer (ground truth for the self-test)
# --------------------------------------------------------------------------- #
# screen primaries as a camera would see them (small cross-channel leak = realism)
SCREEN_RGB = {
    "white": (250, 250, 248),
    "red":   (238, 26, 24),
    "green": (22, 226, 40),
    "blue":  (26, 34, 240),
}


def synth_photo(layout, model, screen, size=(760, 1500), tilt=0.05, seed=0,
                gradient=0.16, noise=2.5):
    """Render a fake backlit-pad photo for one screen colour.

    model: dict channel -> {"a": per-mm absorption, "b": ln(surface T)}.
    Applies a mild perspective warp + brightness gradient + noise so the
    analyser is exercised on a non-trivial image.  Returns HxWx3 uint8.
    """
    rng = np.random.default_rng(seed + hash(screen) % 1000)
    W, H = size
    pw, ph = layout["pad_w_mm"], layout["pad_h_mm"]

    # place the pad as a slightly tilted quad (pad-mm outline -> image-px)
    mx, my = 0.12 * W, 0.06 * H
    quad = np.array([
        [mx + tilt * W, my],                       # (0,0)   bottom-left in mm
        [W - mx, my + tilt * H],                   # (pw,0)  bottom-right
        [W - mx - tilt * W, H - my],               # (pw,ph) top-right
        [mx, H - my - tilt * H],                   # (0,ph)  top-left
    ], float)
    pad_corners = np.array(layout["pad_corners"], float)
    Hmm = homography(pad_corners, quad)

    # screen LINEAR emission (SCREEN_RGB are the sRGB pixels a camera records of
    # the bare screen); we render in linear light and sRGB-encode like a camera,
    # so the analyser's linearisation recovers the ground-truth absorption.
    L = srgb_to_linear(np.array(SCREEN_RGB[screen], float) / 255.0)
    ss = 2                                          # supersample for clean edges
    Wi, Hi = W * ss, H * ss
    grad_y = np.linspace(1.0, 1.0 - gradient, Hi)[:, None]     # top brighter
    canvas_lin = np.ones((Hi, Wi, 3)) * L[None, None, :] * grad_y[..., None]
    canvas = linear_to_srgb(canvas_lin) * 255.0
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    def poly(cx, cy, w, h):
        c = np.array([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                      (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)])
        return [tuple(p * ss) for p in project(Hmm, c)]

    for cell in layout["cells"]:                    # transmittance-tinted cells
        t = cell["thickness_mm"]
        T = np.array([np.exp(model[c]["b"] - model[c]["a"] * t) for c in CHANNELS])
        cy_px = project(Hmm, [(cell["cx"], cell["cy"])])[0][1]
        g = 1.0 - gradient * (cy_px / H)            # local gradient at the cell
        col = tuple(int(v) for v in
                    np.clip(linear_to_srgb(L * g * T) * 255.0, 0, 255))
        d.polygon(poly(cell["cx"], cell["cy"], cell["w"], cell["h"]), fill=col)

    reg = layout["register_markers"]["corners"]     # opaque black markers + dot
    for nm, m in reg.items():
        d.polygon(poly(m["cx"], m["cy"], m["w"], m["h"]), fill=(6, 6, 8))

    out = img.resize((W, H), Image.LANCZOS)
    arr = np.asarray(out, float)
    if noise:
        arr = arr + rng.normal(0, noise, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Self-test: synth -> analyze -> compare to ground truth
# --------------------------------------------------------------------------- #
GROUND_TRUTH = {   # a teal-ish transparent filament: absorbs red, passes green/blue
    "R": {"a": 0.85, "b": np.log(0.94)},
    "G": {"a": 0.22, "b": np.log(0.95)},
    "B": {"a": 0.40, "b": np.log(0.93)},
}


def run_selftest(out_dir, tol=0.08):
    os.makedirs(out_dir, exist_ok=True)
    layout = _load_or_make_layout(out_dir)

    photos = {}
    for screen in ("white", "red", "green", "blue"):
        arr = synth_photo(layout, GROUND_TRUTH, screen, seed=7)
        Image.fromarray(arr).save(os.path.join(out_dir, "synth_%s.png" % screen))
        photos[screen] = arr

    cal = analyze(layout, photos, name="SYNTH-teal", diag_dir=out_dir)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)

    # compare: white screen (all channels) + primary vector (diagonal screens)
    print("\nself-test: recovered vs ground-truth absorption a (per mm)")
    print("  %-24s %8s %8s %7s" % ("source", "true", "recovered", "err"))
    ok = True
    white = cal["screens"]["white"]["per_channel"]
    for cname in CHANNELS:
        true_a = GROUND_TRUTH[cname]["a"]
        got = white.get(cname, {}).get("a", float("nan"))
        err = abs(got - true_a)
        ok = ok and err <= tol
        print("  white/%-18s %8.3f %8.3f %7.3f %s"
              % (cname, true_a, got, err, "" if err <= tol else "  <-- FAIL"))
    for cname, screen in (("R", "red"), ("G", "green"), ("B", "blue")):
        true_a = GROUND_TRUTH[cname]["a"]
        got = cal["primary_absorption_per_mm"].get(cname, float("nan"))
        err = abs(got - true_a)
        ok = ok and err <= tol
        print("  primary(%s from %-8s %8.3f %8.3f %7.3f %s"
              % (cname, screen + ")", true_a, got, err,
                 "" if err <= tol else "  <-- FAIL"))

    print("\nwrote synth photos, detect_*.png, curves.png, calibration.json to %s"
          % out_dir)
    print("SELF-TEST %s (tol=%.3f)" % ("PASSED" if ok else "FAILED", tol))
    return 0 if ok else 1


def _load_or_make_layout(out_dir):
    """Use an existing pad layout if present, else generate a default one."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(out_dir, "layout.json"),
                 os.path.join(here, "pad", "layout.json")):
        if os.path.exists(cand):
            with open(cand) as f:
                return json.load(f)
    # fall back to generating a default pad layout via the generator module
    sys.path.insert(0, here)
    import make_calibration_pad as mk
    ns = argparse.Namespace(
        screen_w_mm=64.0, screen_h_mm=138.0, margin_mm=3.0, step_mm=0.1,
        max_mm=2.0, cols=4, cell_fill=0.7, min_cell_mm=6.0, edge_mm=2.0,
        header_mm=9.0, base_plate_mm=0.4, marker_mm=6.0,
        marker_inset_mm=1.0, marker_h_mm=0.4, marker_gap_mm=1.5)
    layout, _, _ = mk.build_layout(ns)
    return layout


# --------------------------------------------------------------------------- #
_PIL_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def _load_photo(path):
    """Load a photo as an 8-bit sRGB array.

    Common formats load via PIL.  Anything else is tried as a camera RAW via
    rawpy/libraw (auto-detects DNG/ARW/CR2/CR3/NEF/RW2/...): we develop it LINEAR
    (gamma 1.0, no auto-bright, camera white balance) -- which drops the phone's
    proprietary tone curve and in-camera processing -- then re-encode with a KNOWN
    sRGB curve, so the analyser's gamma decode recovers true linear light exactly.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _PIL_EXT:
        try:
            import rawpy
            with rawpy.imread(path) as r:
                lin = r.postprocess(gamma=(1, 1), no_auto_bright=True,
                                    output_bps=16, use_camera_wb=True)
            lin = lin.astype(np.float64) / 65535.0
            sys.stderr.write("decoded RAW %s (%dx%d) via rawpy -> linear\n"
                             % (os.path.basename(path), lin.shape[1], lin.shape[0]))
            return (np.clip(linear_to_srgb(lin), 0, 1) * 255).astype(np.uint8)
        except ImportError:
            raise SystemExit("error: %s looks like RAW but rawpy isn't installed "
                             "(pip install rawpy)" % path)
        except Exception as e:
            sys.stderr.write("rawpy could not read %s (%s); trying PIL\n"
                             % (path, e))
    return np.asarray(Image.open(path).convert("RGB"))


def run_measure(opts):
    with open(opts.layout) as f:
        layout = json.load(f)
    photos = {s: _load_photo(getattr(opts, s))
              for s in ("white", "red", "green", "blue") if getattr(opts, s)}
    if not photos:
        raise SystemExit("error: pass at least --white (and optionally r/g/b)")
    cals = None
    if opts.cal or opts.cal_dir:
        from solve_recipe import load_filament
        cals = {}
        for spec in opts.cal or []:
            name, path = spec.split("=", 1)
            cals[name] = load_filament(name, path)
        if opts.cal_dir:
            import glob
            for path in sorted(glob.glob(os.path.join(opts.cal_dir, "*",
                                                       "calibration.json"))):
                nm = os.path.basename(os.path.dirname(path))
                cals[nm] = load_filament(nm, path)
    rows = measure(layout, photos, cals=cals)
    os.makedirs(opts.out_dir, exist_ok=True)
    with open(os.path.join(opts.out_dir, "measured.json"), "w") as f:
        json.dump(rows, f, indent=2)

    has_pred = any("predicted_hex" in r for r in rows)
    hdr = "%-22s %-9s" % ("cell (mm)", "measured")
    hdr += " %-9s %6s" % ("predict", "dE") if has_pred else ""
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        lab = "+".join("%s%.2f" % (k[:1], v) for k, v in r["composition_mm"].items()) \
            or "cell %d" % r["index"]
        line = "%-22s #%-8s" % (lab, r.get("measured_hex", "?"))
        if "predicted_hex" in r:
            line += " #%-8s %6.2f" % (r["predicted_hex"], r["delta_e"])
        print(line)
    sys.stderr.write("\nwrote %s\n" % os.path.join(opts.out_dir, "measured.json"))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="synthetic render -> analyse -> verify")
    st.add_argument("--out-dir", default="/tmp/cal_selftest")
    st.add_argument("--tol", type=float, default=0.08)

    an = sub.add_parser("analyze", help="analyse real photos of a printed pad")
    an.add_argument("--layout", required=True, help="path to layout.json")
    an.add_argument("--name", default="filament")
    an.add_argument("--white"), an.add_argument("--red")
    an.add_argument("--green"), an.add_argument("--blue")
    an.add_argument("--out-dir", default="filament/cal_out")
    an.add_argument("--dark-frac", type=float, default=0.10,
                    help="marker darkness threshold as frac of screen brightness")
    an.add_argument("--ref-floor-frac", type=float, default=0.18)
    an.add_argument("--max-dim", type=int, default=1600,
                    help="downscale photos to this max dimension (speed + "
                         "anti-alias; default 1600)")
    an.add_argument("--blur", type=float, default=0.03,
                    help="sampling blur as a fraction of cell size, to smooth "
                         "print layer-line texture (default 0.03; 0 disables)")
    an.add_argument("--layer-mm", type=float, default=None,
                    help="slicer layer height the pad was printed at -- absorption "
                         "is layer-height dependent, so record it and reuse the "
                         "same height for mixture pads + panes")

    sy = sub.add_parser("synth", help="render one synthetic pad photo")
    sy.add_argument("--layout", required=True)
    sy.add_argument("--screen", choices=list(SCREEN_RGB), default="white")
    sy.add_argument("--out", required=True)

    me = sub.add_parser("measure",
                        help="measure each cell's colour (e.g. a stack pad) and, "
                             "with --cal, compare to the stacking prediction")
    me.add_argument("--layout", required=True)
    me.add_argument("--white"), me.add_argument("--red")
    me.add_argument("--green"), me.add_argument("--blue")
    me.add_argument("--cal", action="append",
                    help="name=calibration.json to predict + compare (repeatable)")
    me.add_argument("--cal-dir", help="folder of <name>/calibration.json")
    me.add_argument("--out-dir", default="filament/measured")

    opts = p.parse_args(argv)

    if opts.cmd == "selftest":
        return run_selftest(opts.out_dir, tol=opts.tol)
    if opts.cmd == "measure":
        return run_measure(opts)

    if opts.cmd == "synth":
        with open(opts.layout) as f:
            layout = json.load(f)
        arr = synth_photo(layout, GROUND_TRUTH, opts.screen)
        Image.fromarray(arr).save(opts.out)
        sys.stderr.write("wrote %s\n" % opts.out)
        return 0

    # analyze
    with open(opts.layout) as f:
        layout = json.load(f)
    photos = {}
    for screen in ("white", "red", "green", "blue"):
        path = getattr(opts, screen)
        if path:
            photos[screen] = _load_photo(path)
    if not photos:
        raise SystemExit("error: pass at least one of --white/--red/--green/--blue")
    os.makedirs(opts.out_dir, exist_ok=True)
    cal = analyze(layout, photos, name=opts.name,
                  ref_floor_frac=opts.ref_floor_frac, dark_frac=opts.dark_frac,
                  max_dim=opts.max_dim, blur_frac=opts.blur, layer_mm=opts.layer_mm,
                  diag_dir=opts.out_dir)
    out = os.path.join(opts.out_dir, "calibration.json")
    with open(out, "w") as f:
        json.dump(cal, f, indent=2)
    sys.stderr.write("filament '%s': primary absorption per mm = %s\n"
                     % (opts.name, cal["primary_absorption_per_mm"]))
    sys.stderr.write("wrote %s (+ detect_*.png, curves.png)\n" % out)
    rel = cal["reliability"]
    sys.stderr.write("\nfilament class: %s\n" % rel["filament_class"].upper())
    for c in ("R", "G", "B"):
        sys.stderr.write("  channel %s: %s\n" % (c, rel["per_channel"][c]))
    sys.stderr.write("\nCAPTURE REQUIREMENTS\n")
    sys.stderr.write("  colour   : %s\n" % rel["color_requirement"])
    sys.stderr.write("  exposure : %s\n" % rel["exposure_requirement"])
    if cal["warnings"]:
        sys.stderr.write("\n!! SHOT QUALITY WARNINGS -- results may be unreliable:\n")
        for wmsg in cal["warnings"]:
            sys.stderr.write("  - %s\n" % wmsg)
        sys.stderr.write("\n" + CAPTURE_TIPS + "\n")
    else:
        sys.stderr.write("shot quality looks OK (exposure + fits within range).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
