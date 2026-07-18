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
from PIL import Image, ImageDraw


CHANNELS = ("R", "G", "B")


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


def detect_markers(rgb, dark_frac=0.10):
    """Find the 4 corner markers + orientation dot in an HxWx3 uint8 image.

    Uses ``V = max(R,G,B)`` so it works over any screen colour (a pure-blue
    screen has low luminance but high V).  The 4 corner markers are the
    OUTERMOST dark blobs -- picked by extreme position, not by area -- because a
    thick, strongly-absorbing cell can go nearly as dark as a marker but is
    always interior to them.  Returns (corners, dot).
    """
    v = rgb.max(axis=2).astype(np.float32)
    thr = max(18.0, dark_frac * float(np.percentile(v, 95)))
    mask = v < thr

    h, w = v.shape
    comps = _components(mask, min_area=max(9, int(0.00002 * h * w)))
    # keep square-ish blobs (markers are squares; the dot is a small square)
    square = [c for c in comps if 0.55 <= c["w"] / max(c["h"], 1) <= 1.8]
    if len(square) < 4:
        raise SystemExit("error: found only %d marker-like blobs (need 4). "
                         "Try adjusting --dark-frac or check the photo." %
                         len(square))
    pts = np.array([[c["cx"], c["cy"]] for c in square], float)
    s, diff = pts[:, 0] + pts[:, 1], pts[:, 0] - pts[:, 1]
    # the 4 outermost blobs: extremes of x+y and x-y (image corners)
    idx = []
    for k in (int(np.argmin(s)), int(np.argmax(s)),
              int(np.argmin(diff)), int(np.argmax(diff))):
        if k not in idx:
            idx.append(k)
    if len(idx) < 4:                                # degenerate: top up by
        centroid = pts.mean(axis=0)                 # distance from centroid
        for k in np.argsort(-((pts - centroid) ** 2).sum(axis=1)):
            if int(k) not in idx:
                idx.append(int(k))
            if len(idx) == 4:
                break
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
    t = np.asarray(thick, float)
    T = np.asarray(trans, float)
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
    return {"a": float(a), "b": float(b), "T0": float(np.exp(b)),
            "r2": float(1 - ss_res / ss_tot), "n": int(len(t))}


# --------------------------------------------------------------------------- #
# Core analysis of one filament (multiple screen photos)
# --------------------------------------------------------------------------- #
def analyze(layout, photos, name="filament", ref_floor_frac=0.18, dark_frac=0.10,
            diag_dir=None):
    """photos: dict screen_colour -> HxWx3 uint8 array.  Returns calibration dict."""
    cells = layout["cells"]
    thick = np.array([c["thickness_mm"] for c in cells], float)
    cell_xy = np.array([[c["cx"], c["cy"]] for c in cells], float)
    cell_wh = np.array([[c["w"], c["h"]] for c in cells], float)
    win = layout["reference_windows"]
    win_xy = np.array([[w["cx"], w["cy"]] for w in win], float)

    screens = {}
    samples = []
    for screen, rgb in photos.items():
        corners, dot = detect_markers(rgb, dark_frac=dark_frac)
        H = order_and_fit(corners, dot, layout)

        # bare-screen reference windows -> brightness plane per channel
        win_rgb = np.array([sample_patch(rgb, H, w["cx"], w["cy"],
                                         2 * w["r"], 2 * w["r"], frac=0.9)
                            for w in win], float)
        good = np.isfinite(win_rgb).all(axis=1)
        planes = fit_reference_plane(win_xy[good], win_rgb[good])

        # which channels does this screen actually light?
        ref_at_cells = eval_plane(planes, cell_xy)           # N x 3
        max_ref = float(np.nanmax(win_rgb))
        floor = ref_floor_frac * max_ref
        lit = eval_plane(planes, win_xy[good]).mean(axis=0) > floor

        per_channel = {}
        cell_rgb = np.array([sample_patch(rgb, H, c["cx"], c["cy"],
                                          c["w"], c["h"]) for c in cells], float)
        for ci, cname in enumerate(CHANNELS):
            if not lit[ci]:
                continue
            ref = ref_at_cells[:, ci]
            T = np.where(ref > floor, cell_rgb[:, ci] / np.clip(ref, 1e-6, None),
                         np.nan)
            fit = fit_absorption(thick, T)
            if fit:
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
            "max_ref": round(max_ref, 1),
        }
        if diag_dir is not None:
            _draw_diag(rgb, H, cells, win, corners, dot,
                       os.path.join(diag_dir, "detect_%s.png" % screen))

    # headline: absorption per display primary (diagonal), falling back to white
    primary = {}
    for cname, screen in (("R", "red"), ("G", "green"), ("B", "blue")):
        src = None
        if screen in screens and cname in screens[screen]["per_channel"]:
            src = screens[screen]["per_channel"][cname]
        elif "white" in screens and cname in screens["white"]["per_channel"]:
            src = screens["white"]["per_channel"][cname]
        if src:
            primary[cname] = round(src["a"], 5)

    cal = {
        "filament": name,
        "model": "ln T = b - a*t  (a = absorption per mm, T0 = exp(b) surface term)",
        "step_mm": layout["step_mm"], "max_mm": layout["max_mm"],
        "primary_absorption_per_mm": primary,
        "screens": screens,
        "samples": samples,
    }
    if diag_dir is not None:
        _draw_curves(screens, os.path.join(diag_dir, "curves.png"),
                     layout["max_mm"])
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

    base = np.array(SCREEN_RGB[screen], float)
    # supersample for clean edges
    ss = 2
    Wi, Hi = W * ss, H * ss
    grad_y = np.linspace(1.0, 1.0 - gradient, Hi)[:, None]     # top brighter
    canvas = np.ones((Hi, Wi, 3), float) * base[None, None, :]
    canvas *= grad_y[..., None]
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    def poly(cx, cy, w, h):
        c = np.array([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                      (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)])
        return [tuple(p * ss) for p in project(Hmm, c)]

    for cell in layout["cells"]:                    # transmittance-tinted cells
        t = cell["thickness_mm"]
        col = []
        for ci, cname in enumerate(CHANNELS):
            m = model[cname]
            col.append(base[ci] * np.exp(m["b"] - m["a"] * t))
        # apply local gradient at the cell centre
        cy_px = project(Hmm, [(cell["cx"], cell["cy"])])[0][1]
        g = 1.0 - gradient * (cy_px / H)
        col = tuple(int(np.clip(v * g, 0, 255)) for v in col)
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
def _load_photo(path):
    return np.asarray(Image.open(path).convert("RGB"))


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

    sy = sub.add_parser("synth", help="render one synthetic pad photo")
    sy.add_argument("--layout", required=True)
    sy.add_argument("--screen", choices=list(SCREEN_RGB), default="white")
    sy.add_argument("--out", required=True)

    opts = p.parse_args(argv)

    if opts.cmd == "selftest":
        return run_selftest(opts.out_dir, tol=opts.tol)

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
                  diag_dir=opts.out_dir)
    out = os.path.join(opts.out_dir, "calibration.json")
    with open(out, "w") as f:
        json.dump(cal, f, indent=2)
    sys.stderr.write("filament '%s': primary absorption per mm = %s\n"
                     % (opts.name, cal["primary_absorption_per_mm"]))
    sys.stderr.write("wrote %s (+ detect_*.png, curves.png)\n" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
