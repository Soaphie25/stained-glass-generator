#!/usr/bin/env python3
"""Convert a PNG raster image into stained-glass fabrication artifacts.

Given one PNG image, this tool produces three outputs intended for a
stained-glass 3D-printing pipeline (printer nozzle = 0.4 mm, so 1 px = 0.4 mm
by default and the SVGs carry physical mm units):

  1. ``<input>_silhouette.svg`` - the outline of all non-transparent pixels:
     the outer boundary plus any interior holes, as one path with even-odd
     fill.  Aborts if the opaque pixels form 2+ disconnected regions.

  2. ``<input>_leading.svg`` - the black "leading" lines (the segmentation
     between glass panes), vectorised into a closed, filled Bezier outline.
     The black pixels are skeletonised to centrelines, each centreline is
     given a single uniform width measured from the pixels, the centrelines
     are offset to ribbons and boolean-unioned (so crossings merge cleanly),
     and the union boundary is re-fitted with minimal-control-point Beziers.
     A 1 px x 10 px black line yields a 0.4 mm x 4 mm filled rectangle.

  3. ``<input>_fragments.png`` - the glass fragments = silhouette minus the
     leading.  Colours are kept from the input (``--fragment-color original``)
     or quantised into N groups (``--fragment-color quantized``).

Usage:
    python3 scripts/png_to_stained_glass_svg.py input.png [options]
"""

import argparse
import glob
import os
import re
import sys

# --- Third-party dependencies (guarded for a clear error message) ----------
try:
    import numpy as np
    from PIL import Image, ImageDraw
    import cv2
    from scipy import ndimage
    from scipy.cluster.vq import kmeans2
    from skimage.measure import label
    from skimage.morphology import skeletonize
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover - environment guard
    sys.stderr.write(
        "error: missing dependency (%s).\n"
        "Install with: pip install pillow numpy scipy scikit-image "
        "opencv-python shapely\n" % exc
    )
    sys.exit(3)


class AbortError(Exception):
    """Raised for unrecoverable conditions that must stop the program."""


# The printer has 4 base colour slots but supports simple colour mixing, so a
# few more distinct colours are printable.  Switching/mixing is still costly,
# so the fragments image is capped at this many distinct colours.
MAX_PRINT_COLORS = 24
DEFAULT_PRINT_COLORS = 12
# Glass-adjacent panes (no black between) closer than this in RGB are merged --
# they are one band split into shades by anti-aliasing, not a real colour pane.
COLOR_MERGE_TOL = 110.0
# In 'tier' mode, boost a stroke's effective width before the bold cut, so a
# borderline MAIN/FOREGROUND outline leans BOLD.  LENGTH: up to this fraction for
# the longest chains (short thin splitters stay thin).  BLOCK: up to this fraction
# for a line running fully alongside a black block (a garment outline whose black
# the block partly absorbs, so it measures thin).  0 = pure width tiering.
LENGTH_TIER_BIAS = 0.4
BLOCK_TIER_BIAS = 1.0


# ===========================================================================
# IO / masks
# ===========================================================================
def load_rgba(path):
    """Load a PNG as an HxWx4 uint8 RGBA array."""
    try:
        img = Image.open(path).convert("RGBA")
    except (FileNotFoundError, OSError) as exc:
        raise AbortError("cannot read image '%s': %s" % (path, exc))
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise AbortError("image '%s' has unexpected/zero size" % path)
    return arr


def opaque_mask(rgba, alpha_min):
    """Boolean mask of non-transparent pixels (alpha >= alpha_min)."""
    return rgba[:, :, 3] >= alpha_min


def black_mask(rgba, opaque, lum_thresh, min_area):
    """Opaque + near-black pixels, with small speck/noise components removed.

    'Near-black' means dark in *every* channel (max(R,G,B) < threshold), which
    captures the achromatic black leading while excluding saturated dark
    colours such as navy (e.g. dark-blue stars) that would otherwise be
    misread as leading.  Components with fewer than ``min_area`` pixels
    (8-connected) are dropped so specks never become leading.
    """
    mask = opaque & (rgba[:, :, :3].max(axis=2) < lum_thresh)
    if min_area > 1 and mask.any():
        lbl, n = label(mask, connectivity=2, return_num=True)
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= int(min_area)
        keep[0] = False  # background
        mask = keep[lbl]
    return mask


def split_black_lines_blocks(black, block_thick_px):
    """Split the black mask into thin LEADING lines and thick BLOCK regions.

    A solid black area (e.g. a dark garment) should print as a BLACK GLASS
    fragment, not as leading; only the thin black lines are leading.  A
    morphological OPENING (erode then dilate by a disk of radius ~half the
    threshold) deletes everything thinner than ``block_thick_px`` while
    restoring the thick regions to full size -- so the opening IS the blocks and
    the thin lines are what it removed.  (Do NOT morphologically reconstruct: a
    block usually TOUCHES the leading network, and reconstruction would flood the
    whole connected black net into one 'block'.)  Small opening specks are
    dropped.  Returns ``(line_mask, block_mask)``; ``block_thick_px <= 0``
    disables it (all black stays leading).
    """
    if block_thick_px is None or block_thick_px <= 0 or not black.any():
        return black, np.zeros_like(black)
    r = max(1, int(round(block_thick_px / 2.0)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    block = ndimage.binary_dilation(
        ndimage.binary_erosion(black, disk), disk)   # opening
    if block.any():                                  # drop tiny opening specks
        lbl, n = label(block, connectivity=2, return_num=True)
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= max(1, int(np.pi * r * r))   # >= ~one disk
        keep[0] = False
        block = keep[lbl]
    # Smooth the block boundary: the opening leaves a scalloped/zig-zag edge
    # (from the disk + pixel staircase) that a Bezier fit can't fully remove
    # once it becomes a partition seam.  Gaussian-blur the mask and re-threshold
    # so the block->glass seam is a smooth curve.
    if block.any():
        sigma = max(1.0, r * 0.75)
        block = ndimage.gaussian_filter(block.astype(np.float32), sigma) > 0.5
    line = black & ~block
    return line, block


# ===========================================================================
# Minimal-control-point cubic Bezier fitting (Schneider, Graphics Gems 1990)
# ===========================================================================
def _q(ctrl, t):
    """Evaluate a cubic Bezier at parameter t."""
    mt = 1.0 - t
    return (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] \
        + 3 * mt * (t ** 2) * ctrl[2] + (t ** 3) * ctrl[3]


def _q_prime(ctrl, t):
    mt = 1.0 - t
    return 3 * (mt ** 2) * (ctrl[1] - ctrl[0]) \
        + 6 * mt * t * (ctrl[2] - ctrl[1]) \
        + 3 * (t ** 2) * (ctrl[3] - ctrl[2])


def _q_prime2(ctrl, t):
    mt = 1.0 - t
    return 6 * mt * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) \
        + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _chord_length_parameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i - 1] + np.linalg.norm(points[i] - points[i - 1]))
    total = u[-1]
    if total == 0:
        return np.linspace(0.0, 1.0, len(points))
    return np.asarray(u) / total


def _generate_bezier(points, u, left_tan, right_tan):
    bez = [points[0], None, None, points[-1]]
    a = np.zeros((len(u), 2, 2))
    for i, ui in enumerate(u):
        a[i][0] = left_tan * (3 * (1 - ui) ** 2 * ui)
        a[i][1] = right_tan * (3 * (1 - ui) * ui ** 2)

    c = np.zeros((2, 2))
    x = np.zeros(2)
    line = [points[0], points[0], points[-1], points[-1]]
    for i, ui in enumerate(u):
        c[0][0] += np.dot(a[i][0], a[i][0])
        c[0][1] += np.dot(a[i][0], a[i][1])
        c[1][0] += np.dot(a[i][0], a[i][1])
        c[1][1] += np.dot(a[i][1], a[i][1])
        tmp = points[i] - _q(line, ui)
        x[0] += np.dot(a[i][0], tmp)
        x[1] += np.dot(a[i][1], tmp)

    det_c0_c1 = c[0][0] * c[1][1] - c[1][0] * c[0][1]
    det_c0_x = c[0][0] * x[1] - c[1][0] * x[0]
    det_x_c1 = x[0] * c[1][1] - x[1] * c[0][1]

    alpha_l = 0.0 if det_c0_c1 == 0 else det_x_c1 / det_c0_c1
    alpha_r = 0.0 if det_c0_c1 == 0 else det_c0_x / det_c0_c1

    seg_len = np.linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * seg_len
    if alpha_l < epsilon or alpha_r < epsilon:
        bez[1] = bez[0] + left_tan * (seg_len / 3.0)
        bez[2] = bez[3] + right_tan * (seg_len / 3.0)
    else:
        # Clamp control-point overshoot to the chord length: large alpha values
        # make the cubic bulge past the data and can self-intersect adjacent
        # segments (spikes / "holed rectangle" artefacts on complex rings).
        alpha_l = min(alpha_l, seg_len)
        alpha_r = min(alpha_r, seg_len)
        bez[1] = bez[0] + left_tan * alpha_l
        bez[2] = bez[3] + right_tan * alpha_r
    return bez


def _reparameterize(bez, points, u):
    out = []
    for point, ui in zip(points, u):
        d = _q(bez, ui) - point
        num = np.dot(d, _q_prime(bez, ui))
        den = np.dot(_q_prime(bez, ui), _q_prime(bez, ui)) \
            + np.dot(d, _q_prime2(bez, ui))
        out.append(ui if den == 0 else ui - num / den)
    return np.asarray(out)


def _compute_max_error(points, bez, u):
    max_dist = 0.0
    split = len(points) // 2
    for i, (point, ui) in enumerate(zip(points, u)):
        dist = np.linalg.norm(_q(bez, ui) - point) ** 2
        if dist > max_dist:
            max_dist = dist
            split = i
    return max_dist, split


def _fit_cubic(points, left_tan, right_tan, error):
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.0
        return [[points[0], points[0] + left_tan * dist,
                 points[1] + right_tan * dist, points[1]]]

    u = _chord_length_parameterize(points)
    bez = _generate_bezier(points, u, left_tan, right_tan)
    max_err, split = _compute_max_error(points, bez, u)

    if max_err < error:
        return [bez]

    if max_err < error * error:
        for _ in range(20):
            u_prime = _reparameterize(bez, points, u)
            bez = _generate_bezier(points, u_prime, left_tan, right_tan)
            max_err, split = _compute_max_error(points, bez, u_prime)
            if max_err < error:
                return [bez]
            u = u_prime

    # Guard against degenerate split at the ends.
    if split <= 0:
        split = 1
    if split >= len(points) - 1:
        split = len(points) - 2

    center_tan = _normalize(points[split - 1] - points[split + 1])
    left = _fit_cubic(points[:split + 1], left_tan, center_tan, error)
    right = _fit_cubic(points[split:], -center_tan, right_tan, error)
    return left + right


def fit_curve(points, max_error):
    """Fit a chain of cubic Beziers to an ordered point list.

    ``points`` is an (N, 2) array.  Returns a list of cubics, each a list of
    four 2D control points.  Consecutive duplicate points are dropped first.
    """
    pts = np.asarray(points, dtype=np.float64)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 2:
        return []
    left_tan = _normalize(pts[1] - pts[0])
    right_tan = _normalize(pts[-2] - pts[-1])
    return _fit_cubic(pts, left_tan, right_tan, max_error)


# ===========================================================================
# SVG path emission
# ===========================================================================
def _fmt(v):
    """Format a coordinate compactly."""
    return ("%.4f" % v).rstrip("0").rstrip(".")


def beziers_to_d(beziers, px_mm):
    """Render one closed ring of cubics to an SVG path-data fragment."""
    if not beziers:
        return ""
    s = px_mm
    start = beziers[0][0]
    parts = ["M %s,%s" % (_fmt(start[0] * s), _fmt(start[1] * s))]
    for bez in beziers:
        parts.append("C %s,%s %s,%s %s,%s" % (
            _fmt(bez[1][0] * s), _fmt(bez[1][1] * s),
            _fmt(bez[2][0] * s), _fmt(bez[2][1] * s),
            _fmt(bez[3][0] * s), _fmt(bez[3][1] * s)))
    parts.append("Z")
    return " ".join(parts)


def beziers_to_open_d(beziers, px_mm):
    """Render an OPEN cubic chain (no Z) -- for stroked centrelines."""
    if not beziers:
        return ""
    s = px_mm
    start = beziers[0][0]
    parts = ["M %s,%s" % (_fmt(start[0] * s), _fmt(start[1] * s))]
    for bez in beziers:
        parts.append("C %s,%s %s,%s %s,%s" % (
            _fmt(bez[1][0] * s), _fmt(bez[1][1] * s),
            _fmt(bez[2][0] * s), _fmt(bez[2][1] * s),
            _fmt(bez[3][0] * s), _fmt(bez[3][1] * s)))
    return " ".join(parts)


REG_MARK_MM = 1.0  # size of the corner registration marks (mm)


def _frame_rect(width_mm, height_mm):
    """Identical corner registration marks on every output SVG.

    Printer apps that position by the VISIBLE content bbox (ignoring the page
    and invisible elements) would otherwise misalign the three files, whose
    real extents differ.  Four small filled squares at the exact canvas corners
    force an identical content bbox in all three AND serve as physical
    registration marks to line the printed layers up.  Set REG_MARK_MM = 0 to
    disable.
    """
    s = REG_MARK_MM
    if s <= 0:
        return ""
    corners = [(0.0, 0.0), (width_mm - s, 0.0),
               (0.0, height_mm - s), (width_mm - s, height_mm - s)]
    return "\n".join(
        '  <rect x="%s" y="%s" width="%s" height="%s" fill="#000000" '
        'stroke="none"/>' % (_fmt(x), _fmt(y), _fmt(s), _fmt(s))
        for (x, y) in corners)


def write_multicolor_svg(colored_paths, width_mm, height_mm):
    """Build an SVG where each entry is (path-data, fill-colour).

    Used for the fragments SVG: one filled vector pane per entry.
    """
    paths = "\n".join(
        '  <path d="%s" fill="%s" fill-rule="evenodd" stroke="none"/>'
        % (d, color) for (d, color) in colored_paths if d)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="%smm" height="%smm" viewBox="0 0 %s %s">\n'
        '%s\n%s\n</svg>\n'
        % (_fmt(width_mm), _fmt(height_mm), _fmt(width_mm), _fmt(height_mm),
           _frame_rect(width_mm, height_mm), paths))


def write_stroke_svg(stroke_items, filled_d, width_mm, height_mm,
                     color="#000000"):
    """SVG of stroked open paths (each (d, stroke-width-mm)) + filled extras.

    Used for the leading in 'stroke' style: centrelines as strokes; blobs/
    dead-ends (no centreline) kept as filled paths so nothing is lost.
    """
    parts = []
    for d, sw in stroke_items:
        if d:
            parts.append(
                '  <path d="%s" fill="none" stroke="%s" stroke-width="%s" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
                % (d, color, _fmt(sw)))
    for d in filled_d:
        if d:
            parts.append('  <path d="%s" fill="%s" fill-rule="evenodd" '
                         'stroke="none"/>' % (d, color))
    body = "\n".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="%smm" height="%smm" viewBox="0 0 %s %s">\n'
        '%s\n%s\n</svg>\n'
        % (_fmt(width_mm), _fmt(height_mm), _fmt(width_mm), _fmt(height_mm),
           _frame_rect(width_mm, height_mm), body))


def write_svg(paths_d, width_mm, height_mm, fill, stroke, stroke_width,
              fill_rule="evenodd"):
    """Build a standalone SVG document string from path-data fragments."""
    style = 'fill="%s" fill-rule="%s"' % (fill, fill_rule)
    if stroke != "none":
        style += ' stroke="%s" stroke-width="%s"' % (stroke, _fmt(stroke_width))
    else:
        style += ' stroke="none"'
    paths = "\n".join(
        '  <path d="%s" %s/>' % (d, style) for d in paths_d if d)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="%smm" height="%smm" viewBox="0 0 %s %s">\n'
        '%s\n%s\n</svg>\n'
        % (_fmt(width_mm), _fmt(height_mm), _fmt(width_mm), _fmt(height_mm),
           _frame_rect(width_mm, height_mm), paths))


# ===========================================================================
# SVG 1 - silhouette
# ===========================================================================
def keep_largest_component(opaque, connectivity):
    """Reduce the silhouette to its largest connected region.

    If the opaque pixels form several disconnected regions, only the largest
    is kept and the rest are dropped (returned as ``dropped`` sizes so the
    caller can warn).  Aborts only if there are no opaque pixels at all.
    """
    conn = 2 if connectivity == 8 else 1
    labels, n = label(opaque, connectivity=conn, return_num=True)
    if n == 0:
        raise AbortError("no opaque pixels found; nothing to outline")
    if n == 1:
        return opaque, []
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    largest = int(np.argmax(sizes))
    dropped = sorted((int(s) for i, s in enumerate(sizes)
                      if i not in (0, largest) and s > 0), reverse=True)
    return labels == largest, dropped


def silhouette_contours(opaque):
    """Return (outer_ring, [hole_rings]) as (N, 2) float arrays in (x, y)."""
    mask = (opaque.astype(np.uint8)) * 255
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise AbortError("could not trace the silhouette boundary")
    hierarchy = hierarchy[0]
    outer = None
    holes = []
    best_area = -1.0
    for i, cnt in enumerate(contours):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if hierarchy[i][3] == -1:  # top-level -> outer boundary
            area = cv2.contourArea(cnt)
            if area > best_area:  # keep the largest outer ring
                best_area = area
                outer = pts
        else:
            holes.append(pts)
    if outer is None:
        outer = max((c.reshape(-1, 2).astype(np.float64) for c in contours),
                    key=len)
    return outer, holes


def _corner_indices(pts, angle_thresh_deg=50.0):
    """Indices of vertices where the polyline turns sharply (a corner)."""
    n = len(pts)
    corners = []
    thresh = np.radians(angle_thresh_deg)
    for i in range(n):
        v1 = pts[i] - pts[(i - 1) % n]
        v2 = pts[(i + 1) % n] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        if np.arccos(cos_a) > thresh:  # turning angle
            corners.append(i)
    return corners


def _open_corner_indices(pts, angle_thresh_deg=40.0):
    """Sharp-turn vertices of an OPEN polyline (interior only, no wrap)."""
    thresh = np.radians(angle_thresh_deg)
    corners = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        if np.arccos(cos_a) > thresh:
            corners.append(i)
    return corners


def fit_open_arc(pts, fit_tol):
    """Corner-aware Bezier fit of an OPEN polyline.

    The arc is split at sharp corners and each straight/curved span is fitted
    independently, so straight seam runs stay straight and junction corners
    stay sharp -- a single smooth fit across a corner bulges or hooks (the
    spire's vertical leading was bent into a belly around an adjacent pane).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return fit_curve(pts, fit_tol)
    splits = [0] + _open_corner_indices(pts) + [len(pts) - 1]
    beziers = []
    for k in range(len(splits) - 1):
        seg = pts[splits[k]:splits[k + 1] + 1]
        if len(seg) >= 2:
            beziers += fit_curve(seg, fit_tol)
    return beziers


def _polyline_to_beziers(pts):
    """Convert a polyline into straight (degenerate) cubic segments."""
    bez = []
    for i in range(len(pts) - 1):
        p0 = np.asarray(pts[i], dtype=np.float64)
        p3 = np.asarray(pts[i + 1], dtype=np.float64)
        bez.append([p0, p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0, p3])
    return bez


def _ring_self_intersects(beziers):
    """True if the closed curve traced by ``beziers`` is not a simple ring."""
    if not beziers:
        return False
    sampled = _sample_beziers(beziers, step=0.4)
    if len(sampled) < 4:
        return False
    ring = Polygon(sampled)
    return not ring.is_valid


def _ring_to_beziers(ring, simplify_tol, fit_tol):
    """Close, simplify and Bezier-fit a ring, preserving sharp corners.

    Corners (high turning angle) split the ring so straight edges stay
    straight and right angles stay sharp instead of being rounded by a
    single smooth closed fit.  If the smooth fit self-intersects (which makes
    nonzero-winding printers misread the shape -- spikes / "holed rectangle"),
    it falls back to the simple simplified polygon so the ring stays valid.
    """
    # Simplify with topology preservation (keeps thin features like 1px lines).
    # Do NOT gate on poly.is_valid: raw pixel contours are frequently invalid
    # (self-touching), and skipping simplification then emits the full staircase
    # (jittery edges, thousands of cubics).  Polygon.simplify handles invalid
    # input fine.
    coords = np.asarray(ring, dtype=np.float64)
    if simplify_tol > 0 and len(coords) > 4:
        poly = Polygon(coords)
        if not poly.is_valid:
            # Clean self-touching/pinched rings so the result (and the polyline
            # fallback below) is a simple ring -- otherwise the fit traces a
            # figure-8 that printers read as a "holed rectangle".
            poly = poly.buffer(0)
            if poly.geom_type == "MultiPolygon" and not poly.is_empty:
                poly = max(poly.geoms, key=lambda g: g.area)
        if (poly.geom_type == "Polygon" and not poly.is_empty
                and poly.exterior is not None):
            simp = poly.simplify(simplify_tol, preserve_topology=True)
            if simp.geom_type == "Polygon" and simp.exterior is not None:
                c = np.asarray(simp.exterior.coords)
                if len(c) >= 4:
                    coords = c
    # Drop the duplicate closing vertex; work on the unique ring.
    if len(coords) > 1 and np.linalg.norm(coords[0] - coords[-1]) < 1e-9:
        pts = coords[:-1].astype(np.float64)
    else:
        pts = coords.astype(np.float64)
    n = len(pts)
    closed = np.vstack([pts, pts[0]])
    if n < 3:
        return fit_curve(closed, fit_tol)

    corners = _corner_indices(pts)
    if not corners:
        beziers = fit_curve(closed, fit_tol)  # smooth closed curve
    else:
        # Fit each arc between consecutive corners independently.
        beziers = []
        for k in range(len(corners)):
            i0 = corners[k]
            i1 = corners[(k + 1) % len(corners)]
            if i1 > i0:
                seg = pts[i0:i1 + 1]
            else:  # wrap around the closing seam
                seg = np.vstack([pts[i0:], pts[:i1 + 1]])
            if len(seg) >= 2:
                beziers += fit_curve(seg, fit_tol)

    if _ring_self_intersects(beziers):
        beziers = _polyline_to_beziers(closed)  # simple, printer-safe fallback
    return beziers


def contours_to_paths(outer, holes, px_mm, simplify_tol, fit_tol):
    """Bezier-fit the outer ring + holes into one even-odd path-data string."""
    rings = [outer] + holes
    fragments = []
    for ring in rings:
        if len(ring) < 3:
            continue
        beziers = _ring_to_beziers(ring, simplify_tol, fit_tol)
        frag = beziers_to_d(beziers, px_mm)
        if frag:
            fragments.append(frag)
    return " ".join(fragments)


# ===========================================================================
# SVG 2 - leading lines
# ===========================================================================
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_branches(skel):
    """Split a 1-px skeleton into ordered branch polylines (lists of (r, c)).

    Branches run between skeleton nodes (degree != 2: endpoints and
    junctions).  Pure loops (cycles with no node) are emitted as closed
    polylines.  Each skeleton edge appears in exactly one branch.
    """
    coords = set(map(tuple, np.argwhere(skel)))
    if not coords:
        return []

    def nbrs(p):
        r, c = p
        out = []
        for dr, dc in _NEIGHBORS:
            q = (r + dr, c + dc)
            if q in coords:
                out.append(q)
        return out

    degree = {p: len(nbrs(p)) for p in coords}
    nodes = {p for p in coords if degree[p] != 2}
    visited_edges = set()
    branches = []

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    # Walk every branch starting from each node neighbour.
    for node in nodes:
        for first in nbrs(node):
            if edge_key(node, first) in visited_edges:
                continue
            path = [node, first]
            visited_edges.add(edge_key(node, first))
            prev, cur = node, first
            while cur not in nodes:
                nxt = [q for q in nbrs(cur) if q != prev]
                if not nxt:
                    break
                step = nxt[0]
                visited_edges.add(edge_key(cur, step))
                path.append(step)
                prev, cur = cur, step
            branches.append(path)

    # Remaining unvisited degree-2 pixels form pure loops.
    for start in coords:
        if degree[start] != 2:
            continue
        nb = nbrs(start)
        if any(edge_key(start, n) not in visited_edges for n in nb):
            first = next(n for n in nb if edge_key(start, n) not in visited_edges)
            path = [start, first]
            visited_edges.add(edge_key(start, first))
            prev, cur = start, first
            while cur != start:
                nxt = [q for q in nbrs(cur) if q != prev]
                if not nxt:
                    break
                step = nxt[0]
                if edge_key(cur, step) in visited_edges and step != start:
                    break
                visited_edges.add(edge_key(cur, step))
                path.append(step)
                prev, cur = cur, step
            branches.append(path)

    # Isolated single pixels (degree 0) survive as length-1 branches.
    for p in coords:
        if degree[p] == 0:
            branches.append([p])

    return branches


def merge_branches_through_junctions(branches, max_turn_cos=-0.3):
    """Merge skeleton branches into continuous strokes that pass through
    junctions.

    At each junction the incident branch-ends are paired by best collinearity
    (straightest through-path), so two crossing lines become two continuous
    strokes instead of four stubs meeting at the centre.  This cleans up
    intersections (no flat-cap noise at the crossing) and greatly reduces the
    number of emitted leading paths.  ``max_turn_cos`` is the cosine threshold:
    two ends merge only if their outgoing directions are nearly opposite
    (dot <= this), i.e. the line goes roughly straight through.
    """
    if not branches:
        return []

    def key(p):
        return (int(p[0]), int(p[1]))

    def end_dir(br, end):
        pts = np.asarray(br, dtype=np.float64)
        if len(pts) < 2:
            return np.array([0.0, 0.0])
        if end == 0:
            v = pts[min(len(pts) - 1, 3)] - pts[0]
        else:
            v = pts[max(0, len(pts) - 4)] - pts[-1]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    node_ends = {}
    for bi, br in enumerate(branches):
        for end in (0, 1):
            node_ends.setdefault(key(br[0] if end == 0 else br[-1]), []).append(
                (bi, end))

    pair = {}
    for ends in node_ends.values():
        if len(ends) < 2:
            continue
        dirs = {e: end_dir(branches[e[0]], e[1]) for e in ends}
        cand = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                cand.append((float(np.dot(dirs[ends[i]], dirs[ends[j]])),
                             ends[i], ends[j]))
        cand.sort(key=lambda t: t[0])
        used = set()
        for d, ei, ej in cand:
            if d > max_turn_cos:
                break
            if ei in used or ej in used:
                continue
            used.add(ei)
            used.add(ej)
            pair[ei] = ej
            pair[ej] = ei

    visited = set()

    def traverse(start):
        bi, end = start
        pts = []
        while bi not in visited:
            visited.add(bi)
            seg = list(branches[bi])
            if end == 1:
                seg = seg[::-1]
            if pts and key(pts[-1]) == key(seg[0]):
                seg = seg[1:]
            pts.extend(seg)
            nxt = pair.get((bi, 1 - end))
            if nxt is None:
                break
            bi, end = nxt
        return pts

    strokes = []
    for bi in range(len(branches)):
        for end in (0, 1):
            if bi not in visited and (bi, end) not in pair:
                strokes.append(traverse((bi, end)))
    for bi in range(len(branches)):  # leftover pure loops
        if bi not in visited:
            strokes.append(traverse((bi, 0)))
    return [s for s in strokes if len(s) >= 1]


def branch_half_width_px(branch_rc, edt, min_line_width):
    """One uniform half-width per stroke = median local thickness / 2.

    ``edt`` is the Euclidean distance transform of the (padded) black mask.
    A 1-px-wide line has EDT ~= 1 along its centre -> width = 2*EDT-1 = 1 px.
    """
    vals = [edt[r, c] for (r, c) in branch_rc]
    edt_med = float(np.median(vals)) if vals else 0.5
    width = max(2.0 * edt_med - 1.0, float(min_line_width))
    return width / 2.0


def _sample_beziers(beziers, step=1.0):
    """Sample a Bezier chain into a dense polyline (approx. ``step`` px apart)."""
    pts = []
    for k, b in enumerate(beziers):
        cp = [np.asarray(x, dtype=np.float64) for x in b]
        length = sum(np.linalg.norm(cp[i + 1] - cp[i]) for i in range(3))
        m = max(2, int(np.ceil(length / step)))
        for i in range(m + 1):
            if k > 0 and i == 0:
                continue
            pts.append(_q(cp, i / m))
    return pts


def _smooth_centerline(xy, tol):
    """Smooth a skeleton centreline by fitting a Bezier and resampling it.

    Removes the 1-px staircase while keeping genuine curves smooth (a polyline
    Douglas-Peucker simplify would instead turn curves into angular segments).
    Straight lines fit to a single straight cubic and are returned unchanged.
    """
    pts = np.asarray(xy, dtype=np.float64)
    if len(pts) < 3 or tol <= 0:
        return [tuple(p) for p in pts]
    # Douglas-Peucker first: snaps near-straight runs (and the small medial
    # wobble where other strokes join at junctions) to straight lines, so a
    # straight border stroke stays straight instead of gently waving.
    simp = np.asarray(LineString(pts).simplify(max(tol, 2.0)).coords)
    if len(simp) >= 2:
        pts = simp
    beziers = fit_curve(pts, tol)
    if not beziers:
        return [tuple(p) for p in pts]
    return [tuple(p) for p in _sample_beziers(beziers)]


def _extend_polyline(coords_xy, ext):
    """Extend a polyline by ``ext`` at both ends along its terminal segments.

    Ensures flat-capped buffers reach the true pixel extent and that branches
    overlap at shared junction nodes so the union has no seams.
    """
    pts = list(coords_xy)
    if len(pts) < 2 or ext <= 0:
        return pts
    d0 = _normalize(np.asarray(pts[0]) - np.asarray(pts[1]))
    d1 = _normalize(np.asarray(pts[-1]) - np.asarray(pts[-2]))
    pts[0] = (np.asarray(pts[0]) + d0 * ext).tolist()
    pts[-1] = (np.asarray(pts[-1]) + d1 * ext).tolist()
    return pts


def build_ribbons(branches, edt, min_line_width, cap_style, smooth_tol):
    """Buffer each centreline to a polygon ribbon using its own width.

    The centreline polyline is simplified (Douglas-Peucker, ``smooth_tol``)
    first to remove the 1-px skeleton staircase; straight lines are
    unaffected, so thin features keep their exact width.
    """
    cap_map = {"round": 1, "flat": 2, "square": 3}
    cap = cap_map.get(cap_style, 2)
    ribbons = []
    for branch_rc in branches:
        half = branch_half_width_px(branch_rc, edt, min_line_width)
        # Convert (row, col) -> (x, y) = (col, row).
        xy = [(c, r) for (r, c) in branch_rc]
        if len(xy) == 1:
            x, y = xy[0]
            ribbons.append(box(x - half, y - half, x + half, y + half))
            continue
        xy = _smooth_centerline(xy, smooth_tol)
        xy = _extend_polyline(xy, half)
        ribbon = LineString(xy).buffer(half, cap_style=cap, join_style=1)
        if not ribbon.is_empty:
            ribbons.append(ribbon)
    return ribbons


def union_components(ribbons):
    """Boolean-union ribbons and return a list of Polygons."""
    if not ribbons:
        return []
    merged = unary_union([r for r in ribbons if not r.is_empty])
    if merged.is_empty:
        return []
    if merged.geom_type == "Polygon":
        return [merged]
    return [g for g in merged.geoms if g.geom_type == "Polygon"]


def prune_spur_branches(branches, skel, max_spur_len):
    """Drop short skeleton hairs hanging off junctions (skeletonisation noise).

    A branch is a spur if it is short and runs from a junction (degree>=3) to
    a free endpoint (degree 1).  Removing these tames the jitter where several
    strokes intersect.
    """
    coords = set(map(tuple, np.argwhere(skel)))
    degree = {}
    for (r, c) in coords:
        d = 0
        for dr, dc in _NEIGHBORS:
            if (r + dr, c + dc) in coords:
                d += 1
        degree[(r, c)] = d
    kept = []
    for br in branches:
        da = degree.get(tuple(br[0]), 0)
        db = degree.get(tuple(br[-1]), 0)
        is_hair = (len(br) < max_spur_len
                   and min(da, db) <= 1 and max(da, db) >= 3)
        if not is_hair:
            kept.append(br)
    return kept


def contour_polygons(comp_mask):
    """Trace a solid black blob (e.g. an eye dot) as filled polygon(s)."""
    m = (comp_mask.astype(np.uint8)) * 255
    contours, hierarchy = cv2.findContours(
        m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    hierarchy = hierarchy[0]
    outers = {}
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == -1 and len(cnt) >= 3:
            outers[i] = [cnt.reshape(-1, 2).astype(np.float64), []]
    for i, cnt in enumerate(contours):
        parent = hierarchy[i][3]
        if parent != -1 and parent in outers and len(cnt) >= 3:
            outers[parent][1].append(cnt.reshape(-1, 2).astype(np.float64))
    polys = []
    for ext, holes in outers.values():
        poly = Polygon(ext, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)
    return polys


def _flatten_polys(geoms):
    """Return a flat list of non-empty shapely Polygons (recurses collections)."""
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Polygon":
            out.append(g)
        elif g.geom_type in ("MultiPolygon", "GeometryCollection"):
            out.extend(_flatten_polys(list(g.geoms)))
    return out


def _mask_to_polygon(mask):
    """Build a cleaned shapely (Multi)Polygon from a binary mask via contours."""
    m = (mask.astype(np.uint8)) * 255
    contours, hierarchy = cv2.findContours(
        m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hierarchy = hierarchy[0]
    outers = {}
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == -1 and len(cnt) >= 3:
            outers[i] = [cnt.reshape(-1, 2).astype(np.float64), []]
    for i, cnt in enumerate(contours):
        par = hierarchy[i][3]
        if par != -1 and par in outers and len(cnt) >= 3:
            outers[par][1].append(cnt.reshape(-1, 2).astype(np.float64))
    polys = []
    for ext, holes in outers.values():
        p = Polygon(ext, holes)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    merged = unary_union(polys)
    return merged if not merged.is_empty else None


def build_silhouette_polygon(outer, holes, simplify_tol):
    """Clean, simplified shapely Polygon of the silhouette (for clipping panes)."""
    poly = Polygon(outer, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    polys = _flatten_polys([poly])
    if not polys:
        return None
    poly = max(polys, key=lambda g: g.area)
    if simplify_tol > 0:
        poly = poly.simplify(simplify_tol, preserve_topology=True)
    return poly if poly.geom_type == "Polygon" and not poly.is_empty else None


def vectorize_leading(black, min_line_width, cap_style, smooth_tol):
    """Vectorise the black leading into a list of individual ribbon polygons.

    Round 'blobs' (eye dots) are traced as filled contours; elongated 'strokes'
    use the skeleton centreline + per-stroke median width, merged through
    junctions for clean intersections.  Ribbons are individual simple polygons
    (printer-safe; no holed union).
    """
    cap = {"flat": 2, "round": 1, "square": 3}.get(cap_style, 2)
    ribbons = []
    lbl, n = label(black, connectivity=2, return_num=True)
    for ci in range(1, n + 1):
        comp = lbl == ci
        if comp.sum() < 3:
            continue
        edt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
        max_thick = float(edt.max())
        skel = skeletonize(comp)
        skel_len = int(skel.sum())
        if skel_len < 3 or skel_len <= 2.5 * (2.0 * max_thick):
            ribbons += contour_polygons(comp)  # blob: keep its real shape
            continue
        branches = trace_branches(skel)
        branches = prune_spur_branches(
            branches, skel, max(3.0, 1.5 * (2.0 * max_thick)))
        ribbons += build_ribbons(branches, edt, min_line_width, cap_style,
                                 smooth_tol)
    return _flatten_polys(ribbons)


def polygon_to_path(poly, px_mm, simplify_tol, fit_tol):
    """Bezier-fit a polygon (exterior + holes) into one even-odd path.

    Accepts a MultiPolygon too (a buffered/repaired blob can split), emitting
    every part's rings into one even-odd path.
    """
    fragments = []
    rings = []
    for part in _flatten_polys([poly]):
        rings.append(np.asarray(part.exterior.coords))
        rings += [np.asarray(r.coords) for r in part.interiors]
    for ring in rings:
        if len(ring) < 4:
            continue
        beziers = _ring_to_beziers(ring, simplify_tol, fit_tol)
        frag = beziers_to_d(beziers, px_mm)
        if frag:
            fragments.append(frag)
    return " ".join(fragments)


# ===========================================================================
# PNG 3 - glass fragments (silhouette split along the leading skeleton)
# ===========================================================================
def split_into_fragments(opaque, black):
    """Partition the silhouette into glass panes seeded by the glass regions.

    Fragments are the connected glass (opaque non-black) regions; every opaque
    pixel -- including the thick black leading -- is then assigned to its
    nearest glass region.  The pane boundaries fall along the middle of the
    leading (its medial axis), the fragments tile the *entire* silhouette with
    no empty area, and because every fragment is seeded by glass none can come
    out black (black is printed later as the leading).
    """
    glass = opaque & ~black
    labels = label(glass, connectivity=1)
    zero = labels == 0  # black leading + background
    if zero.any() and (~zero).any():
        idx = ndimage.distance_transform_edt(
            zero, return_distances=False, return_indices=True)
        labels = labels[tuple(idx)]
    labels[~opaque] = 0
    return labels


def _quantize_reps(rep, ids, areas, k):
    """Cluster per-fragment colours into k groups, weighted by pane area.

    Area weighting means large panes drive the colour slots and tiny slivers
    do not waste one of the few available colours.
    """
    cols = np.array([rep[i] for i in ids], dtype=np.float64)
    uniq = np.unique(np.round(cols), axis=0)
    k = max(1, min(int(k), len(uniq)))
    weights = np.maximum(
        1, np.round(areas / max(areas.sum(), 1) * 4000).astype(int))
    samples = np.repeat(cols, weights, axis=0)
    centroids, _ = kmeans2(samples, k, minit="++", seed=0, missing="warn")
    dist = ((cols[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    lab = np.argmin(dist, axis=1)
    return {fid: centroids[lab[j]] for j, fid in enumerate(ids)}


def _merge_adjacent_similar(labels, glass, rgb, tol):
    """Merge panes that touch DIRECTLY in the glass (no black between them) and
    have near-identical mean colour.

    These are spurious 'colour cross-cuts': a single band split into two slightly
    different quantised shades by anti-aliasing near its black separators -> a
    seam with NO black that prints as a small-rectangle cross-cut.  Panes with a
    real black divider between them are NOT glass-adjacent (black breaks the
    adjacency), so genuine leading lines -- even between same-colour panes -- are
    never merged.  Distinct abutting colours (a deliberate lead-less colour
    boundary, e.g. cheeks) exceed ``tol`` and are kept.
    """
    maxl = int(labels.max())
    if maxl < 2:
        return labels
    # Mean glass colour per label, vectorised (labelled ndimage.mean over all
    # labels at once -- not a full-image mask per pane).
    lab_g = np.where(glass, labels, 0)
    idx = np.arange(maxl + 1)
    mean = np.zeros((maxl + 1, 3), dtype=np.float64)
    for ch in range(3):
        mean[:, ch] = np.nan_to_num(
            ndimage.mean(rgb[:, :, ch].astype(np.float64), lab_g, idx))

    parent = np.arange(maxl + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h, w = labels.shape
    for dr, dc in [(0, 1), (1, 0)]:
        a = labels[0:h - dr, 0:w - dc]
        b = labels[dr:h, dc:w]
        both = glass[0:h - dr, 0:w - dc] & glass[dr:h, dc:w]  # no black between
        m = both & (a != b) & (a > 0) & (b > 0)
        for x, y in set(zip(a[m].tolist(), b[m].tolist())):
            if np.linalg.norm(mean[x] - mean[y]) < tol:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx
    root = np.array([find(i) for i in range(maxl + 1)])
    return root[labels]


def _merge_small_panes(labels, min_area):
    """Merge panes smaller than ``min_area`` into their longest-shared-border
    neighbour, iterating so chains of small panes collapse into one survivor.

    Keeps regions connected (no dangling seams) -- unlike drop-to-0 + nearest
    flood, which splits a removed pane across whichever panes happen to be
    closest and can leave broken leading.
    """
    labels = labels.copy()
    hh, ww = labels.shape
    for _ in range(25):
        sizes = np.bincount(labels.ravel())
        smalls = [i for i in range(1, len(sizes))
                  if 0 < sizes[i] < min_area]
        if not smalls:
            break
        # Work in each small pane's BOUNDING BOX (not the full image) -- a tiny
        # pane has a tiny bbox, so this is orders of magnitude faster at full
        # resolution.
        objs = ndimage.find_objects(labels)
        changed = False
        for i in sorted(smalls, key=lambda j: -sizes[j]):
            sl = objs[i - 1] if i - 1 < len(objs) else None
            if sl is None:
                continue
            sl = (slice(max(0, sl[0].start - 1), min(hh, sl[0].stop + 1)),
                  slice(max(0, sl[1].start - 1), min(ww, sl[1].stop + 1)))
            sub = labels[sl]                     # view into labels
            m = sub == i
            ring = ndimage.binary_dilation(m, np.ones((3, 3), bool)) & ~m
            nbr = sub[ring]
            nbr = nbr[(nbr > 0) & (nbr != i)]
            if nbr.size == 0:
                continue
            vals, counts = np.unique(nbr, return_counts=True)
            sub[m] = int(vals[np.argmax(counts)])
            changed = True
        if not changed:
            break
    return labels


def split_into_fragments_by_color(opaque, black, rgb, num_colors, min_frag,
                                  color_merge_tol=COLOR_MERGE_TOL):
    """Partition the silhouette by colour *and* leading.

    Glass colours are quantised to a palette, a majority filter removes
    anti-alias boundary noise, and connected same-colour glass regions become
    panes -- so colour patches with no surrounding lead line (e.g. cheeks)
    become their own panes, while the black leading still separates panes
    (black breaks glass connectivity).  Tiny specks below ``min_frag`` and the
    black leading are absorbed into the nearest pane so the silhouette stays
    fully tiled.
    """
    glass = opaque & ~black
    samples = rgb[glass].astype(np.float64)
    if len(samples) == 0:
        return split_into_fragments(opaque, black)
    k = max(1, min(int(num_colors), len(np.unique(np.round(samples), axis=0))))
    centroids, _ = kmeans2(samples, k, minit="++", seed=0, missing="warn")
    centroids = centroids[np.isfinite(centroids).all(axis=1)]
    if len(centroids) == 0:
        return split_into_fragments(opaque, black)

    # Per-pixel nearest palette colour.
    h, w = opaque.shape
    cmap = np.full((h, w), -1, dtype=np.int32)
    flat = rgb[glass].astype(np.float64)
    dist = ((flat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    cmap[glass] = np.argmin(dist, axis=1)

    # Majority filter (2 passes) to clean 1-px anti-alias boundary noise.
    kk = len(centroids)
    for _ in range(2):
        counts = np.stack(
            [ndimage.uniform_filter((cmap == c).astype(np.float64), size=3)
             for c in range(kk)])
        new = np.argmax(counts, axis=0)
        cmap = np.where(glass, new, -1)

    # Connected same-colour glass regions -> panes.
    labels = np.zeros((h, w), dtype=np.int32)
    nxt = 0
    for c in range(kk):
        m = glass & (cmap == c)
        if not m.any():
            continue
        lbl, n = label(m, connectivity=1, return_num=True)
        labels[m] = lbl[m] + nxt
        nxt += n

    # Heal spurious colour cross-cuts: panes touching directly in the glass (no
    # black between) with near-identical colour are one band split by anti-alias
    # shading -> merge them so they don't print as small-rectangle cross-cuts.
    # Real black-lined dividers are not glass-adjacent, so they are untouched.
    if nxt > 0:
        if color_merge_tol > 0:
            labels = _merge_adjacent_similar(labels, glass, rgb, color_merge_tol)

    # Merge panes below min_frag into their most-bordering neighbour (NOT a
    # drop-to-0 + distance flood, which fragments a dropped pane across several
    # neighbours and leaves dangling/broken leading at higher thresholds).
    # Iterative so a chain of small panes collapses into one survivor.
    if min_frag > 1 and nxt > 0:
        labels = _merge_small_panes(labels, int(min_frag))
    zero = labels == 0
    if zero.any() and (~zero).any():
        idx = ndimage.distance_transform_edt(
            zero, return_distances=False, return_indices=True)
        labels = labels[tuple(idx)]
    labels[~opaque] = 0
    return labels


def fragment_colors(rgba, labels, glass, mode, num_colors):
    """Solid-fill each fragment, using at most ``num_colors`` glass colours.

    ``glass`` marks opaque non-black pixels (the real pane colour); the black
    leading area inside a fragment is filled with the pane colour so the
    printed glass has no black gaps (black is printed later as the leading).
    Each pane starts as the median of its glass pixels; if that yields more
    than ``num_colors`` distinct colours, or in ``quantized`` mode, the pane
    colours are reduced to ``num_colors`` so the result fits the printer's
    colour slots.  Returns (rgb_image, n_colours_used).
    """
    rgb = rgba[:, :, :3]
    out = np.zeros_like(rgb)
    ids = [i for i in np.unique(labels) if i != 0]
    if not ids:
        return out, 0
    rep = {}
    areas = np.zeros(len(ids), dtype=np.float64)
    for j, fid in enumerate(ids):
        region = labels == fid
        areas[j] = region.sum()
        src = region & glass
        if not src.any():
            src = region
        rep[fid] = np.median(rgb[src], axis=0)

    distinct = len(np.unique(np.round(np.array([rep[i] for i in ids])), axis=0))
    if mode == "quantized" or distinct > num_colors:
        rep = _quantize_reps(rep, ids, areas, num_colors)

    for fid in ids:
        out[labels == fid] = np.clip(rep[fid], 0, 255).astype(np.uint8)
    n_used = len(np.unique(out[labels > 0], axis=0))
    return out, n_used


def trace_region_arcs(labels):
    """Trace the boundary arcs of a labelled partition (shared-edge graph).

    Walks the pixel 'cracks' between differing labels and groups them into arcs
    that run between junction corners (where 3+ regions meet) and separate one
    pair of regions.  Each arc is traced ONCE, so when faces are rebuilt from
    these arcs adjacent panes share the identical edge -> exact partition with
    no gaps or overlaps.  Returns a list of corner polylines in (x, y) px.
    """
    from collections import defaultdict
    Lp = np.pad(labels.astype(np.int64), 1, constant_values=0)
    adj = defaultdict(list)

    def add(a, b):
        adj[a].append(b)
        adj[b].append(a)

    pair_at = {}  # frozenset edge-key -> region pair (for junction detection)

    def edge(a, b, pa, pb):
        add(a, b)
        pair_at[(a, b) if a <= b else (b, a)] = (min(pa, pb), max(pa, pb))

    diff = Lp[:, :-1] != Lp[:, 1:]
    for r, c in zip(*np.nonzero(diff)):
        edge((c + 1, r), (c + 1, r + 1), int(Lp[r, c]), int(Lp[r, c + 1]))
    diff = Lp[:-1, :] != Lp[1:, :]
    for r, c in zip(*np.nonzero(diff)):
        edge((c, r + 1), (c + 1, r + 1), int(Lp[r, c]), int(Lp[r + 1, c]))

    def ekey(a, b):
        return (a, b) if a <= b else (b, a)

    def is_junction(cn):
        e = adj[cn]
        if len(e) != 2:
            return True
        return pair_at[ekey(cn, e[0])] != pair_at[ekey(cn, e[1])]

    visited = set()
    arcs = []

    def walk(start, first):
        pair = pair_at[ekey(start, first)]
        path = [start, first]
        visited.add(ekey(start, first))
        prev, cur = start, first
        while not is_junction(cur):
            nxt = [n for n in adj[cur] if n != prev and ekey(cur, n) not in visited]
            if not nxt:
                break
            n = nxt[0]
            visited.add(ekey(cur, n))
            path.append(n)
            prev, cur = cur, n
        return [(x - 1, y - 1) for (x, y) in path], pair

    for j in [cn for cn in adj if is_junction(cn)]:
        for nbr in adj[j]:
            if ekey(j, nbr) not in visited:
                arcs.append(walk(j, nbr))
    for cn in list(adj):  # pure loops with no junction (e.g. an island ring)
        for nbr in adj[cn]:
            if ekey(cn, nbr) not in visited:
                arcs.append(walk(cn, nbr))
    return arcs


def _arc_is_simple(pts):
    """True if the open/closed arc traced by ``pts`` does not self-intersect.

    A closed arc (first == last point, e.g. a pane fully enclosed by one arc)
    is simple as long as it only touches itself at the shared endpoint.
    """
    if len(pts) < 3:
        return True
    line = LineString(pts)
    if line.is_simple:
        return True
    # Closed ring: is_simple is False merely because the ends coincide; accept
    # it when the corresponding ring is a valid (non-self-crossing) polygon.
    a, b = np.asarray(pts[0]), np.asarray(pts[-1])
    if np.linalg.norm(a - b) < 1e-6 and len(pts) >= 4:
        return Polygon(pts).is_valid
    return False


def _smooth_arc(path, simplify_tol, fit_tol, sample_step=1.0):
    """Simplify + Bezier-smooth a boundary arc, preserving its endpoints.

    ``sample_step`` (px) scales the bezier resampling + thinning -- pass the
    full-res/downscaled ratio so a full-res run keeps the same vertex density as
    the old downscaled pipeline (avoids a vertex blow-up at original resolution).
    """
    pts = np.asarray(path, dtype=np.float64)
    if simplify_tol > 0 and len(pts) > 2:
        pts = np.asarray(LineString(pts).simplify(simplify_tol).coords)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    beziers = fit_open_arc(pts, fit_tol)
    if not beziers:
        return [tuple(p) for p in pts]
    sampled = _sample_beziers(beziers, step=max(1.0, sample_step))
    # The smoothed arc is shared by the fragment faces (polygonize): if the
    # Bezier fit bulged into a self-crossing, the polygonised pane would
    # self-intersect (a "holed rectangle"/bowtie on nonzero-winding printers).
    # Fall back to the simple simplified polyline, which cannot self-cross.
    if not _arc_is_simple(sampled):
        return [tuple(p) for p in pts]
    # Thin out the 1px resampling: a long arc otherwise yields hundreds of
    # near-collinear points, and a large concave pane built from them (e.g. the
    # bottom water band) can have 500+ vertices, which some printer importers
    # mis-fill as a "holed rectangle".  The arc is SHARED, so simplifying it
    # once keeps adjacent faces exactly aligned.  0.25px is far below the
    # 0.4mm print resolution -> lossless in practice.
    if len(sampled) > 4:
        simp = np.asarray(
            LineString(sampled).simplify(0.25 * max(1.0, sample_step)).coords)
        if len(simp) >= 2 and _arc_is_simple(simp):
            sampled = simp
    return [tuple(p) for p in sampled]


def _polygon_polyline_d(poly, px_mm):
    """Emit a polygon (exterior + holes) as a path.

    The ring orientation is normalised (exterior CCW, holes CW) so the hole is
    cut correctly under BOTH even-odd and nonzero winding -- the printer reads
    nonzero, where a hole wound the SAME way as its exterior would fill in (an
    apparent self-intersection / "holed rectangle").
    """
    from shapely.geometry.polygon import orient
    if poly.geom_type == "Polygon" and not poly.is_empty:
        poly = orient(poly, sign=1.0)  # exterior CCW, interiors CW
    s = px_mm
    parts = []
    for ring in [poly.exterior] + list(poly.interiors):
        coords = list(ring.coords)
        if len(coords) < 3:
            continue
        parts.append("M %s,%s" % (_fmt(coords[0][0] * s), _fmt(coords[0][1] * s)))
        for x, y in coords[1:]:
            parts.append("L %s,%s" % (_fmt(x * s), _fmt(y * s)))
        parts.append("Z")
    return " ".join(parts)


def compute_partition_arcs(labels, simplify_tol, fit_tol, min_simplify=2.0,
                           black=None, sample_step=1.0):
    """Trace + smooth the partition boundary arcs (shared by leading & faces).

    Smoothing is done a bit more strongly than a single tolerance so straight
    runs and thin-line medials come out clean (not jittery) -- the SAME smoothed
    arcs feed both the fragment faces and the leading, so they stay registered.
    ``min_simplify`` is the pre-Bezier DP-simplify floor; lower it (smooth mode)
    to keep tight-curve detail (e.g. wave-foam spirals) instead of decimating it.

    Returns ``(arcs, arc_widths, interior_arcs, interior_widths)``: ``arcs`` is
    every arc (faces need them to close at the border); ``interior_arcs`` drops
    border arcs (a pane vs outside, label 0) so the seam leading doesn't double
    the came; the ``*_widths`` are each arc's black thickness in px, measured on
    the RAW pixel-crack path (which sits exactly on the black medial -- accurate
    even for 1px lines, unlike the smoothed arc whose medial drifts off thin
    lines).  Used to tier the leading into thin/bold.
    """
    edt = (ndimage.distance_transform_edt(black) if black is not None
           else None)
    h, w = (black.shape if black is not None else (0, 0))

    def raw_width(path):
        if edt is None:
            return 0.0
        vals = []
        for (x, y) in path:
            r, c = int(round(y)), int(round(x))
            win = edt[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if win.size:
                vals.append(float(win.max()))
        return 2.0 * float(np.median(vals)) if vals else 0.0

    arcs = []
    arc_w = []
    interior = []
    interior_w = []
    for path, pair in trace_region_arcs(labels):
        sm = _smooth_arc(path, max(simplify_tol, min_simplify), fit_tol,
                         sample_step)
        if len(sm) >= 2:
            wpx = raw_width(path)
            arcs.append(sm)
            arc_w.append(wpx)
            if pair[0] != 0:  # 0 = outside the silhouette -> a border arc
                interior.append(sm)
                interior_w.append(wpx)
    return arcs, arc_w, interior, interior_w


def fragments_to_colored_paths(frag_rgb, labels, smoothed_arcs, clip_poly, px_mm):
    """Vectorise the glass panes as an EXACT planar partition.

    Faces are rebuilt with ``shapely.polygonize`` from the shared smoothed
    boundary arcs -- so adjacent panes share the identical edge with no gaps
    and no overlaps.  Each face is then clipped to ``clip_poly`` (the clean
    simplified silhouette) so the panes' OUTER edge is the exact silhouette
    (the raw border arcs overshoot the sharp corners and would otherwise make
    the fragment outline bulge/asymmetric and push the bounding box out).
    Edges are emitted as dense polylines (exact sharing); coloured by region.
    """
    from shapely.ops import polygonize, unary_union
    lines = [LineString(s) for s in smoothed_arcs if len(s) >= 2]
    if not lines:
        return []
    faces = list(polygonize(unary_union(lines)))
    h, w = labels.shape
    # A thin SLIVER face's representative point can land on a labels==0 pixel (a
    # seam/leading pixel); precompute the nearest nonzero-label pixel so the face
    # is coloured by its adjacent pane instead of dropping out.
    zero = labels == 0
    near = (ndimage.distance_transform_edt(zero, return_distances=False,
                                           return_indices=True)
            if zero.any() else None)

    def _color_at(pt):
        c, r = int(round(pt.x)), int(round(pt.y))
        if not (0 <= r < h and 0 <= c < w):
            return None
        if int(labels[r, c]) == 0 and near is not None:
            r, c = int(near[0][r, c]), int(near[1][r, c])   # nearest glass pixel
        fid = int(labels[r, c])
        if fid == 0:
            return None
        col = frag_rgb[r, c]
        return "#%02x%02x%02x" % (int(col[0]), int(col[1]), int(col[2]))

    emitted = []                                 # (geom, hex) glass panes
    for face in faces:
        if face.is_empty or face.area < 0.5:
            continue
        hexc = _color_at(face.representative_point())
        if hexc is None:
            continue
        clipped = face if clip_poly is None else face.intersection(clip_poly)
        if not clipped.is_valid:
            clipped = clipped.buffer(0)          # repair any self-touch from clip
        for pg in _flatten_polys([clipped]):
            if not pg.is_valid:
                pg = pg.buffer(0)
            for g in _flatten_polys([pg]):
                if g.area >= 0.5:
                    emitted.append((g, hexc))

    # Close HOLES: arcs that don't quite close into a face near junctions/borders
    # leave thin silhouette regions no face enclosed (the "hole in the zigzag
    # around the grass").  Fill each leftover piece with the nearest pane's colour
    # so the panes tile the silhouette EXACTLY -- no holes, no overlaps.
    if clip_poly is not None and emitted:
        leftover = clip_poly.difference(unary_union([g for g, _ in emitted]))
        for piece in _flatten_polys([leftover]):
            if piece.is_empty or piece.area < 0.5:
                continue
            if not piece.is_valid:
                piece = piece.buffer(0)
            for g in _flatten_polys([piece]):
                if g.area < 0.5:
                    continue
                hexc = _color_at(g.representative_point())
                if hexc is None:                 # fall back to nearest pane
                    hexc = min(emitted, key=lambda gc: g.distance(gc[0]))[1]
                emitted.append((g, hexc))

    items = []
    for g, hexc in emitted:
        d = _polygon_polyline_d(g, px_mm)
        if d:
            items.append((g.area, d, hexc))
    items.sort(key=lambda t: -t[0])
    return [(d, c) for (_, d, c) in items]


def leading_skeleton_strokes(black, min_line_width, smooth_tol, px_mm, fit_tol,
                             uniform_mm, clip_poly, merge=False, width_scale=1.0):
    """Leading as strokes from the skeleton of ALL the black (the full line art).

    Unlike the arc/seam approach, this traces every black line once -- so faint
    pane dividers and feature outlines (tree, butterfly) that are not partition
    seams are still captured and stay continuous/enclosed.  Returns
    (strokes=[(path-data, stroke-width-mm)], filled_polys) where filled_polys are
    round 'blob' contours (eye dots) that have no centreline.  Branches are NOT
    merged through junctions (kept per user request); they share junction
    endpoints so the outlines close.
    """
    strokes = []
    filled = []
    lbl, n = label(black, connectivity=2, return_num=True)
    for ci in range(1, n + 1):
        comp = lbl == ci
        if comp.sum() < 3:
            continue
        edt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
        max_thick = float(edt.max())
        skel = skeletonize(comp)
        skel_len = int(skel.sum())
        if skel_len < 3 or skel_len <= 2.5 * (2.0 * max_thick):
            filled += contour_polygons(comp)  # blob (eye dot)
            continue
        branches = trace_branches(skel)
        branches = prune_spur_branches(
            branches, skel, max(3.0, 1.5 * (2.0 * max_thick)))
        if merge:
            branches = merge_branches_through_junctions(branches)
        for br in branches:
            half = branch_half_width_px(br, edt, min_line_width)
            width = (uniform_mm if uniform_mm > 0
                     else 2.0 * half * px_mm * width_scale)
            xy = _smooth_centerline([(c, r) for (r, c) in br], smooth_tol)
            if len(xy) < 2:
                continue
            line = LineString(xy)
            if clip_poly is not None:
                line = line.intersection(clip_poly)
            for seg in _flatten_lines(line):
                coords = np.asarray(seg.coords)
                if len(coords) >= 2:
                    d = beziers_to_open_d(fit_curve(coords, fit_tol), px_mm)
                    if d:
                        strokes.append((d, width))
    return strokes, filled


def leading_from_arcs(smoothed_arcs, black, min_line_width, cap_style, clip_poly,
                      uniform_px=0.0):
    """Build the leading by buffering the LEADED partition arcs to their width.

    The leading is derived from the SAME boundary arcs as the fragment faces, so
    every leading line is centred exactly on a fragment seam -> leading and
    fragments register perfectly.  An arc is 'leaded' if black runs along it (a
    colour-only seam, e.g. cheeks, has no black -> no leading).  Width is the
    local black thickness (distance transform).  Each arc is a separate simple
    ribbon (no boolean union -> printer-safe), clipped to the silhouette so the
    border leading sits inside the outline.
    """
    h, w = black.shape
    edt = ndimage.distance_transform_edt(
        np.pad(black, 1, constant_values=False))[1:-1, 1:-1]
    cap = {"flat": 2, "round": 1, "square": 3}.get(cap_style, 2)
    ribbons = []
    for s in smoothed_arcs:
        if len(s) < 2:
            continue
        halfs = []
        for (x, y) in s:
            c, r = int(round(x)), int(round(y))
            r0, r1 = max(0, r - 1), min(h, r + 2)
            c0, c1 = max(0, c - 1), min(w, c + 2)
            win = edt[r0:r1, c0:c1]
            m = float(win.max()) if win.size else 0.0
            if m > 0:
                halfs.append(m)
        if not halfs or len(halfs) < 0.4 * len(s):
            continue  # colour-only seam -> no leading
        if uniform_px > 0:
            half = uniform_px / 2.0
        else:
            half = max(float(np.median(halfs)) - 0.5, min_line_width / 2.0)
        rib = LineString(s).buffer(half, cap_style=cap, join_style=1)
        if not rib.is_empty:
            ribbons.append(rib)
    if clip_poly is not None:
        ribbons = _flatten_polys([r.intersection(clip_poly) for r in ribbons])
    return ribbons


def _flatten_lines(geom):
    """Flatten a (Multi)LineString / collection into a list of LineStrings."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out += _flatten_lines(g)
        return out
    return []


def _two_width_tiers(widths):
    """Snap a 1-D array of per-stroke widths to TWO tiers (thin / bold).

    Otsu threshold in LOG space -- log makes the tiers multiplicative (a bold
    line is ~2x a thin one) and stops a few very-bold outliers (e.g. foam
    bubbles) from dragging a plain Otsu/2-means threshold up so high that the
    thin background group gets lumped in with the bold one.  Each stroke gets
    its tier's MEDIAN width; falls back to one width if the tiers aren't clearly
    separated (< 1.4x).
    """
    a = np.asarray(widths, dtype=np.float64)
    if len(a) < 4 or (a.max() - a.min()) < 1.0:   # < ~1px spread -> one tier
        return np.full(len(a), float(np.median(a)))
    la = np.log(np.maximum(a, 1e-3))
    best_var, thr = -1.0, None
    for t in np.unique(la):
        lo, hi = la[la < t], la[la >= t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        var = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2  # between-class
        if var > best_var:
            best_var, thr = var, t
    if thr is None:
        return np.full(len(a), float(np.median(a)))
    thr = np.exp(thr)
    lo_w = float(np.median(a[a < thr]))
    hi_w = float(np.median(a[a >= thr]))
    if hi_w < 1.4 * max(lo_w, 1e-6):               # tiers too close -> one tier
        return np.full(len(a), float(np.median(a)))
    return np.where(a < thr, lo_w, hi_w)


def _link_arc_strokes(arcs, widths, angle_deg=35.0, width_ratio=1.7):
    """Link arcs that continue the SAME line through a junction into strokes.

    A long line (halo circle, a bold straight) is broken into many seam-arcs by
    crossing panes; per-arc it comes out with jittering width AND faceted.  At
    each junction we pair the two arc-ends that best fit a straight line THROUGH
    the junction (tangents most opposite) AND have the closest width, then walk
    those links into ordered chains.  Returns a list of chains, each a list of
    ``(arc_index, reversed)`` in path order -- so the caller can concatenate the
    pieces into ONE continuous line, give it one width, and smooth it as a whole.
    """
    import collections
    n = len(arcs)

    def tangent(a, end):
        a = np.asarray(a)
        k = min(len(a) - 1, 4)
        return (a[k] - a[0]) if end == 0 else (a[-1 - k] - a[-1])  # away

    ends = collections.defaultdict(list)
    for i, a in enumerate(arcs):
        if len(a) < 2:
            continue
        a = np.asarray(a)
        ends[(round(float(a[0][0])), round(float(a[0][1])))].append((i, 0))
        ends[(round(float(a[-1][0])), round(float(a[-1][1])))].append((i, 1))

    partner = {}                                     # (arc,end) -> (arc,end)
    for junc, lst in ends.items():
        if len(lst) < 2:
            continue
        cand = []
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                i, ei = lst[x]
                j, ej = lst[y]
                if i == j:
                    continue
                ti, tj = tangent(arcs[i], ei), tangent(arcs[j], ej)
                ni, nj = np.hypot(*ti), np.hypot(*tj)
                if ni < 1e-6 or nj < 1e-6:
                    continue
                cos = np.dot(ti, tj) / (ni * nj)     # both away; straight -> -1
                dev = np.degrees(np.arccos(np.clip(cos, -1, 1)))  # 180 = straight
                if dev <= 180 - angle_deg:            # not collinear enough
                    continue
                wi, wj = max(widths[i], 1e-6), max(widths[j], 1e-6)
                wr = max(wi, wj) / min(wi, wj)
                # width is a PRIORITY, not a hard filter: pass 0 = same-width
                # (pair these first, so crossings match bold<->bold / thin<->thin
                # correctly); pass 1 = collinear but different width (links only a
                # FREE end with no same-width continuation -- rescues a straight
                # band segment that was mis-measured thin so the whole band links
                # into one bold line instead of leaving thin gaps).
                tier = 0 if wr < width_ratio else 1
                cand.append((tier, 180 - dev, (i, ei), (j, ej)))
        cand.sort(key=lambda c: (c[0], c[1]))        # same-width first, then straightest
        used = set()
        for _tier, _dev, a, b in cand:
            if a in used or b in used:
                continue
            partner[a] = b
            partner[b] = a
            used.add(a)
            used.add(b)

    visited = set()
    chains = []

    def walk(i, e):                                  # e = end we ENTER arc i at
        chain = []
        while i not in visited:
            visited.add(i)
            chain.append((i, e == 1))                # entered at end 1 -> reversed
            nxt = partner.get((i, 1 - e))            # leave by the other end
            if nxt is None:
                break
            i, e = nxt
        return chain

    for i in range(n):                               # open chains (from free ends)
        if len(arcs[i]) < 2 or i in visited:
            continue
        for e in (0, 1):
            if (i, e) not in partner:
                chains.append(walk(i, e))
                break
    for i in range(n):                               # remaining closed loops
        if len(arcs[i]) >= 2 and i not in visited:
            chains.append(walk(i, 0))
    return chains


def _round_junctions(pts, jidx, fit_tol, max_disp):
    """Locally round the linked junctions of a merged chain.

    The chain's arcs were each Bezier-fit independently, so where two arcs meet
    they have MISMATCHED tangents -- a kink.  Re-fitting the whole chain keeps
    that kink (the fit subdivides at the high-error junction).  Since a linked
    junction is straight-through by construction (it only got linked because the
    two tangents nearly continue), blend a Gaussian-smoothed copy of the polyline
    into a small window around each junction so the join becomes a smooth arc,
    while the arc BODIES (away from junctions) stay exactly on their seam.

    CRITICAL: leading is a connected NETWORK -- a junction is a triple/quad point
    where OTHER arcs (a crossing line, a stub) also end.  Those arcs are not part
    of this chain and do NOT move, so if we pull the rounded chain too far off the
    junction they no longer touch it (gaps / odd crossings -- the "unstable
    junction" bug).  Fix: CLAMP every point's displacement to ``max_disp`` (half
    the chain's own stroke width).  Then the original junction point always stays
    within this stroke's painted band, so anything ending there still overlaps it,
    and the move is too small to manufacture a new intersection.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 5 or not jidx or max_disp <= 0:
        return pts
    sigma = max(1.5, 2.0 * fit_tol)
    sm = np.stack([ndimage.gaussian_filter1d(pts[:, 0], sigma, mode="nearest"),
                   ndimage.gaussian_filter1d(pts[:, 1], sigma, mode="nearest")],
                  axis=1)
    idx = np.arange(n)
    w = np.zeros(n)
    std = 1.6 * sigma                                # blend-window half-width
    for j in jidx:
        w = np.maximum(w, np.exp(-((idx - j) ** 2) / (2.0 * std * std)))
    w[0] = w[-1] = 0.0                               # pin the chain endpoints
    disp = (sm - pts) * w[:, None]
    dlen = np.hypot(disp[:, 0], disp[:, 1])
    scale = np.where(dlen > max_disp, max_disp / np.maximum(dlen, 1e-9), 1.0)
    return pts + disp * scale[:, None]


def leading_strokes_from_arcs(smoothed_arcs, black, min_line_width, clip_poly,
                              px_mm, fit_tol, uniform_mm=0.0, smooth=False,
                              width_scale=1.0, tier=True, arc_widths=None,
                              tier_thin_mm=0.0, tier_bold_mm=0.0,
                              link_angle_deg=35.0, link_width_ratio=1.7,
                              block_mask=None):
    """Leading from the LEADED partition arcs.  One constant width per stroke.

    Returns ``(stroke_items, ribbon_d)`` (ribbon_d kept for the caller's API but
    always empty).  Width per stroke:
    * ``uniform_mm > 0`` -> that fixed width for every line.
    * else the arc's measured black width (``arc_widths``, px, from the raw
      crack); ``tier=True`` snaps it to TWO tiers (bold foreground vs thin
      background) so a thin line is never averaged up toward a bold one;
      ``tier=False`` keeps each stroke's own measured width.
    ``width_scale`` multiplies the measured widths; ``smooth`` Bezier-fits.
    """
    h, w = black.shape
    edt = ndimage.distance_transform_edt(
        np.pad(black, 1, constant_values=False))[1:-1, 1:-1]

    def leaded(s):
        on = 0
        for (x, y) in s:
            c, r = int(round(x)), int(round(y))
            win = edt[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if win.size and win.max() > 0:
                on += 1
        return len(s) >= 2 and on >= 0.4 * len(s)

    # Arcs that run along a black-BLOCK edge are handled by the bold block CAME,
    # not the thin seam leading -- else the thin seam traces block-edge detail
    # (bulbs, corners) the came skips over and prints as a parallel DOUBLE line.
    block_near = None            # tight ring (2px): "arc IS the block edge"
    block_side = None            # wider ring: "arc runs ALONGSIDE a block"
    if block_mask is not None and block_mask.any():
        block_near = ndimage.binary_dilation(block_mask, iterations=2)
        # "alongside" band ~1.6mm, so a foreground outline that sits a little off
        # the block edge (e.g. a shoulder just above the sleeve, especially at a
        # higher --black-block-mm where the block shrinks) still gets the bias.
        side_it = max(5, int(round(1.6 / px_mm)))
        block_side = ndimage.binary_dilation(block_mask, iterations=side_it)

    def on_block_edge(s):
        if block_near is None:
            return False
        hit = 0
        for (x, y) in s:
            c, r = int(round(x)), int(round(y))
            if 0 <= r < h and 0 <= c < w and block_near[r, c]:
                hit += 1
        return len(s) >= 2 and hit >= 0.5 * len(s)

    def block_side_frac(coords):
        # fraction of a chain running ALONGSIDE a block -> a garment/foreground
        # outline (near the black block its own black is partly absorbed, so it
        # measures thin; bias it BOLD).
        if block_side is None or len(coords) == 0:
            return 0.0
        hit = 0
        for (x, y) in coords:
            c, r = int(round(x)), int(round(y))
            if 0 <= r < h and 0 <= c < w and block_side[r, c]:
                hit += 1
        return hit / len(coords)

    if arc_widths is None:
        arc_widths = [0.0] * len(smoothed_arcs)

    # Link arcs that continue the SAME line through each junction into ONE
    # stroke: concatenate the pieces, give the whole chain a single width
    # (median of members), and -- when smoothing -- fit the chain as one
    # continuous curve.  So a halo/bold line broken by crossings prints at a
    # consistent width and smoothly, instead of jittering thin<->bold and
    # faceting per segment.  ``link_angle_deg`` is the slope threshold: at a
    # junction two ends only continue each other when their tangents are within
    # this angle of a straight line -- a bigger slope difference is a corner,
    # not one continuous line, so the chain stops there.
    leaded_idx = [i for i, s in enumerate(smoothed_arcs)
                  if leaded(s) and not on_block_edge(s)]
    lead_arcs = [np.asarray(smoothed_arcs[i], dtype=float) for i in leaded_idx]
    lead_w = [max(float(arc_widths[i]), min_line_width) for i in leaded_idx]
    if len(lead_arcs) > 1:
        chains = _link_arc_strokes(lead_arcs, lead_w, angle_deg=link_angle_deg,
                                   width_ratio=link_width_ratio)
    else:
        chains = [[(i, False)] for i in range(len(lead_arcs))]

    def _merge_chain(chain):
        pts = []
        jidx = []                                    # indices of linked junctions
        for (ai, rev) in chain:
            a = lead_arcs[ai]
            a = a[::-1] if rev else a
            if pts:
                if np.hypot(pts[-1][0] - a[0][0],
                            pts[-1][1] - a[0][1]) < 1e-6:
                    a = a[1:]
                jidx.append(len(pts) - 1)            # the shared junction vertex
            pts.extend(tuple(p) for p in a)
        return np.asarray(pts, dtype=float), jidx

    def stroke_d(coords):
        return (beziers_to_open_d(fit_open_arc(coords, fit_tol), px_mm)
                if smooth else _polyline_open_d(coords, px_mm))

    items = []
    auto_jobs = []          # (coords, chain-width-px, chain-len-px)
    for chain in chains:
        cw = max(float(np.median([lead_w[ai] for ai, _ in chain])),
                 min_line_width)
        coords_full, jidx = _merge_chain(chain)
        if len(coords_full) < 2:
            continue
        chain_len = float(np.hypot(*(coords_full[1:] - coords_full[:-1]).T).sum())
        blk_frac = block_side_frac(coords_full)
        if smooth and jidx:                          # round the linked joins
            # cap the move at half the stroke width so the junction stays under
            # the stroke (keeps crossing/stub lines connected -- no gaps).
            coords_full = _round_junctions(coords_full, jidx, fit_tol,
                                           max_disp=0.5 * cw)
        line = LineString(coords_full)
        if clip_poly is not None:
            line = line.intersection(clip_poly)
        for seg in _flatten_lines(line):
            coords = np.asarray(seg.coords)
            if len(coords) < 2:
                continue
            seg_len = float(np.hypot(*(coords[1:] - coords[:-1]).T).sum())
            if uniform_mm > 0:
                width = uniform_mm / px_mm
                # Drop seam stubs shorter than their own width (a fat dot, not a
                # line): the two tiny panes just share no drawn lead there.
                if seg_len < width:
                    continue
                d = stroke_d(coords)
                if d:
                    items.append((d, width * px_mm))
            else:
                if seg_len < cw:
                    continue
                auto_jobs.append((coords, cw, chain_len, blk_frac))

    # Measured-width modes: 'tier' snaps the per-stroke widths to TWO tiers
    # (bold foreground vs thin background) so a thin line is never blended up
    # toward a bold one; 'auto' keeps each stroke's own measured width.
    bold_w_mm = 2.0 * min_line_width * px_mm         # fallback bold width
    if uniform_mm > 0:
        bold_w_mm = uniform_mm
    if auto_jobs:
        widths = np.array([wp for _, wp, _, _ in auto_jobs])
        lengths = np.array([ln for _, _, ln, _ in auto_jobs])
        blkfrac = np.array([bf for _, _, _, bf in auto_jobs])
        if tier:
            base = _two_width_tiers(widths)
            lo, hi = float(base.min()), float(base.max())
            # exact mm overrides win (no width_scale); else measured*scale
            thin_px = (tier_thin_mm / px_mm) if tier_thin_mm > 0 \
                else lo * width_scale
            bold_px = (tier_bold_mm / px_mm) if tier_bold_mm > 0 \
                else hi * width_scale
            thin_mask = base <= lo + 1e-9
            if hi > lo + 1e-9 and (~thin_mask).any() and thin_mask.any():
                # Bias a stroke's effective width UP before the bold cut, so a
                # borderline line leans BOLD when it is a main/foreground outline:
                #  - LENGTH: a long continuous outline (vs short thin splitters).
                #  - BLOCK adjacency: a line running alongside a black block is a
                #    garment/foreground outline whose black is partly absorbed by
                #    the block, so it measures thin (the "shoulder above the
                #    sleeve" case).
                thr = float(np.sqrt(widths[thin_mask].max()
                                    * widths[~thin_mask].min()))
                l90 = float(np.percentile(lengths, 90))
                ln_n = np.clip(lengths / max(l90, 1e-6), 0.0, 1.0)
                eff = widths * (1.0 + LENGTH_TIER_BIAS * ln_n
                                + BLOCK_TIER_BIAS * blkfrac)
                out_px = np.where(eff >= thr, bold_px, thin_px)
            else:
                out_px = np.where(thin_mask, thin_px, bold_px)
            bold_w_mm = bold_px * px_mm
        else:
            out_px = widths * width_scale
            bold_w_mm = float(np.percentile(out_px, 90)) * px_mm
        for (coords, _, _, _), gw in zip(auto_jobs, out_px):
            d = stroke_d(coords)
            if d:
                items.append((d, gw * px_mm))
    ribbons = []
    return items, ribbons, bold_w_mm


def _polyline_open_d(coords, px_mm):
    """Open polyline (M..L..) path-data straight from the vertices (no Z)."""
    s = px_mm
    if len(coords) < 2:
        return ""
    parts = ["M %s,%s" % (_fmt(coords[0][0] * s), _fmt(coords[0][1] * s))]
    for x, y in coords[1:]:
        parts.append("L %s,%s" % (_fmt(x * s), _fmt(y * s)))
    return " ".join(parts)


def _ring_polyline_d(coords, px_mm):
    """Closed ring (M..L..Z) path-data straight from the polygon vertices."""
    s = px_mm
    if len(coords) > 1 and np.linalg.norm(coords[0] - coords[-1]) < 1e-9:
        coords = coords[:-1]
    if len(coords) < 3:
        return ""
    parts = ["M %s,%s" % (_fmt(coords[0][0] * s), _fmt(coords[0][1] * s))]
    for x, y in coords[1:]:
        parts.append("L %s,%s" % (_fmt(x * s), _fmt(y * s)))
    parts.append("Z")
    return " ".join(parts)


def perimeter_came_strokes(clip_poly, px_mm, uniform_mm, min_line_width,
                           black=None, smooth=False, simplify_tol=1.2,
                           fit_tol=0.4, width_scale=1.0):
    """The silhouette outline (+ holes) as closed stroked leading rings.

    A perimeter 'came': the lead line that frames the whole piece.  Each ring is
    emitted as the EXACT clip_poly polyline (NOT a re-Bezier-fit) so the came
    sits precisely on the fragment outer edges, which are clipped to the same
    clip_poly.  With ``uniform_mm <= 0`` (auto) the width is the median black
    thickness measured along the ring; otherwise the uniform width is used.
    """
    if clip_poly is None or clip_poly.is_empty:
        return []
    edt = None
    if uniform_mm <= 0 and black is not None:
        edt = ndimage.distance_transform_edt(
            np.pad(black, 1, constant_values=False))[1:-1, 1:-1]
        h, w = black.shape

    def ring_width_mm(coords):
        if uniform_mm > 0:
            return uniform_mm
        if edt is None:
            return min_line_width * px_mm
        vals = []
        for (x, y) in coords:
            c, r = int(round(x)), int(round(y))
            if 0 <= r < h and 0 <= c < w:
                win = edt[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                m = float(win.max()) if win.size else 0.0
                if m > 0:
                    vals.append(m)
        half = (max(float(np.median(vals)) - 0.5, min_line_width / 2.0)
                if vals else min_line_width / 2.0)
        return 2.0 * half * px_mm * width_scale

    items = []
    geoms = (clip_poly.geoms if clip_poly.geom_type == "MultiPolygon"
             else [clip_poly])
    for g in geoms:
        for ring in [g.exterior] + list(g.interiors):
            coords = np.asarray(ring.coords)
            if len(coords) < 4:
                continue
            if smooth:
                d = beziers_to_d(
                    _ring_to_beziers(coords, simplify_tol, fit_tol), px_mm)
            else:
                d = _ring_polyline_d(coords, px_mm)
            if d:
                items.append((d, ring_width_mm(coords)))
    return items


def perimeter_came_polys(clip_poly, px_mm, uniform_mm, min_line_width):
    """The silhouette outline as a filled ribbon (for 'fill' leading style)."""
    if clip_poly is None or clip_poly.is_empty:
        return []
    half = (uniform_mm / px_mm if uniform_mm > 0 else min_line_width) / 2.0
    geoms = (clip_poly.geoms if clip_poly.geom_type == "MultiPolygon"
             else [clip_poly])
    polys = []
    for g in geoms:
        for ring in [g.exterior] + list(g.interiors):
            ribbon = LineString(ring.coords).buffer(half)
            ribbon = ribbon.intersection(clip_poly)  # stay inside the piece
            polys.extend(_flatten_polys([ribbon]))
    return polys


def block_came_strokes(block_mask, clip_poly, px_mm, came_mm, simplify_tol,
                       min_line_width, smooth=False, fit_tol=0.4, min_area=0.0):
    """Bold came ringing each black-block (garment) region.

    A black block is a big BLACK-GLASS pane; in real stained glass its outline is
    a bold CAME, not a thin seam.  The partition seam along a block edge measures
    ~0 width (it runs at the EDGE of the black, not down a line's middle) so it
    would tier THIN and look patchy -- the "leg edge jitters / shoulder should be
    bold" problem.  Instead ring each block with ONE bold continuous came: build
    a polygon from the (already boundary-smoothed) block mask, DP-simplify each
    ring for a clean curve, clip to the silhouette, and emit at ``came_mm``.
    """
    if block_mask is None or not block_mask.any():
        return []
    # Ring the block ON its true edge (do NOT morphologically open it first: an
    # opening shifts the came inward off the real boundary, so it no longer
    # covers the block-edge seam -> a parallel thin+bold DOUBLE line).  DP-simplify
    # the ring for a clean curve; small specks/thin bits are dropped below.
    poly = _mask_to_polygon(block_mask)
    if poly is None or poly.is_empty:
        return []
    poly = poly.simplify(max(simplify_tol, 1.0), preserve_topology=True)
    if poly.is_empty:
        return []
    w = max(came_mm, min_line_width * px_mm)
    came_px = w / px_mm
    items = []
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        if g.is_empty or g.geom_type != "Polygon":
            continue
        # Only ring SUBSTANTIAL, solid blocks (the garment body).  Skip:
        #  - blocks below the area floor (small isolated specks / gem-ring bits
        #    that came out as their own blocks -> a came just draws a stray
        #    circle; they stay filled black glass), and
        #  - blocks so thin the came would fill them (inradius proxy
        #    2*area/perimeter < 1.5*came_width -> a dot/annulus, not a region).
        if g.area < min_area:
            continue
        if 2.0 * g.area / max(g.exterior.length, 1e-6) < 1.5 * came_px:
            continue
        # Skip ELONGATED blocks (a thick band/stripe, aspect > ~2.5): ringing one
        # makes a long thin came LOOP = two parallel bold lines (the "double line
        # loop").  It stays filled black glass -> reads as a bold black band,
        # which is what a divider should look like.  Only BULKY blocks (the
        # garment body) get a came.  Elongation via PCA of the boundary points
        # (eigenvalue ratio ~ aspect^2) -- robust to degenerate/sliver polygons,
        # unlike shapely's minimum_rotated_rectangle (oriented_envelope emits
        # divide-by-zero RuntimeWarnings on thin slivers).
        try:
            pts = np.asarray(g.exterior.coords)[:-1].astype(float)
            pts = pts - pts.mean(0)
            ev = np.linalg.eigvalsh(np.cov(pts.T))       # ascending
            if ev[0] > 1e-9 and np.sqrt(ev[1] / ev[0]) > 2.5:
                continue
        except Exception:
            pass
        for ring in [g.exterior] + list(g.interiors):
            coords = np.asarray(ring.coords)
            if len(coords) < 4:
                continue
            line = LineString(coords)
            if clip_poly is not None:
                line = line.intersection(clip_poly)
            for seg in _flatten_lines(line):
                c = np.asarray(seg.coords)
                if len(c) < 2:
                    continue
                closed = np.linalg.norm(c[0] - c[-1]) < 1e-6
                if smooth:
                    d = (beziers_to_d(_ring_to_beziers(c, simplify_tol, fit_tol),
                                      px_mm) if closed
                         else beziers_to_open_d(fit_open_arc(c, fit_tol), px_mm))
                else:
                    d = (_ring_polyline_d(c, px_mm) if closed
                         else _polyline_open_d(c, px_mm))
                if d:
                    items.append((d, w))
    return items


def write_fragments_png(frag_rgb, labels, path):
    """Write the fragments PNG: every silhouette pixel coloured, rest clear."""
    h, w = labels.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = frag_rgb
    out[:, :, 3] = np.where(labels > 0, 255, 0).astype(np.uint8)
    Image.fromarray(out, mode="RGBA").save(path)


def rasterize_polys(polys, shape):
    """Rasterise union polygons (exterior filled, holes cut) to a bool mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polys:
        ext = np.round(np.asarray(poly.exterior.coords)).astype(np.int32)
        cv2.fillPoly(mask, [ext], 255)
        for ring in poly.interiors:
            hole = np.round(np.asarray(ring.coords)).astype(np.int32)
            cv2.fillPoly(mask, [hole], 0)
    return mask > 0


def rasterize_fitted_leading(polys, shape, simplify_tol, fit_tol):
    """Rasterise the Bezier-FITTED leading (matches the emitted leading SVG).

    Used to clip fragment overlap so glass never pokes past the printed
    leading.  Each ring is fit and sampled exactly like the leading output.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polys:
        rings = [(np.asarray(poly.exterior.coords), 255)]
        rings += [(np.asarray(r.coords), 0) for r in poly.interiors]
        for ring, val in rings:
            if len(ring) < 4:
                continue
            pts = _sample_beziers(_ring_to_beziers(ring, simplify_tol, fit_tol),
                                  step=0.5)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.round(np.asarray(pts)).astype(np.int32)],
                             val)
    return mask > 0


def _parse_svg_subpaths(d):
    """Parse an SVG path 'd' into subpaths as point lists, flattening cubics."""
    toks = re.findall(r"[MLCZmlcz]|-?\d+\.?\d*(?:e-?\d+)?", d)
    i, cur, start, subs, pts = 0, None, None, [], []

    def flush():
        if len(pts) >= 2:
            subs.append(list(pts))
        pts.clear()

    while i < len(toks):
        t = toks[i]
        if t in "Mm":
            flush()
            cur = (float(toks[i + 1]), float(toks[i + 2])); i += 3
            start = cur; pts.append(cur)
        elif t in "Ll":
            cur = (float(toks[i + 1]), float(toks[i + 2])); i += 3
            pts.append(cur)
        elif t in "Cc":
            p0 = cur
            c1 = (float(toks[i + 1]), float(toks[i + 2]))
            c2 = (float(toks[i + 3]), float(toks[i + 4]))
            p3 = (float(toks[i + 5]), float(toks[i + 6])); i += 7
            for k in range(1, 9):
                u = k / 8.0; v = 1 - u
                pts.append((
                    v**3 * p0[0] + 3*v*v*u*c1[0] + 3*v*u*u*c2[0] + u**3*p3[0],
                    v**3 * p0[1] + 3*v*v*u*c1[1] + 3*v*u*u*c2[1] + u**3*p3[1]))
            cur = p3
        elif t in "Zz":
            if start:
                pts.append(start)
            i += 1
        else:
            i += 1
    flush()
    return subs


def write_preview_png(frag_svg, lead_svg, px_mm, path, ss=3):
    """Anti-aliased 'what-you-print' preview.

    Rasterizes the FRAGMENT and LEADING SVGs (the actual vector output) at
    ``ss``x supersample and box-downscales with LANCZOS, so pane and leading
    edges are smooth -- matching what a system SVG renderer shows, instead of
    the hard pixel edges of a label-map render.
    """
    def attr(tag, name):
        m = re.search(r'%s="([^"]*)"' % name, tag)
        return m.group(1) if m else None

    def hexrgb(s):
        s = s.lstrip("#")
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', frag_svg)
    w_mm, h_mm = float(m.group(1)), float(m.group(2))
    s = ss / px_mm                                    # mm -> supersampled px
    W, H = max(1, int(round(w_mm * s))), max(1, int(round(h_mm * s)))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)

    def sc(sub):
        return [(x * s, y * s) for x, y in sub]

    # 1) glass faces (skip the corner registration squares = black fills).
    for tag in re.findall(r"<path\b[^>]*>", frag_svg):
        fill = attr(tag, "fill")
        d = attr(tag, "d")
        if not d or not fill or fill in ("none", "#000000"):
            continue
        col = hexrgb(fill) + (255,)
        for sub in _parse_svg_subpaths(d):
            poly = sc(sub)
            if len(poly) >= 3:
                drw.polygon(poly, fill=col)

    # 2) black leading on top (strokes at their width; filled blobs; skip the
    #    tiny <=1.3mm registration squares).
    for tag in re.findall(r"<path\b[^>]*>", lead_svg):
        d = attr(tag, "d")
        if not d:
            continue
        subs = _parse_svg_subpaths(d)
        sw = attr(tag, "stroke-width")
        if sw:
            wpx = max(1, int(round(float(sw) * s)))
            for sub in subs:
                poly = sc(sub)
                if len(poly) >= 2:
                    drw.line(poly, fill=(0, 0, 0, 255), width=wpx,
                             joint="curve")
        elif (attr(tag, "fill") or "") == "#000000":
            xs = [x for sub in subs for x, y in sub]
            ys = [y for sub in subs for x, y in sub]
            if xs and (max(xs) - min(xs) <= 1.3 and max(ys) - min(ys) <= 1.3):
                continue                              # registration mark
            for sub in subs:
                poly = sc(sub)
                if len(poly) >= 3:
                    drw.polygon(poly, fill=(0, 0, 0, 255))

    img.resize((max(1, W // ss), max(1, H // ss)), Image.LANCZOS).save(path)


# ===========================================================================
# CLI / orchestration
# ===========================================================================
def _line_width_arg(v):
    """Parse --line-width: 'tier' / 'auto' (keywords) or a number (fixed mm)."""
    s = str(v).strip().lower()
    if s in ("tier", "auto"):
        return s
    try:
        f = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--line-width must be 'tier', 'auto', or a width in mm")
    if f <= 0:
        raise argparse.ArgumentTypeError(
            "a fixed --line-width must be > 0 mm (use 'tier' or 'auto')")
    return f


def parse_command_line(args):
    p = argparse.ArgumentParser(
        description="Convert a PNG into stained-glass SVG + fragment PNG.")
    p.add_argument("input_png", help="input PNG image")
    p.add_argument("--silhouette-svg", help="output path for silhouette SVG")
    p.add_argument("--leading-svg", help="output path for leading SVG")
    p.add_argument("--fragments-png", help="output path for fragments PNG")
    p.add_argument("--fragments-svg", help="output path for fragments SVG")
    p.add_argument("--fragments-dir",
                   help="folder for the per-colour fragment SVGs (default "
                        "<input>_fragments/); one SVG per glass colour, since the "
                        "3D-printer app ignores SVG fill colour")
    p.add_argument("--px-mm", type=float, default=0.4,
                   help="millimetres per pixel (default 0.4)")
    p.add_argument("--max-size-mm", type=float, default=250.0,
                   help="max printable size per side in mm; the image is "
                        "downscaled to fit (default 250)")
    p.add_argument("--autocrop", dest="autocrop", action="store_true",
                   default=True,
                   help="trim fully-transparent border pixels to the content "
                        "bbox before sizing/scaling (default on)")
    p.add_argument("--no-autocrop", dest="autocrop", action="store_false",
                   help="keep the original canvas including transparent padding")
    p.add_argument("--connectivity", type=int, choices=(4, 8), default=8,
                   help="connectivity for the silhouette region check")
    p.add_argument("--lum-threshold", type=int, default=90,
                   help="leading/black if ALL of R,G,B < this (0-255, default "
                        "90). Catches anti-aliased thin line cores (dark grey) "
                        "so faint dividers stay continuous; still excludes "
                        "saturated dark colours like navy (max channel high)")
    p.add_argument("--alpha-min", type=int, default=128,
                   help="opaque if alpha >= this (0-255, default 128); higher "
                        "values drop faint anti-alias edge pixels (less noise)")
    p.add_argument("--fit-tolerance", type=float, default=0.4,
                   help="max Bezier fit error in px (default 0.4)")
    p.add_argument("--simplify-tolerance", type=float, default=0.3,
                   help="ring simplify tolerance in px (default 0.3)")
    p.add_argument("--smooth-tolerance", type=float, default=1.2,
                   help="de-staircase tolerance in px for centrelines and the "
                        "silhouette contour (default 1.2)")
    p.add_argument("--cap-style", choices=("flat", "round", "square"),
                   default="flat", help="leading line end caps (default flat)")
    p.add_argument("--line-width", type=_line_width_arg, default="tier",
                   metavar="tier|auto|MM",
                   help="leading width mode: 'tier' (default) = each stroke's "
                        "measured width snapped to TWO tiers (bold foreground "
                        "outlines vs thin background splitters); 'auto' = each "
                        "stroke its OWN measured average width; or a NUMBER = one "
                        "fixed uniform width in mm")
    p.add_argument("--line-width-scale", type=float, default=1.0,
                   help="in 'tier'/'auto' modes, multiply every measured width by "
                        "this factor to thicken (>1) or thin (<1) the leading "
                        "(default 1.0)")
    p.add_argument("--tier-thin", type=float, default=0.0,
                   help="in 'tier' mode, force the THIN-tier width to exactly "
                        "this many mm (0 = auto from the measured black; "
                        "overrides --line-width-scale for the thin tier)")
    p.add_argument("--tier-bold", type=float, default=0.0,
                   help="in 'tier' mode, force the BOLD-tier width to exactly "
                        "this many mm (0 = auto)")
    p.add_argument("--link-angle", type=float, default=35.0,
                   help="slope threshold (deg) for linking arcs into one "
                        "continuous line at a junction: two ends join only if "
                        "their tangents are within this angle of a straight line "
                        "(bigger difference = corner, not one line; default 35)")
    p.add_argument("--link-width-ratio", type=float, default=1.7,
                   help="max width ratio for linking two arcs into one line at a "
                        "junction (arcs of very different width are not the same "
                        "line; default 1.7)")
    p.add_argument("--smooth-curves", action="store_true",
                   help="emit leading as smooth Bezier curves instead of the raw "
                        "arc polyline (smoother look on curvy art; off by default)")
    p.add_argument("--merge-leading", action="store_true",
                   help="merge leading strokes through junctions into longer "
                        "continuous lines (default off = one stroke per branch)")
    p.add_argument("--perimeter-came", dest="perimeter_came",
                   action="store_true", default=True,
                   help="add the silhouette outline (and holes) as leading -- a "
                        "perimeter 'came' line around the whole piece (default on)")
    p.add_argument("--no-perimeter-came", dest="perimeter_came",
                   action="store_false",
                   help="do not lead the silhouette outline")
    p.add_argument("--min-line-width", type=float, default=1.5,
                   help="minimum leading width in px (default 1.5 = 0.6mm at "
                        "0.4mm/px)")
    p.add_argument("--min-black-area", type=int, default=2,
                   help="drop black components smaller than this many px")
    p.add_argument("--black-block-mm", type=float, default=3.0,
                   help="black regions THICKER than this (mm) are treated as "
                        "black GLASS fragments, not leading (e.g. a dark garment)"
                        "; only thinner black stays as leading. 0 disables "
                        "(all black = leading). Default 3")
    p.add_argument("--fragment-color", choices=("original", "quantized"),
                   default="quantized",
                   help="'original' keeps real pane colours (merged down to "
                        "the slot limit if needed); 'quantized' always reduces "
                        "to --num-colors (default)")
    p.add_argument("--num-colors", type=int, default=DEFAULT_PRINT_COLORS,
                   help="max distinct glass colours (hard max %d; black leading "
                        "is printed separately). One per-colour fragment SVG is "
                        "written, so at most this many. Default %d"
                        % (MAX_PRINT_COLORS, DEFAULT_PRINT_COLORS))
    p.add_argument("--segment-mode", choices=("color", "leading"),
                   default="color",
                   help="'color' splits panes by colour and leading (keeps "
                        "lead-less colour patches like cheeks); 'leading' "
                        "splits only by the black leading. Default color")
    p.add_argument("--min-fragment-area", type=int, default=32,
                   help="merge colour panes smaller than this many px into a "
                        "neighbour (color mode only, default 32)")
    p.add_argument("--color-merge-tol", type=float, default=COLOR_MERGE_TOL,
                   help="merge glass-adjacent panes (no black between) whose mean"
                        " RGB differ by less than this, to heal spurious colour "
                        "cross-cuts (default %.0f); higher = more merging, 0 off. "
                        "Never merges black-lined dividers." % COLOR_MERGE_TOL)
    p.add_argument("--fragment-overlap", type=float, default=2.0,
                   help="grow each fragment pane by this many px into its "
                        "neighbours so panes tile with no gaps (default 2)")
    p.add_argument("--reg-mark-size", type=float, default=REG_MARK_MM,
                   help="size in mm of the corner registration marks added to "
                        "every SVG so they align by content bbox (0 disables, "
                        "default %.1f)" % REG_MARK_MM)
    p.add_argument("--preview", action="store_true",
                   help="also write a composite preview PNG (glass + leading)")
    p.add_argument("--preview-png", help="output path for the preview PNG")
    p.add_argument("--silhouette-fill", choices=("none", "black"),
                   default="none",
                   help="silhouette as stroked outline (none) or filled")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(args)


def _default_out(input_png, suffix):
    stem, _ = os.path.splitext(input_png)
    return stem + suffix


def crop_to_content(rgba, alpha_min):
    """Trim fully-transparent border rows/cols to the non-transparent bbox.

    So a PNG with transparent padding prints at its true content size (and the
    max-size scaling is decided on the content, not the padded canvas).  Returns
    (cropped_rgba, (left, top)) -- the offset is informational; all three output
    SVGs share the cropped canvas so they still register.
    """
    opaque = rgba[:, :, 3] >= alpha_min
    if not opaque.any():
        return rgba, (0, 0)
    ys, xs = np.where(opaque)
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    if r0 == 0 and c0 == 0 and r1 == rgba.shape[0] and c1 == rgba.shape[1]:
        return rgba, (0, 0)
    return rgba[r0:r1, c0:c1], (c0, r0)


def fit_to_max_size(rgba, px_mm, max_size_mm):
    """Downscale the image so neither side exceeds ``max_size_mm`` when printed.

    Returns the (possibly resized) RGBA array and the scale factor applied.
    Keeps ``px_mm`` constant (1 px stays = nozzle size) by reducing pixels.
    """
    h, w = rgba.shape[:2]
    longest_mm = max(w, h) * px_mm
    if longest_mm <= max_size_mm:
        return rgba, 1.0
    scale = max_size_mm / longest_mm
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(rgba, mode="RGBA").resize(
        (new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8), scale


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    opts = parse_command_line(args)
    sys.setrecursionlimit(10000)
    global REG_MARK_MM
    REG_MARK_MM = opts.reg_mark_size

    def log(msg):
        if opts.verbose:
            sys.stderr.write(msg + "\n")

    try:
        # All SVGs go into one output folder (<input>_fragments/ or
        # --fragments-dir); the PNGs stay beside the input unless overridden.
        out_dir = opts.fragments_dir or _default_out(opts.input_png, "_fragments")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(opts.input_png))[0]

        def _in_dir(suffix):
            return os.path.join(out_dir, stem + suffix)

        sil_path = opts.silhouette_svg or _in_dir("_silhouette.svg")
        lead_path = opts.leading_svg or _in_dir("_leading.svg")
        frag_path = opts.fragments_png or _in_dir("_fragments.png")

        rgba = load_rgba(opts.input_png)
        if opts.autocrop:
            ph, pw = rgba.shape[:2]
            rgba, (cl, ct) = crop_to_content(rgba, opts.alpha_min)
            ch, cw = rgba.shape[:2]
            if (cw, ch) != (pw, ph):
                log("auto-cropped transparent border: %dx%d -> %dx%d px "
                    "(offset %d,%d)" % (pw, ph, cw, ch, cl, ct))
        # Extract leading + fragments at the ORIGINAL pixel resolution (so thin
        # black lines are never broken by a raster downscale), then scale the
        # resulting VECTORS down to fit the printer.  Instead of shrinking the
        # image, we shrink the mm-per-px (out_px_mm) and grow the px-space
        # thresholds by the same factor `up`, so behaviour matches the old
        # downscaled pipeline but with full-res fidelity.
        h, w = rgba.shape[:2]
        natural_mm = max(w, h) * opts.px_mm
        up = (natural_mm / opts.max_size_mm) if natural_mm > opts.max_size_mm \
            else 1.0
        opts.px_mm = opts.px_mm / up            # output mm-per-px (vector scale)
        opts.min_line_width *= up
        opts.smooth_tolerance *= up
        opts.fit_tolerance *= up
        opts.min_fragment_area = max(1, int(round(opts.min_fragment_area * up * up)))
        opts.min_black_area = max(1, int(round(opts.min_black_area * up * up)))
        if up > 1.0:
            log("extracting at full %dx%d px; vectors scaled by 1/%.3f to fit "
                "%.0f mm (%.4f mm/px)" % (w, h, up, opts.max_size_mm, opts.px_mm))
        width_mm, height_mm = w * opts.px_mm, h * opts.px_mm
        log("processing %dx%d px -> %.2fx%.2f mm" % (w, h, width_mm, height_mm))

        opaque = opaque_mask(rgba, opts.alpha_min)

        # --- SVG 1: silhouette -------------------------------------------
        opaque, dropped = keep_largest_component(opaque, opts.connectivity)
        if dropped:
            sys.stderr.write(
                "warning: silhouette had %d disconnected opaque regions; kept "
                "the largest (%d px) and dropped %d other(s) (sizes: %s px)\n"
                % (len(dropped) + 1, int(opaque.sum()), len(dropped),
                   ", ".join(str(s) for s in dropped)))
        outer, holes = silhouette_contours(opaque)
        clip_poly = build_silhouette_polygon(outer, holes, opts.smooth_tolerance)
        # Emit the silhouette from the SAME clip_poly polyline used to clip the
        # fragment panes and to draw the perimeter came -> all three outlines are
        # identical and register exactly (a re-Bezier-fit would diverge from the
        # polyline fragment edges, e.g. the left triangle edge).
        if clip_poly is not None:
            sil_d = _polygon_polyline_d(clip_poly, opts.px_mm)
        else:
            sil_d = contours_to_paths(outer, holes, opts.px_mm,
                                      opts.smooth_tolerance, opts.fit_tolerance)
        if opts.silhouette_fill == "black":
            sil_svg = write_svg([sil_d], width_mm, height_mm,
                                fill="black", stroke="none", stroke_width=0)
        else:
            sil_svg = write_svg([sil_d], width_mm, height_mm,
                                fill="none", stroke="black",
                                stroke_width=0.1)
        with open(sil_path, "w") as f:
            f.write(sil_svg)
        log("wrote %s (%d holes)" % (sil_path, len(holes)))

        # --- Partition: glass panes (labels) shared by leading + fragments --
        black = black_mask(rgba, opaque, opts.lum_threshold, opts.min_black_area)
        # Thick black areas (e.g. a dark garment) are BLOCKS -> treat as black
        # GLASS fragments; only the thin black lines stay as leading.
        block_px = (opts.black_block_mm / opts.px_mm
                    if opts.black_block_mm > 0 else 0.0)
        black, black_block = split_black_lines_blocks(black, block_px)
        if black_block.any():
            log("detected %d black-block px -> black glass fragments (leading "
                "keeps the thin lines)" % int(black_block.sum()))
        num_colors = opts.num_colors
        if num_colors > MAX_PRINT_COLORS:
            sys.stderr.write(
                "warning: --num-colors %d exceeds the %d printer colour slots; "
                "clamping to %d\n"
                % (num_colors, MAX_PRINT_COLORS, MAX_PRINT_COLORS))
            num_colors = MAX_PRINT_COLORS
        glass = opaque & ~black  # real pane colours (exclude black leading)
        if opts.segment_mode == "color":
            labels = split_into_fragments_by_color(
                opaque, black, rgba[:, :, :3], num_colors,
                opts.min_fragment_area, opts.color_merge_tol)
        else:
            labels = split_into_fragments(opaque, black)
        frag_rgb, n_used = fragment_colors(rgba, labels, glass,
                                           opts.fragment_color, num_colors)
        write_fragments_png(frag_rgb, labels, frag_path)
        n_frag = len([i for i in np.unique(labels) if i != 0])
        log("wrote %s (%d fragments, %d glass colours)"
            % (frag_path, n_frag, n_used))

        # Boundary arcs traced + smoothed ONCE; shared by leading and fragments
        # so leading lines sit exactly on the fragment seams.  Fragments use ALL
        # arcs (faces must close at the border); the seam leading uses only the
        # interior arcs when the perimeter came is on, so the silhouette outline
        # is leaded exactly once (came) instead of doubled (came + border arc).
        # --smooth-curves keeps more arc detail (lower DP floor) so tight
        # curves (wave-foam spirals) stay smooth, not decimated, before the
        # Bezier fit; the default 2.0 floor de-noises the pixel staircase.
        arc_floor = (0.8 if opts.smooth_curves else 2.0) * up
        arcs, arc_w, interior_arcs, interior_w = compute_partition_arcs(
            labels, opts.smooth_tolerance, opts.fit_tolerance, arc_floor, black,
            up)
        lead_arcs, lead_w = ((interior_arcs, interior_w) if opts.perimeter_came
                             else (arcs, arc_w))

        # --- SVG 2: leading (leaded shared arcs; filled ribbons or strokes) -
        n_lead = 0
        lead_polys = []  # leading region polygons (for the preview composite)
        if not black.any():
            sys.stderr.write(
                "warning: no black pixels above threshold; "
                "leading SVG will be empty\n")
            lead_svg = write_svg([], width_mm, height_mm,
                                 fill="black", stroke="none", stroke_width=0)
        else:
            # Leading = the partition SEAM arcs (the medial of the black between
            # panes).  A black line gives its centreline; a thin coloured pane's
            # black outline gives a loop AROUND the pane (so thin branches stay
            # enclosed instead of collapsing to a single line, which the all-
            # black skeleton does).  Coverage check (measured width) finds the
            # non-seam black (texture dots) to add as 'extra' WITHOUT duplicating
            # seam lines.
            # Resolve the width MODE: a number = fixed uniform mm; 'tier' =
            # 2-tier measured; 'auto' = per-stroke measured (no tiers).
            if isinstance(opts.line_width, str):
                uniform_mm = 0.0
                width_tier = (opts.line_width == "tier")
            else:
                uniform_mm = opts.line_width
                width_tier = False
            uniform_px = uniform_mm / opts.px_mm if uniform_mm > 0 else 0.0
            # Coverage uses ALL arcs (incl. the border) so border black counts
            # as already-leaded and is not re-added as 'extra' (which would
            # double the perimeter came).
            cover_polys = leading_from_arcs(arcs, black, opts.min_line_width,
                                            opts.cap_style, clip_poly, 0.0)
            arc_mask = (rasterize_polys(cover_polys, (h, w)) if cover_polys
                        else np.zeros((h, w), dtype=bool))
            # Dilate the seam-ribbon coverage by ~the resolution factor so the
            # black line's EDGES (which sit just outside a measured-width ribbon,
            # and farther out at full resolution) count as covered -- otherwise
            # they leak into 'extra' and print as small double lines beside the
            # seams.
            covered = ndimage.binary_dilation(
                arc_mask, np.ones((3, 3), bool),
                iterations=max(2, int(round(2 * up))))
            # The perimeter came covers the silhouette-border black, but the
            # border ARC sits at the glass/black interface where the black EDT
            # ~= 0, so its coverage ribbon is ~empty and the frame black would
            # fall through to 'extra' -> a skeleton stroke parallel to the came
            # (a double line along the edge).  Mark a band along the silhouette
            # boundary (as wide as the local frame black) as covered.
            if opts.perimeter_came and clip_poly is not None:
                edt_black = ndimage.distance_transform_edt(black)
                ring1 = rasterize_polys(
                    _flatten_polys([clip_poly.boundary.buffer(up)]), (h, w))
                frame_half = (float(edt_black[black & ring1].max())
                              if (black & ring1).any() else 0.0)
                band_w = max(frame_half + up, uniform_px / 2.0 + up, 2.0 * up)
                border_band = rasterize_polys(
                    _flatten_polys([clip_poly.boundary.buffer(band_w)]), (h, w))
                covered = covered | border_band
            uncovered = black & ~covered
            extra_mask = np.zeros((h, w), dtype=bool)
            if uncovered.any():
                ul, _un = label(uncovered, connectivity=2, return_num=True)
                sizes = np.bincount(ul.ravel())
                keep = sizes >= max(int(8 * up * up), opts.min_black_area)
                keep[0] = False
                extra_mask = keep[ul]
            # Seam arcs -> leading strokes.  fixed mm -> one width; 'tier' ->
            # two measured tiers; 'auto' -> each stroke its own measured width.
            # lead_arcs excludes the border when the came is on (no double).
            stroke_items, seam_ribbons, bold_w_mm = leading_strokes_from_arcs(
                lead_arcs, black, opts.min_line_width, clip_poly, opts.px_mm,
                opts.fit_tolerance, uniform_mm, opts.smooth_curves,
                opts.line_width_scale, width_tier, lead_w,
                opts.tier_thin, opts.tier_bold,
                opts.link_angle, opts.link_width_ratio,
                black_block if opts.black_block_mm > 0 else None)
            # Uncovered black (lone lines, texture, dots) -> skeleton strokes +
            # filled blobs, so nothing is dropped or doubled.
            extra_polys = []
            if extra_mask.any():
                ex_strokes, ex_filled = leading_skeleton_strokes(
                    extra_mask, opts.min_line_width, opts.smooth_tolerance,
                    opts.px_mm, opts.fit_tolerance, uniform_mm,
                    clip_poly, opts.merge_leading, opts.line_width_scale)
                stroke_items = stroke_items + ex_strokes
                extra_polys = ex_filled
            if opts.perimeter_came:
                # Came stays a POLYLINE even in smooth mode so it keeps matching
                # the silhouette + fragment outer edges exactly (a bezier came
                # would diverge -> a border rim).
                stroke_items = stroke_items + perimeter_came_strokes(
                    clip_poly, opts.px_mm, uniform_mm,
                    opts.min_line_width, black, False, opts.smooth_tolerance,
                    opts.fit_tolerance, opts.line_width_scale)
            # Bold came ringing each black block (garment): its outline is a came,
            # not a thin seam.  Width = fixed line width, or the bold tier.
            if opts.black_block_mm > 0 and black_block.any():
                came_mm = uniform_mm if uniform_mm > 0 else bold_w_mm
                # Only outline garment-scale blocks: a block worth a came is much
                # bigger than the opening disk (~0.79*block_px^2); small black
                # bits stay filled black glass (no stray circle).
                block_min_area = 8.0 * block_px * block_px
                stroke_items = stroke_items + block_came_strokes(
                    black_block, clip_poly, opts.px_mm, came_mm,
                    opts.smooth_tolerance, opts.min_line_width,
                    opts.smooth_curves, opts.fit_tolerance, block_min_area)
            filled_d = [polygon_to_path(p, opts.px_mm, opts.smooth_tolerance,
                                        opts.fit_tolerance)
                        for p in extra_polys] + seam_ribbons
            n_lead = len(stroke_items) + len(filled_d)
            lead_svg = write_stroke_svg(stroke_items, filled_d,
                                        width_mm, height_mm)
            lead_polys = list(cover_polys) + list(extra_polys)  # preview
        with open(lead_path, "w") as f:
            f.write(lead_svg)
        log("wrote %s (%d leading)" % (lead_path, n_lead))

        # --- SVG 3: glass fragments (faces from the SAME shared arcs) ----
        frag_svg_path = opts.fragments_svg or _in_dir("_fragments.svg")
        colored = fragments_to_colored_paths(frag_rgb, labels, arcs, clip_poly,
                                             opts.px_mm)
        frag_svg = write_multicolor_svg(colored, width_mm, height_mm)
        with open(frag_svg_path, "w") as f:
            f.write(frag_svg)
        log("wrote %s (%d vector panes)" % (frag_svg_path, len(colored)))

        # Per-colour fragment SVGs: one file per glass colour (the 3D-printer
        # app ignores SVG fill, so each colour must be its own file = its own
        # print layer).  Each file keeps the corner registration marks, so all
        # the colour layers align when imported.
        for stale in glob.glob(os.path.join(out_dir, "color_*.svg")):
            os.remove(stale)  # drop leftovers from a prior (higher-colour) run
        by_color = {}
        for d, col in colored:
            by_color.setdefault(col, []).append((d, col))
        for i, (col, items) in enumerate(sorted(by_color.items()), 1):
            fn = os.path.join(out_dir,
                              "color_%02d_%s.svg" % (i, col.lstrip("#")))
            with open(fn, "w") as f:
                f.write(write_multicolor_svg(items, width_mm, height_mm))
        log("wrote %d per-colour fragment SVGs to %s/"
            % (len(by_color), out_dir))

        if opts.preview:
            prev_path = opts.preview_png or _in_dir("_preview.png")
            write_preview_png(frag_svg, lead_svg, opts.px_mm, prev_path)
            log("wrote %s" % prev_path)

    except AbortError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
