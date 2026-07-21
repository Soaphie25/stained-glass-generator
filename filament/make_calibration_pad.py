#!/usr/bin/env python3
"""Generate a single-filament transmittance calibration pad for stained glass.

The pad is printed ONCE per transparent filament, then laid on a phone/tablet
screen that displays a full-screen solid colour (white / red / green / blue).
Photographing it over each colour lets us separate the filament's per-channel
transmittance from its self-scatter, as a function of THICKNESS.

Outputs (default into ``filament/pad/``):
  * ``calibration_pad.stl``  -- the printable model (import into your slicer).
  * ``layout.json``          -- cell / reference-window / marker positions in mm
                                (pad origin = bottom-left), consumed by the
                                photo analyser.
  * ``calibration_pad_preview.png`` -- top-view sketch to eyeball before printing.

Design (all one transparent filament, so NO coloured fiducials):
  * A continuous transparent BASE PLATE (--base-plate-mm) spans the whole pad and
    makes it ONE RIGID PIECE, so the cells stay fixed relative to the markers no
    matter how you set it on the screen.  (The old design connected cells with
    single-layer bridges and left the markers on a *separate* frame piece -- the
    cells could shift relative to the markers, so sampling positions were lost.)
  * A grid of CELLS built ON TOP of the base plate; each cell's total light path
    is  base_plate + increment  (increment = --step-mm, 2*--step-mm, ... up to
    --base-plate-mm + --max-mm).  We fit transmittance vs. this TOTAL thickness,
    so the base plate is just part of every slab -- it does not bias the fit.
  * Reference windows are real HOLES cut through the base plate, giving true
    bare-screen samples to normalise out per-photo exposure / white-balance /
    brightness gradients.
  * 4 corner registration squares.  DEFAULT: square HOLES through the plate
    (bright when backlit) -> the pad prints in ONE filament with NO colour swap;
    you hand-pick the 4 corners in the analyser ("Pick markers manually").
    --black-markers restores opaque black caps (auto-detectable, needs a swap).

Print notes: slice at a layer height that divides both --base-plate-mm and
--step-mm (e.g. 0.1 mm layers) so thicknesses are exact.  100% infill, transparent
filament, no top/bottom colour changes.
"""
import argparse
import json
import os
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Minimal binary-STL writer (axis-aligned boxes -> triangles)
# --------------------------------------------------------------------------- #
def _box_tris(x0, y0, z0, x1, y1, z1):
    """12 triangles (2 per face) of an axis-aligned box, CCW outward."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),   # bottom 0-3
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]   # top    4-7
    faces = [
        (0, 3, 2), (0, 2, 1),      # bottom (-Z)
        (4, 5, 6), (4, 6, 7),      # top    (+Z)
        (0, 1, 5), (0, 5, 4),      # -Y
        (2, 3, 7), (2, 7, 6),      # +Y
        (1, 2, 6), (1, 6, 5),      # +X
        (3, 0, 4), (3, 4, 7),      # -X
    ]
    return [(v[a], v[b], v[c]) for a, b, c in faces]


def write_stl(path, boxes):
    """boxes: list of (x0,y0,z0,x1,y1,z1) in mm."""
    tris = []
    for b in boxes:
        tris.extend(_box_tris(*b))
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            a, b, c = np.array(a), np.array(b), np.array(c)
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-12 else np.zeros(3)
            f.write(struct.pack("<3f", *n))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(b"\0\0")
    return len(tris)


def _boxes_to_mesh(boxes):
    """boxes -> (vertices, triangles) index lists for a 3MF <mesh>."""
    verts, tris = [], []
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)]
    for (x0, y0, z0, x1, y1, z1) in boxes:
        b = len(verts)
        verts += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                  (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        tris += [(b + a, b + bb, b + c) for a, bb, c in faces]
    return verts, tris


def _mesh_xml(boxes):
    verts, tris = _boxes_to_mesh(boxes)
    vs = "".join('<vertex x="%.4f" y="%.4f" z="%.4f"/>' % v for v in verts)
    ts = "".join('<triangle v1="%d" v2="%d" v3="%d"/>' % t for t in tris)
    return "<mesh><vertices>%s</vertices><triangles>%s</triangles></mesh>" \
        % (vs, ts)


def write_3mf(path, body_boxes, marker_boxes, offset=(10.0, 10.0),
              body_color="#BFD8FFCC", marker_color="#111111FF"):
    """One 3MF: body (transparent) + markers (black) as a single grouped object.

    Standards-compliant core 3MF (millimetre units) with a base-materials
    resource, so it opens directly in Bambu Studio / any slicer as one object
    with two coloured parts, already aligned -- assign the black part to your
    black filament and slice.
    """
    tx, ty = offset
    core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    matns = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    if not marker_boxes:                            # single-filament pad (holes)
        model = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<model unit="millimeter" xml:lang="en-US" xmlns="%s" xmlns:m="%s">\n'
            ' <resources>\n'
            '  <basematerials id="1">'
            '<base name="Transparent" displaycolor="%s"/></basematerials>\n'
            '  <object id="2" type="model" pid="1" pindex="0">%s</object>\n'
            ' </resources>\n'
            ' <build><item objectid="2" '
            'transform="1 0 0 0 1 0 0 0 1 %.3f %.3f 0"/></build>\n'
            '</model>\n'
        ) % (core, matns, body_color, _mesh_xml(body_boxes), tx, ty)
        content_types = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
            'content-types">'
            '<Default Extension="rels" ContentType="application/vnd.'
            'openxmlformats-package.relationships+xml"/>'
            '<Default Extension="model" ContentType="application/vnd.ms-package.'
            '3dmanufacturing-3dmodel+xml"/></Types>'
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            '</Relationships>'
        )
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", rels)
            z.writestr("3D/3dmodel.model", model)
        return
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="%s" xmlns:m="%s">\n'
        ' <resources>\n'
        '  <basematerials id="1">\n'
        '   <base name="Transparent" displaycolor="%s"/>\n'
        '   <base name="Black" displaycolor="%s"/>\n'
        '  </basematerials>\n'
        '  <object id="2" type="model" pid="1" pindex="0">%s</object>\n'
        '  <object id="3" type="model" pid="1" pindex="1">%s</object>\n'
        '  <object id="4" type="model">\n'
        '   <components>\n'
        '    <component objectid="2"/>\n'
        '    <component objectid="3"/>\n'
        '   </components>\n'
        '  </object>\n'
        ' </resources>\n'
        ' <build>\n'
        '  <item objectid="4" transform="1 0 0 0 1 0 0 0 1 %.3f %.3f 0"/>\n'
        ' </build>\n'
        '</model>\n'
    ) % (core, matns, body_color, marker_color,
         _mesh_xml(body_boxes), _mesh_xml(marker_boxes), tx, ty)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
        'content-types">'
        '<Default Extension="rels" ContentType="application/vnd.'
        'openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.'
        '3dmanufacturing-3dmodel+xml"/></Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
        'relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _plate_with_holes(x0, y0, x1, y1, z0, z1, holes):
    """Tile rectangle [x0,x1]x[y0,y1] (height z0..z1) into axis-aligned boxes,
    dropping any sub-tile whose centre lands inside a hole rect.

    Our box mesh writer has no boolean subtraction, so we decompose
    plate-minus-holes on the grid formed by the hole edges.  holes: list of
    (hx0, hy0, hx1, hy1).
    """
    xs = sorted({x0, x1, *(h[0] for h in holes), *(h[2] for h in holes)})
    ys = sorted({y0, y1, *(h[1] for h in holes), *(h[3] for h in holes)})
    xs = [x for x in xs if x0 - 1e-9 <= x <= x1 + 1e-9]
    ys = [y for y in ys if y0 - 1e-9 <= y <= y1 + 1e-9]
    boxes = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(hx0 < cx < hx1 and hy0 < cy < hy1
                   for (hx0, hy0, hx1, hy1) in holes):
                continue
            boxes.append((xs[i], ys[j], z0, xs[i + 1], ys[j + 1], z1))
    return boxes


def build_layout(opts):
    """Compute pad geometry; return (layout dict, body_boxes, marker_boxes)."""
    n = int(round(opts.max_mm / opts.step_mm))
    increments = [round((i + 1) * opts.step_mm, 4) for i in range(n)]
    base = opts.base_plate_mm                   # continuous rigid base plate

    pad_w = opts.screen_w_mm - 2 * opts.margin_mm
    pad_h = opts.screen_h_mm - 2 * opts.margin_mm
    if pad_w <= 20 or pad_h <= 40:
        raise SystemExit("error: screen minus margins is too small for a pad")

    header = opts.header_mm
    edge = opts.edge_mm                         # clear plate border around grid

    cols = opts.cols
    rows = int(np.ceil(n / cols))

    grid_x0 = edge
    grid_x1 = pad_w - edge
    grid_y0 = edge                              # bottom
    grid_y1 = pad_h - edge - header             # leave header strip at the top
    gw = grid_x1 - grid_x0
    gh = grid_y1 - grid_y0
    pitch_x = gw / cols
    pitch_y = gh / rows
    csz = min(pitch_x, pitch_y) * opts.cell_fill
    csz = max(opts.min_cell_mm, csz)

    # cells built ON TOP of the plate; total light path = base + increment
    cells = []
    for i, inc in enumerate(increments):
        r = i // cols          # 0 = top row (thin), increasing downward
        c = i % cols
        cx = grid_x0 + (c + 0.5) * pitch_x
        cy = grid_y1 - (r + 0.5) * pitch_y      # top row highest y
        cells.append({"index": i, "thickness_mm": round(base + inc, 4),
                      "increment_mm": inc,
                      "cx": round(cx, 3), "cy": round(cy, 3),
                      "w": round(csz, 3), "h": round(csz, 3)})

    # reference windows = HOLES through the plate at the grid's gap corners
    wr = min(pitch_x, pitch_y) * 0.14
    windows, holes = [], []
    for r in range(rows + 1):
        for c in range(cols + 1):
            wx = grid_x0 + c * pitch_x
            wy = grid_y1 - r * pitch_y
            if edge + 1 < wx < pad_w - edge - 1 and \
               edge + 1 < wy < pad_h - edge - 1:
                windows.append({"cx": round(wx, 3), "cy": round(wy, 3),
                                "r": round(wr, 3)})
                holes.append((wx - wr, wy - wr, wx + wr, wy + wr))

    # 4 corner registration squares.  DEFAULT: square HOLES through the plate --
    # bright when backlit, so the pad prints in ONE filament (no black swap) and
    # you hand-pick these corners in the analyser.  --black-markers restores the
    # old opaque black caps (auto-detectable, but needs a filament change).
    mk = opts.marker_mm
    inset = opts.marker_inset_mm
    corner_xy = {
        "bottom_left":  (inset, inset),
        "bottom_right": (pad_w - inset - mk, inset),
        "top_right":    (pad_w - inset - mk, pad_h - inset - mk),
        "top_left":     (inset, pad_h - inset - mk),
    }
    reg = {}
    for name, (mx0, my0) in corner_xy.items():
        if not opts.black_markers:
            holes.append((mx0, my0, mx0 + mk, my0 + mk))   # cut through the plate
        reg[name] = {"cx": round(mx0 + mk / 2, 3), "cy": round(my0 + mk / 2, 3),
                     "w": mk, "h": mk}

    boxes = []
    # 1) continuous transparent base plate with the window + corner holes (z 0..base)
    boxes += _plate_with_holes(0, 0, pad_w, pad_h, 0.0, base, holes)
    # 2) cells rising from the plate top (z base..base+increment)
    for cell, inc in zip(cells, increments):
        x0, x1 = cell["cx"] - csz / 2, cell["cx"] + csz / 2
        y0, y1 = cell["cy"] - csz / 2, cell["cy"] + csz / 2
        boxes.append((x0, y0, base, x1, y1, round(base + inc, 4)))

    # --black-markers: opaque BLACK caps on the plate corners (a separate part for
    # your black filament) + an orientation dot.  Auto-detectable, but needs one
    # filament swap.  Default (holes) skips this entirely.
    marker_boxes = []
    if opts.black_markers:
        cap = opts.marker_h_mm
        mz0, mz1 = base, round(base + cap, 4)
        for name, (mx0, my0) in corner_xy.items():
            marker_boxes.append((mx0, my0, mz0, mx0 + mk, my0 + mk, mz1))
        dot = mk * 0.45                          # orientation dot by top-left
        dx0 = inset + mk + opts.marker_gap_mm
        dy0 = pad_h - inset - dot
        marker_boxes.append((dx0, dy0, mz0, dx0 + dot, dy0 + dot, mz1))
        reg["orientation_dot"] = {"cx": round(dx0 + dot / 2, 3),
                                  "cy": round(dy0 + dot / 2, 3),
                                  "w": round(dot, 3), "h": round(dot, 3),
                                  "note": "black dot next to the TOP-LEFT corner"}

    marker_desc = ("black opaque cap on top of the plate (separate markers part)"
                   if opts.black_markers else
                   "square HOLE through the plate (bright when backlit); hand-pick "
                   "these 4 corners in the analyser -- pad is ONE filament")
    layout = {
        "units": "mm",
        "screen_w_mm": opts.screen_w_mm, "screen_h_mm": opts.screen_h_mm,
        "margin_mm": opts.margin_mm,
        "pad_w_mm": round(pad_w, 3), "pad_h_mm": round(pad_h, 3),
        "step_mm": opts.step_mm, "max_mm": opts.max_mm,
        "cols": cols, "rows": rows, "cell_mm": round(csz, 3),
        "base_plate_mm": base,
        "thickness_note": "cells[].thickness_mm is the TOTAL light path "
                          "(base_plate_mm + increment_mm); fit transmittance "
                          "against it.",
        "origin": "bottom-left",
        "pad_corners": [[0, 0], [pad_w, 0], [pad_w, pad_h], [0, pad_h]],
        "register_markers": {
            "style": "black_caps" if opts.black_markers else "holes",
            "color": marker_desc,
            "size_mm": mk,
            "corners": reg,
        },
        "cells": cells,
        "reference_windows": windows,
    }
    return layout, boxes, marker_boxes


# --------------------------------------------------------------------------- #
# Preview (top view)
# --------------------------------------------------------------------------- #
def write_preview(layout, path, px_per_mm=8):
    W = int(layout["pad_w_mm"] * px_per_mm)
    H = int(layout["pad_h_mm"] * px_per_mm)
    img = Image.new("RGB", (W + 2, H + 2), (245, 245, 250))
    d = ImageDraw.Draw(img)

    def X(x):
        return int(x * px_per_mm) + 1

    def Y(y):                                   # flip: pad y-up -> image y-down
        return int((layout["pad_h_mm"] - y) * px_per_mm) + 1

    # continuous transparent base plate (faint blue tint)
    d.rectangle([X(0), Y(layout["pad_h_mm"]), X(layout["pad_w_mm"]), Y(0)],
                fill=(214, 228, 240), outline=(120, 120, 130), width=2)
    for w in layout["reference_windows"]:       # window HOLES = bare screen
        rr = w["r"] * px_per_mm
        d.ellipse([X(w["cx"]) - rr, Y(w["cy"]) - rr,
                   X(w["cx"]) + rr, Y(w["cy"]) + rr],
                  fill=(255, 255, 255), outline=(150, 150, 160))
    base = layout.get("base_plate_mm", 0.0)
    tmin, tmax = base + layout["step_mm"], base + layout["max_mm"]
    for c in layout["cells"]:                   # cells shaded by TOTAL thickness
        f = (c["thickness_mm"] - tmin) / max(tmax - tmin, 1e-6)
        g = int(225 - 170 * f)                  # thicker = darker
        x0, y0 = X(c["cx"] - c["w"] / 2), Y(c["cy"] + c["h"] / 2)
        x1, y1 = X(c["cx"] + c["w"] / 2), Y(c["cy"] - c["h"] / 2)
        d.rectangle([x0, y0, x1, y1], fill=(g, g, g), outline=(90, 90, 90))
        d.text((x0 + 2, y0 + 2), "%.1f" % c["thickness_mm"],
               fill=(255, 80, 80) if g < 130 else (60, 60, 60))
    holes_style = layout["register_markers"].get("style") == "holes"
    reg = layout["register_markers"]["corners"]  # corner registration squares
    for name, m in reg.items():
        if holes_style:                          # bright bare-screen holes
            fill, outline = (255, 255, 255), (150, 150, 160)
        else:
            fill = (15, 15, 15)
            outline = (230, 40, 40) if name == "orientation_dot" else (0, 0, 0)
        d.rectangle([X(m["cx"] - m["w"] / 2), Y(m["cy"] + m["h"] / 2),
                     X(m["cx"] + m["w"] / 2), Y(m["cy"] - m["h"] / 2)],
                    fill=fill, outline=outline, width=2)
    img.save(path)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screen-w-mm", type=float, default=64.0,
                   help="phone/tablet ACTIVE display width in mm (default 64 = "
                        "typical 6.1-inch phone)")
    p.add_argument("--screen-h-mm", type=float, default=138.0,
                   help="active display height in mm (default 138)")
    p.add_argument("--margin-mm", type=float, default=3.0,
                   help="clear border left inside the screen (default 3)")
    p.add_argument("--step-mm", type=float, default=0.1,
                   help="thickness step between cells (default 0.1)")
    p.add_argument("--max-mm", type=float, default=2.0,
                   help="max cell thickness (default 2.0 -> 20 cells)")
    p.add_argument("--cols", type=int, default=4, help="grid columns (default 4)")
    p.add_argument("--cell-fill", type=float, default=0.7,
                   help="cell size as fraction of grid pitch (default 0.7)")
    p.add_argument("--min-cell-mm", type=float, default=6.0,
                   help="minimum cell edge (default 6)")
    p.add_argument("--edge-mm", type=float, default=2.0,
                   help="clear plate border around the cell grid (default 2)")
    p.add_argument("--header-mm", type=float, default=9.0)
    p.add_argument("--base-plate-mm", type=float, default=0.4,
                   help="continuous rigid base-plate thickness; makes the pad one "
                        "piece and is added to every cell's light path (default 0.4)")
    p.add_argument("--black-markers", action="store_true",
                   help="use opaque BLACK corner caps (auto-detectable) instead of "
                        "the default corner HOLES; needs a black filament swap")
    p.add_argument("--marker-mm", type=float, default=6.0,
                   help="corner register-marker/hole size (default 6)")
    p.add_argument("--marker-inset-mm", type=float, default=1.0,
                   help="register-marker inset from pad edge (default 1)")
    p.add_argument("--marker-h-mm", type=float, default=0.4,
                   help="black cap thickness on top of the plate; opaque to block "
                        "backlight, top-layers-only = one filament swap (default 0.4)")
    p.add_argument("--marker-gap-mm", type=float, default=1.5,
                   help="gap from top-left corner to the orientation dot (default 1.5)")
    p.add_argument("--also-stl", action="store_true",
                   help="also write the two separate STL parts alongside the 3MF")
    p.add_argument("--bambu-template", default=None,
                   help="Bambu .3mf export to template from (printer/filament "
                        "settings). Defaults to the bundled P2S template so the "
                        "pad opens as a genuine Bambu project; pass your own "
                        "export for a different machine")
    p.add_argument("--plain", action="store_true",
                   help="write a plain core-3MF instead (Bambu flags it 'not "
                        "from Bambu Lab' but any slicer opens it)")
    p.add_argument("--out-dir", default=None,
                   help="output folder (default: <script dir>/pad)")
    opts = p.parse_args(argv)

    out_dir = opts.out_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "pad")
    os.makedirs(out_dir, exist_ok=True)

    layout, body_boxes, marker_boxes = build_layout(opts)
    tmf = os.path.join(out_dir, "calibration_pad.3mf")
    lay = os.path.join(out_dir, "layout.json")
    prev = os.path.join(out_dir, "calibration_pad_preview.png")
    with open(lay, "w") as f:
        json.dump(layout, f, indent=2)
    write_preview(layout, prev)

    # calibration_pad.3mf IS the Bambu project (bundled template unless --plain
    # or a custom --bambu-template); the plain 3MF is only for non-Bambu slicers.
    holes_mode = not marker_boxes                   # default: single-filament pad
    from bambu_mix3mf import write_bambu_color_mix_3mf, default_template
    template = None if opts.plain else (opts.bambu_template or default_template())
    if template:
        if holes_mode:
            bases = [{"colour": "#BFD8FF"}]
            parts = [{"name": "body", "boxes": body_boxes, "slot": 1}]
            kind = "Bambu project (ONE filament; corners are holes)"
        else:
            bases = [{"colour": "#BFD8FF"}, {"colour": "#111111"}]   # body, black
            parts = [{"name": "body", "boxes": body_boxes, "slot": 1},
                     {"name": "markers", "boxes": marker_boxes, "slot": 2}]
            kind = "Bambu project (slot 1=transparent body, slot 2=black markers)"
        write_bambu_color_mix_3mf(tmf, template, bases, parts)
    else:
        write_3mf(tmf, body_boxes, marker_boxes)
        kind = ("PLAIN 3MF -- Bambu flags 'not from Bambu Lab'; drop --plain for "
                "a real Bambu project")

    stls = ""
    if opts.also_stl:
        stl_body = os.path.join(out_dir, "calibration_pad_body.stl")
        write_stl(stl_body, body_boxes)
        stls = "  %s\n" % stl_body
        if marker_boxes:
            stl_mark = os.path.join(out_dir, "calibration_pad_markers.stl")
            write_stl(stl_mark, marker_boxes)
            stls += "  %s\n" % stl_mark

    base = layout["base_plate_mm"]
    if holes_mode:
        note = ("ONE filament -- the 4 corners are HOLES (bright when backlit). "
                "Slice at %.2f mm layer height. In the analyser, use \"Pick markers "
                "manually\" and click the 4 corner holes (any order)." % opts.step_mm)
    else:
        note = ("Assign the black 'Black' part to your black filament, keep the body "
                "on the transparent filament, slice at %.2f mm layer height. The "
                "black caps are top-layers-only = one filament swap near the end."
                % opts.step_mm)
    sys.stderr.write(
        "pad %.1f x %.1f mm inside a %.0f x %.0f mm screen | %d cells, total "
        "thickness %.1f-%.1f mm (base plate %.2f + %.1f-%.1f) | cell %.1f mm\n"
        "wrote:\n"
        "  %s\n     ^ %s"
        "\n%s  %s\n  %s\n%s\n" % (
            layout["pad_w_mm"], layout["pad_h_mm"],
            opts.screen_w_mm, opts.screen_h_mm, len(layout["cells"]),
            base + opts.step_mm, base + opts.max_mm, base, opts.step_mm,
            opts.max_mm, layout["cell_mm"],
            tmf, kind, stls, lay, prev, note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
