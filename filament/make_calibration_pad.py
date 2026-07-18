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
  * A grid of CELLS, each a square tower whose height is the calibrated
    thickness (0.1, 0.2, ... up to --max-mm, in --step-mm steps).
  * The gaps between cells + the border are BARE-SCREEN reference windows
    (needed to normalise out per-photo exposure / white-balance / brightness).
  * A thin base frame + inter-cell bridges hold it together as one piece.
  * A solid ORIENTATION MARKER tower in a top header strip disambiguates which
    corner is which (the pad outline gives the 4 corners for the warp).

Print notes: slice at a layer height that divides --step-mm (e.g. 0.1 mm steps
-> 0.10 or 0.05 mm layers) so the thicknesses are exact.  100% infill, transparent
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
def build_layout(opts):
    """Compute pad geometry; return (layout dict, boxes list)."""
    n = int(round(opts.max_mm / opts.step_mm))
    thicks = [round((i + 1) * opts.step_mm, 4) for i in range(n)]

    pad_w = opts.screen_w_mm - 2 * opts.margin_mm
    pad_h = opts.screen_h_mm - 2 * opts.margin_mm
    if pad_w <= 20 or pad_h <= 40:
        raise SystemExit("error: screen minus margins is too small for a pad")

    frame = opts.frame_mm
    header = opts.header_mm
    base = opts.base_mm

    # choose grid columns so cells are as large as possible but reasonable
    cols = opts.cols
    rows = int(np.ceil(n / cols))

    grid_x0 = frame
    grid_x1 = pad_w - frame
    grid_y0 = frame                     # bottom
    grid_y1 = pad_h - frame - header    # leave header strip at the very top
    gw = grid_x1 - grid_x0
    gh = grid_y1 - grid_y0
    pitch_x = gw / cols
    pitch_y = gh / rows
    cell = min(pitch_x, pitch_y) * opts.cell_fill
    cell = max(opts.min_cell_mm, cell)

    boxes = []

    # base frame (perimeter, thickness = base) for connectivity + outline
    boxes += [
        (0, 0, 0, pad_w, frame, base),                      # bottom edge
        (0, pad_h - frame, 0, pad_w, pad_h, base),          # top edge
        (0, 0, 0, frame, pad_h, base),                      # left edge
        (pad_w - frame, 0, 0, pad_w, pad_h, base),          # right edge
    ]

    cells = []
    for i, t in enumerate(thicks):
        r = i // cols          # 0 = top row (thin), increasing downward
        c = i % cols
        cx = grid_x0 + (c + 0.5) * pitch_x
        cy = grid_y1 - (r + 0.5) * pitch_y      # top row highest y
        x0, x1 = cx - cell / 2, cx + cell / 2
        y0, y1 = cy - cell / 2, cy + cell / 2
        boxes.append((x0, y0, 0.0, x1, y1, max(t, base)))
        cells.append({"index": i, "thickness_mm": t,
                      "cx": round(cx, 3), "cy": round(cy, 3),
                      "w": round(cell, 3), "h": round(cell, 3)})

    # inter-cell bridges (thin, at base height) so nothing is an island
    bw = opts.bridge_mm
    for i in range(len(cells)):
        r, c = i // cols, i % cols
        cxi, cyi = cells[i]["cx"], cells[i]["cy"]
        if c + 1 < cols and (i + 1) < len(cells):        # bridge to right
            cxj = cells[i + 1]["cx"]
            boxes.append((cxi + cell / 2, cyi - bw / 2, 0.0,
                          cxj - cell / 2, cyi + bw / 2, base))
        if (i + cols) < len(cells):                      # bridge downward
            cyj = cells[i + cols]["cy"]
            boxes.append((cxi - bw / 2, cyj + cell / 2, 0.0,
                          cxi + bw / 2, cyi - cell / 2, base))

    # reference windows = gap centres (bare screen sampled there by the analyser)
    windows = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            wx = grid_x0 + c * pitch_x
            wy = grid_y1 - r * pitch_y
            if frame + 1 < wx < pad_w - frame - 1 and \
               frame + 1 < wy < pad_h - frame - 1:
                windows.append({"cx": round(wx, 3), "cy": round(wy, 3),
                                "r": round(min(pitch_x, pitch_y) * 0.12, 3)})

    # BLACK opaque register markers at the 4 corners -> a SEPARATE STL part the
    # user assigns to black filament (AMS / 2-material).  Opaque black reads on
    # ANY screen colour AND any screen size (a small pad on a large iPad), unlike
    # the faint tinted pad outline.  An extra dot inside the top-left corner
    # fixes orientation (which corner is which).
    mk = opts.marker_mm
    inset = opts.marker_inset_mm
    mh = opts.marker_h_mm
    corners = {
        "bottom_left":  (inset, inset),
        "bottom_right": (pad_w - inset - mk, inset),
        "top_right":    (pad_w - inset - mk, pad_h - inset - mk),
        "top_left":     (inset, pad_h - inset - mk),
    }
    marker_boxes = []
    reg = {}
    for name, (mx0, my0) in corners.items():
        marker_boxes.append((mx0, my0, 0.0, mx0 + mk, my0 + mk, mh))
        reg[name] = {"cx": round(mx0 + mk / 2, 3), "cy": round(my0 + mk / 2, 3),
                     "w": mk, "h": mk}
    dot = mk * 0.45                              # orientation dot by top-left
    dx0 = inset + mk + opts.marker_gap_mm
    dy0 = pad_h - inset - dot
    marker_boxes.append((dx0, dy0, 0.0, dx0 + dot, dy0 + dot, mh))
    reg["orientation_dot"] = {"cx": round(dx0 + dot / 2, 3),
                              "cy": round(dy0 + dot / 2, 3),
                              "w": round(dot, 3), "h": round(dot, 3),
                              "note": "black dot next to the TOP-LEFT corner"}

    layout = {
        "units": "mm",
        "screen_w_mm": opts.screen_w_mm, "screen_h_mm": opts.screen_h_mm,
        "margin_mm": opts.margin_mm,
        "pad_w_mm": round(pad_w, 3), "pad_h_mm": round(pad_h, 3),
        "step_mm": opts.step_mm, "max_mm": opts.max_mm,
        "cols": cols, "rows": rows, "cell_mm": round(cell, 3),
        "base_mm": base,
        "origin": "bottom-left",
        "pad_corners": [[0, 0], [pad_w, 0], [pad_w, pad_h], [0, pad_h]],
        "register_markers": {
            "color": "black opaque (print as the separate markers STL)",
            "height_mm": mh, "size_mm": mk,
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

    d.rectangle([X(0), Y(layout["pad_h_mm"]), X(layout["pad_w_mm"]), Y(0)],
                outline=(120, 120, 130), width=2)
    for w in layout["reference_windows"]:       # bare-screen refs = light dots
        rr = w["r"] * px_per_mm
        d.ellipse([X(w["cx"]) - rr, Y(w["cy"]) - rr,
                   X(w["cx"]) + rr, Y(w["cy"]) + rr],
                  fill=(255, 255, 255), outline=(180, 180, 190))
    mx = layout["max_mm"]
    for c in layout["cells"]:                   # cells shaded by thickness
        g = int(235 - 175 * (c["thickness_mm"] / mx))   # thicker = darker
        x0, y0 = X(c["cx"] - c["w"] / 2), Y(c["cy"] + c["h"] / 2)
        x1, y1 = X(c["cx"] + c["w"] / 2), Y(c["cy"] - c["h"] / 2)
        d.rectangle([x0, y0, x1, y1], fill=(g, g, g), outline=(90, 90, 90))
        d.text((x0 + 2, y0 + 2), "%.1f" % c["thickness_mm"],
               fill=(255, 80, 80) if g < 130 else (60, 60, 60))
    reg = layout["register_markers"]["corners"]  # black corners + orient dot
    for name, m in reg.items():
        outline = (230, 40, 40) if name == "orientation_dot" else (0, 0, 0)
        d.rectangle([X(m["cx"] - m["w"] / 2), Y(m["cy"] + m["h"] / 2),
                     X(m["cx"] + m["w"] / 2), Y(m["cy"] - m["h"] / 2)],
                    fill=(15, 15, 15), outline=outline, width=2)
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
    p.add_argument("--frame-mm", type=float, default=2.0)
    p.add_argument("--header-mm", type=float, default=9.0)
    p.add_argument("--base-mm", type=float, default=0.1,
                   help="connective base/frame thickness (default 0.1 = 1 layer)")
    p.add_argument("--bridge-mm", type=float, default=1.5)
    p.add_argument("--marker-mm", type=float, default=6.0,
                   help="black corner register-marker size (default 6)")
    p.add_argument("--marker-inset-mm", type=float, default=1.0,
                   help="register-marker inset from pad edge (default 1)")
    p.add_argument("--marker-h-mm", type=float, default=1.0,
                   help="black marker height; opaque to block backlight (default 1)")
    p.add_argument("--marker-gap-mm", type=float, default=1.5,
                   help="gap from top-left corner to the orientation dot (default 1.5)")
    p.add_argument("--also-stl", action="store_true",
                   help="also write the two separate STL parts alongside the 3MF")
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
    write_3mf(tmf, body_boxes, marker_boxes)
    with open(lay, "w") as f:
        json.dump(layout, f, indent=2)
    write_preview(layout, prev)

    stls = ""
    if opts.also_stl:
        stl_body = os.path.join(out_dir, "calibration_pad_body.stl")
        stl_mark = os.path.join(out_dir, "calibration_pad_markers.stl")
        write_stl(stl_body, body_boxes)
        write_stl(stl_mark, marker_boxes)
        stls = "  %s\n  %s\n" % (stl_body, stl_mark)

    sys.stderr.write(
        "pad %.1f x %.1f mm inside a %.0f x %.0f mm screen | %d cells "
        "%.1f-%.1f mm | cell %.1f mm\n"
        "wrote:\n"
        "  %s   <- open directly in the slicer (body=transparent, "
        "corners=black, already grouped & aligned)\n%s  %s\n  %s\n"
        "Assign the black 'Black' part to your black filament, keep the body on "
        "the transparent filament, slice at %.2f mm layer height.\n" % (
            layout["pad_w_mm"], layout["pad_h_mm"],
            opts.screen_w_mm, opts.screen_h_mm, len(layout["cells"]),
            opts.step_mm, opts.max_mm, layout["cell_mm"],
            tmf, stls, lay, prev, opts.step_mm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
