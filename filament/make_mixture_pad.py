#!/usr/bin/env python3
"""Generate a sub-layer MIXTURE calibration pad for two transparent filaments.

Companion to ``make_calibration_pad.py`` (which ramps a SINGLE filament's
thickness).  This pad instead ramps the MIX RATIO between two filaments A and B
at a FIXED total thickness, to calibrate Bambu-Studio-style "sub-layer" colour
mixing.

We do NOT author the sublayers ourselves.  Each pad is a SOLID block tagged with
a mix ratio; Bambu Studio's Color Mixing does the sub-layer slicing.  That keeps
the calibration pad and the production panes on the exact same mixing engine.

Design (per the agreed spec):
  * ``--steps`` + 1 connected PADS (default 11): pad 0 = pure A, pad N = pure B,
    pad i = (N-i)/N of A + i/N of B.  Each pad is a solid block sitting DIRECTLY
    on the screen, so pad 0 and pad N are exactly A and B in the light path.
  * The pads are tied into ONE rigid piece by a thin connective WEB in the gaps
    only (never under a pad, so it can't tint the mix); the web carries the black
    register markers.
  * Reference windows are HOLES through the web (bare-screen normalisation).

Each pad is emitted as its OWN 3MF object so a per-pad mix ratio can be attached.
The authoritative ratios are in ``layout.json``.  Embedding the ratio in Bambu's
own project config is done from a real Bambu color-mix export (TODO: wire once the
sample lands -- see ``_bambu_mix_config``); the geometry + ratios are ready now.

Output (default ``filament/mixpad/``):
  * ``mixture_pad.3mf``   -- 11 pad parts + web + black markers, one assembly.
  * ``layout.json``       -- pad ratios + positions in mm, for the analyser.
  * ``mixture_pad_preview.png``.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

# reuse the box->mesh + plate-with-holes helpers from the single-filament pad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_calibration_pad import _mesh_xml, write_stl, _plate_with_holes  # noqa: E402
from bambu_mix3mf import write_bambu_color_mix_3mf  # noqa: E402


# --------------------------------------------------------------------------- #
# 3MF writer: one coloured OBJECT per part (pads get their own ids so a per-pad
# mix ratio can be attached), all grouped into a single aligned assembly.
# --------------------------------------------------------------------------- #
def write_3mf_objects(path, objects, offset=(10.0, 10.0)):
    """objects: list of {boxes, name, color}.  Each becomes one <object>."""
    tx, ty = offset
    core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    matns = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    bases = "".join('<base name="%s" displaycolor="%s"/>' % (o["name"], o["color"])
                    for o in objects)
    objs, comps, oid = [], [], 2
    for i, o in enumerate(objects):
        objs.append('<object id="%d" type="model" pid="1" pindex="%d">%s</object>'
                    % (oid, i, _mesh_xml(o["boxes"])))
        comps.append('<component objectid="%d"/>' % oid)
        oid += 1
    asm = oid
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="%s" xmlns:m="%s">\n'
        ' <resources>\n'
        '  <basematerials id="1">%s</basematerials>\n  %s\n'
        '  <object id="%d" type="model"><components>%s</components></object>\n'
        ' </resources>\n'
        ' <build><item objectid="%d" transform="1 0 0 0 1 0 0 0 1 %.3f %.3f 0"/>'
        '</build>\n</model>\n'
    ) % (core, matns, bases, "".join(objs), asm, "".join(comps), asm, tx, ty)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
        'content-types"><Default Extension="rels" ContentType="application/vnd.'
        'openxmlformats-package.relationships+xml"/><Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '</Types>'
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


def _pad_parts(layout, pad_objs, web_boxes, mk_boxes):
    """Map the mixture pad to bambu_mix3mf parts: bases 1=A, 2=B, 3=black; each
    interior pad is a mix of slots 1,2; web -> A; markers -> black."""
    steps = layout["steps"]
    ca, cb = np.array([70, 130, 210]), np.array([220, 140, 60])
    parts = []
    for i, o in enumerate(pad_objs):
        if i == 0:
            parts.append({"name": o["name"], "boxes": o["boxes"], "slot": 1})
        elif i == steps:
            parts.append({"name": o["name"], "boxes": o["boxes"], "slot": 2})
        else:
            b = i / steps
            col = "#%02X%02X%02X" % tuple(int(x) for x in ca * (1 - b) + cb * b)
            parts.append({"name": o["name"], "boxes": o["boxes"],
                          "mix": {"components": [1, 2], "ratios": [1 - b, b],
                                  "colour": col}})
    parts.append({"name": "web_A", "boxes": web_boxes, "slot": 1})
    parts.append({"name": "markers", "boxes": mk_boxes, "slot": 3})
    return parts


BASES = [{"colour": "#66A3D2"}, {"colour": "#D2A366"}, {"colour": "#111111"}]


# --------------------------------------------------------------------------- #
def _mix_color(f, ca=(70, 130, 210), cb=(220, 140, 60)):
    """Interpolated display colour A->B for a fraction-B f, as #RRGGBBAA."""
    c = (np.array(ca) * (1 - f) + np.array(cb) * f).round().astype(int)
    return "#%02X%02X%02XCC" % tuple(c)


def build_layout(opts):
    """Compute geometry; return (layout, pad_objects, web_boxes, marker_boxes)."""
    steps = opts.steps
    n_pads = steps + 1
    T = round(opts.thickness_mm, 4)             # every pad is this thick, solid

    pad_w = opts.screen_w_mm - 2 * opts.margin_mm
    pad_h = opts.screen_h_mm - 2 * opts.margin_mm
    if pad_w <= 20 or pad_h <= 40:
        raise SystemExit("error: screen minus margins is too small for a pad")

    edge, header = opts.edge_mm, opts.header_mm
    cols = opts.cols
    rows = int(np.ceil(n_pads / cols))

    gx0, gx1 = edge, pad_w - edge
    gy0, gy1 = edge, pad_h - edge - header
    pitch_x = (gx1 - gx0) / cols
    pitch_y = (gy1 - gy0) / rows
    csz = max(opts.min_cell_mm, min(pitch_x, pitch_y) * opts.cell_fill)

    pad_objs, pads, pad_holes = [], [], []
    for i in range(n_pads):
        r, c = i // cols, i % cols
        cx = gx0 + (c + 0.5) * pitch_x
        cy = gy1 - (r + 0.5) * pitch_y          # pad 0 at top-left
        x0, x1 = cx - csz / 2, cx + csz / 2
        y0, y1 = cy - csz / 2, cy + csz / 2
        fb = i / steps
        pad_objs.append({"boxes": [(x0, y0, 0.0, x1, y1, T)],
                         "name": "pad%02d_%dB" % (i, round(100 * fb)),
                         "color": _mix_color(fb)})
        pad_holes.append((x0, y0, x1, y1))
        pads.append({"index": i, "ratio_b": round(fb, 4),
                     "pct_a": round(100 * (1 - fb), 2), "pct_b": round(100 * fb, 2),
                     "cx": round(cx, 3), "cy": round(cy, 3),
                     "w": round(csz, 3), "h": round(csz, 3)})

    # reference windows = holes at the gap corners of the pad grid
    wr = min(pitch_x, pitch_y) * 0.13
    windows, win_holes = [], []
    for r in range(rows + 1):
        for c in range(cols + 1):
            wx, wy = gx0 + c * pitch_x, gy1 - r * pitch_y
            if edge + 1 < wx < pad_w - edge - 1 and edge + 1 < wy < pad_h - edge - 1:
                windows.append({"cx": round(wx, 3), "cy": round(wy, 3),
                                "r": round(wr, 3)})
                win_holes.append((wx - wr, wy - wr, wx + wr, wy + wr))

    # connective WEB (filament A, unsampled): whole pad minus pad footprints and
    # window holes -> one rigid piece without tinting any pad's light path.
    web = opts.web_mm
    web_boxes = _plate_with_holes(0, 0, pad_w, pad_h, 0.0, web,
                                  pad_holes + win_holes)

    # BLACK register caps on top of the web corners (+ orientation dot)
    mk, inset, cap = opts.marker_mm, opts.marker_inset_mm, opts.marker_h_mm
    mz0, mz1 = web, round(web + cap, 4)
    corners = {
        "bottom_left":  (inset, inset),
        "bottom_right": (pad_w - inset - mk, inset),
        "top_right":    (pad_w - inset - mk, pad_h - inset - mk),
        "top_left":     (inset, pad_h - inset - mk),
    }
    mk_boxes, reg = [], {}
    for name, (mx0, my0) in corners.items():
        mk_boxes.append((mx0, my0, mz0, mx0 + mk, my0 + mk, mz1))
        reg[name] = {"cx": round(mx0 + mk / 2, 3), "cy": round(my0 + mk / 2, 3),
                     "w": mk, "h": mk}
    dot = mk * 0.45
    dx0, dy0 = inset + mk + opts.marker_gap_mm, pad_h - inset - mk * 0.45
    mk_boxes.append((dx0, dy0, mz0, dx0 + dot, dy0 + dot, mz1))
    reg["orientation_dot"] = {"cx": round(dx0 + dot / 2, 3),
                              "cy": round(dy0 + dot / 2, 3),
                              "w": round(dot, 3), "h": round(dot, 3),
                              "note": "black dot next to the TOP-LEFT corner"}

    layout = {
        "units": "mm", "mode": "sub-layer-mixture",
        "filaments": ["A", "B"], "mixing": "Bambu Studio Color Mixing (solid "
        "pads tagged with a ratio; the slicer makes the sublayers)",
        "screen_w_mm": opts.screen_w_mm, "screen_h_mm": opts.screen_h_mm,
        "margin_mm": opts.margin_mm,
        "pad_w_mm": round(pad_w, 3), "pad_h_mm": round(pad_h, 3),
        "steps": steps, "n_pads": n_pads, "total_thickness_mm": T,
        "web_mm": web, "cols": cols, "rows": rows, "cell_mm": round(csz, 3),
        "origin": "bottom-left",
        "pad_corners": [[0, 0], [pad_w, 0], [pad_w, pad_h], [0, pad_h]],
        "register_markers": {
            "color": "black opaque cap on top of the web (separate part)",
            "cap_mm": cap, "size_mm": mk, "z_mm": [mz0, mz1], "corners": reg},
        "pads": pads,
        "reference_windows": windows,
    }
    return layout, pad_objs, web_boxes, mk_boxes


# --------------------------------------------------------------------------- #
def write_preview(layout, path, px_per_mm=8):
    W = int(layout["pad_w_mm"] * px_per_mm)
    H = int(layout["pad_h_mm"] * px_per_mm)
    img = Image.new("RGB", (W + 2, H + 2), (245, 245, 250))
    d = ImageDraw.Draw(img)

    def X(x):
        return int(x * px_per_mm) + 1

    def Y(y):
        return int((layout["pad_h_mm"] - y) * px_per_mm) + 1

    d.rectangle([X(0), Y(layout["pad_h_mm"]), X(layout["pad_w_mm"]), Y(0)],
                fill=(224, 232, 240), outline=(120, 120, 130), width=2)
    for w in layout["reference_windows"]:
        rr = w["r"] * px_per_mm
        d.ellipse([X(w["cx"]) - rr, Y(w["cy"]) - rr,
                   X(w["cx"]) + rr, Y(w["cy"]) + rr],
                  fill=(255, 255, 255), outline=(150, 150, 160))
    ca, cb = np.array([70, 130, 210]), np.array([220, 140, 60])
    for p in layout["pads"]:
        f = p["pct_b"] / 100.0
        col = tuple((ca * (1 - f) + cb * f).round().astype(int))
        x0, y0 = X(p["cx"] - p["w"] / 2), Y(p["cy"] + p["h"] / 2)
        x1, y1 = X(p["cx"] + p["w"] / 2), Y(p["cy"] - p["h"] / 2)
        d.rectangle([x0, y0, x1, y1], fill=col, outline=(70, 70, 80))
        d.text((x0 + 3, y0 + 3), "%d%%B" % p["pct_b"],
               fill=(255, 255, 255) if 20 < p["pct_b"] < 90 else (30, 30, 30))
    for name, m in layout["register_markers"]["corners"].items():
        oc = (230, 40, 40) if name == "orientation_dot" else (0, 0, 0)
        d.rectangle([X(m["cx"] - m["w"] / 2), Y(m["cy"] + m["h"] / 2),
                     X(m["cx"] + m["w"] / 2), Y(m["cy"] - m["h"] / 2)],
                    fill=(15, 15, 15), outline=oc, width=2)
    img.save(path)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screen-w-mm", type=float, default=64.0)
    p.add_argument("--screen-h-mm", type=float, default=138.0)
    p.add_argument("--margin-mm", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=10,
                   help="ratio steps; makes steps+1 pads (default 10 -> 11 pads, "
                        "0/10/../100%% B)")
    p.add_argument("--thickness-mm", type=float, default=1.0,
                   help="solid pad thickness; Bambu mixes sublayers within it "
                        "(default 1.0)")
    p.add_argument("--cols", type=int, default=3, help="pad grid columns (default 3)")
    p.add_argument("--cell-fill", type=float, default=0.72)
    p.add_argument("--min-cell-mm", type=float, default=8.0)
    p.add_argument("--edge-mm", type=float, default=2.0)
    p.add_argument("--header-mm", type=float, default=9.0)
    p.add_argument("--web-mm", type=float, default=0.3,
                   help="connective web thickness in the gaps (default 0.3)")
    p.add_argument("--marker-mm", type=float, default=6.0)
    p.add_argument("--marker-inset-mm", type=float, default=1.0)
    p.add_argument("--marker-h-mm", type=float, default=0.4)
    p.add_argument("--marker-gap-mm", type=float, default=1.5)
    p.add_argument("--also-stl", action="store_true")
    p.add_argument("--bambu-template",
                   help="a real Bambu color-mix .3mf export; when given, also "
                        "write mixture_pad_bambu.3mf with all mix ratios pre-set")
    p.add_argument("--out-dir", default=None)
    opts = p.parse_args(argv)

    out_dir = opts.out_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "mixpad")
    os.makedirs(out_dir, exist_ok=True)

    layout, pad_objs, web_boxes, mk_boxes = build_layout(opts)
    objects = pad_objs + [
        {"boxes": web_boxes, "name": "web_A", "color": "#CFE6FFCC"},
        {"boxes": mk_boxes, "name": "Black", "color": "#111111FF"},
    ]
    tmf = os.path.join(out_dir, "mixture_pad.3mf")
    lay = os.path.join(out_dir, "layout.json")
    prev = os.path.join(out_dir, "mixture_pad_preview.png")
    write_3mf_objects(tmf, objects)
    with open(lay, "w") as f:
        json.dump(layout, f, indent=2)
    write_preview(layout, prev)

    stls = ""
    if opts.also_stl:
        allb = [b for o in pad_objs for b in o["boxes"]] + web_boxes
        write_stl(os.path.join(out_dir, "mixture_pad_body.stl"), allb)
        write_stl(os.path.join(out_dir, "mixture_pad_markers.stl"), mk_boxes)
        stls = "  (+ body/markers STL)\n"

    if opts.bambu_template:
        parts = _pad_parts(layout, pad_objs, web_boxes, mk_boxes)
        bpath = os.path.join(out_dir, "mixture_pad_bambu.3mf")
        info = write_bambu_color_mix_3mf(bpath, opts.bambu_template, BASES, parts)
        n_slots = info["n_slots"]
        stls += ("  %s   <- Bambu 3MF, %d filament slots (1=A, 2=B, 3=black, "
                 "4-%d=mixes), ratios pre-set\n" % (bpath, n_slots, n_slots))

    sys.stderr.write(
        "mixture pad %.1f x %.1f mm | %d solid pads 0..100%%B in %d%% steps | "
        "%.2f mm thick | pad %.1f mm\n"
        "wrote:\n  %s   <- each pad is its own part; assign filament A/B mix "
        "ratios per pad in Bambu (0%%..100%% B), 'web_A'->A, 'Black'->black\n"
        "%s  %s\n  %s\n" % (
            layout["pad_w_mm"], layout["pad_h_mm"], layout["n_pads"],
            int(round(100 / opts.steps)), layout["total_thickness_mm"],
            layout["cell_mm"], tmf, stls, lay, prev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
