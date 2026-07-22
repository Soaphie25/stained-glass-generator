#!/usr/bin/env python3
"""Generate a sub-layer MIXTURE calibration STRIP for two transparent filaments.

Companion to ``make_calibration_pad.py`` (which ramps a SINGLE filament's
thickness).  This strip instead ramps the MIX RATIO between two filaments A and B
at a FIXED total thickness, to calibrate Bambu-Studio-style "sub-layer" colour
mixing.

Design (the redesigned continuous strip):
  * ONE ROW of contiguous cells, edge to edge, forming a single continuous A->B
    gradient bar -- easy to judge by eye and trivial for CV to segment.
  * ``--ramp-step`` = the %B increment per cell.  Cell 0 = ``--start``%B,
    cell N = ``--end``%B; with the defaults (start 0, end 100, step 10) that is
    11 cells 0/10/../100 %B.  Pick a sub-range (e.g. 20..60) to zoom a region.
  * Each cell is its OWN Bambu part tagged with its mix ratio (Bambu Studio's
    Color Mixing does the sub-layer slicing -- same engine as the panes).  The
    cells share faces, so the strip prints as one rigid piece.
  * total width = ``--cell-w`` * n_cells, total height = ``--cell-h``.
  * NO black markers and NO reference windows: the strip is a clean rectangle;
    hand-pick its 4 physical corners in the analyser (always-on manual picking).
    The analyser normalises exposure from the two pure ends (which equal the
    ironed single-cals) and auto-detects which end is A vs B.

Output (default ``filament/mixpad/``):
  * ``mixture_pad.3mf``   -- the gradient strip (Bambu project, ratios pre-set).
  * ``layout.json``       -- cell ratios + positions in mm, for the analyser.
  * ``mixture_pad_preview.png``.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

# reuse the box->mesh helpers from the single-filament pad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_calibration_pad import _mesh_xml, write_stl  # noqa: E402
from bambu_mix3mf import write_bambu_color_mix_3mf, default_template  # noqa: E402


# --------------------------------------------------------------------------- #
# 3MF writer: one coloured OBJECT per part (cells get their own ids so a per-cell
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


def _shift_boxes(boxes, dx=0.0, dy=0.0):
    """Translate a box list by (dx, dy) -- dx lays batched strips across the plate
    (unused here) and dy stacks them."""
    return [(x0 + dx, y0 + dy, z0, x1 + dx, y1 + dy, z1)
            for (x0, y0, z0, x1, y1, z1) in boxes]


# --------------------------------------------------------------------------- #
def _mix_color(f, ca=(70, 130, 210), cb=(220, 140, 60)):
    """Interpolated display colour A->B for a fraction-B f, as #RRGGBB."""
    c = (np.array(ca) * (1 - f) + np.array(cb) * f).round().astype(int)
    return "#%02X%02X%02X" % tuple(c)


def build_layout(opts):
    """Compute the continuous-strip geometry.

    Returns (layout, fracs) where ``fracs`` is the list of fraction-B per cell.
    The 3MF geometry is assembled in ``main`` (so it can batch/stack strips)."""
    start, end, step = float(opts.start), float(opts.end), float(opts.ramp_step)
    if not (0 <= start < end <= 100):
        raise SystemExit("error: need 0 <= --start < --end <= 100")
    if step <= 0:
        raise SystemExit("error: --ramp-step must be > 0")
    span = end - start
    n_steps = max(1, int(round(span / step)))
    n_cells = n_steps + 1
    actual_step = span / n_steps                 # exact endpoints even if span%step
    fracs = [(start + i * actual_step) / 100.0 for i in range(n_cells)]

    cw, ch = round(opts.cell_w, 4), round(opts.cell_h, 4)
    T = round(opts.depth_mm, 4)
    pad_w = round(cw * n_cells, 4)
    pad_h = ch

    cells = []
    for i, fr in enumerate(fracs):
        x0 = i * cw
        cells.append({"index": i, "ratio_b": round(fr, 4),
                      "pct_a": round(100 * (1 - fr), 2),
                      "pct_b": round(100 * fr, 2),
                      "cx": round(x0 + cw / 2, 3), "cy": round(ch / 2, 3),
                      "w": round(cw, 3), "h": round(ch, 3)})

    layout = {
        "units": "mm", "mode": "sub-layer-mixture-strip",
        "filaments": ["A", "B"],
        "mixing": "Bambu Studio Color Mixing (contiguous strip; each cell is a "
                  "solid part tagged with a ratio; the slicer makes the sublayers)",
        "pad_w_mm": pad_w, "pad_h_mm": pad_h,
        "total_thickness_mm": T, "cell_w_mm": cw, "cell_h_mm": ch,
        "start_pct": start, "end_pct": end, "ramp_step_pct": round(actual_step, 3),
        "n_cells": n_cells, "n_steps": n_steps,
        "origin": "bottom-left",
        "pad_corners": [[0, 0], [pad_w, 0], [pad_w, pad_h], [0, pad_h]],
        # The strip is a clean rectangle: pick its 4 physical corners by hand.  No
        # black caps, no bright holes -- the corners double as the registration.
        "register_markers": {
            "style": "holes",
            "color": "no printed markers -- hand-pick the 4 physical corners of "
                     "the strip (bottom-left, bottom-right, top-right, top-left)",
            "corners": {
                "bottom_left":  {"cx": 0.0,   "cy": 0.0,   "w": 0.0, "h": 0.0},
                "bottom_right": {"cx": pad_w, "cy": 0.0,   "w": 0.0, "h": 0.0},
                "top_right":    {"cx": pad_w, "cy": pad_h, "w": 0.0, "h": 0.0},
                "top_left":     {"cx": 0.0,   "cy": pad_h, "w": 0.0, "h": 0.0}}},
        "reference_windows": [],                 # none: exposure is fixed from the ends
        "pads": cells,
    }
    return layout, fracs


# --------------------------------------------------------------------------- #
def write_preview(layout, path, px_per_mm=8):
    W = int(layout["pad_w_mm"] * px_per_mm)
    H = int(layout["pad_h_mm"] * px_per_mm)
    img = Image.new("RGB", (W + 2, H + 40), (245, 245, 250))
    d = ImageDraw.Draw(img)

    def X(x):
        return int(x * px_per_mm) + 1

    def Y(y):
        return int((layout["pad_h_mm"] - y) * px_per_mm) + 1

    for p in layout["pads"]:
        f = p["pct_b"] / 100.0
        col = tuple(int(_mix_color(f)[k:k + 2], 16) for k in (1, 3, 5))
        x0, y0 = X(p["cx"] - p["w"] / 2), Y(p["cy"] + p["h"] / 2)
        x1, y1 = X(p["cx"] + p["w"] / 2), Y(p["cy"] - p["h"] / 2)
        d.rectangle([x0, y0, x1, y1], fill=col, outline=(70, 70, 80))
        d.text((x0 + 3, y1 + 3), "%d" % p["pct_b"],
               fill=(255, 255, 255) if 20 < p["pct_b"] < 90 else (30, 30, 30))
    # mark the 4 corners the analyser expects hand-picked
    for m in layout["register_markers"]["corners"].values():
        r = 4
        d.ellipse([X(m["cx"]) - r, Y(m["cy"]) - r, X(m["cx"]) + r, Y(m["cy"]) + r],
                  outline=(200, 40, 40), width=2)
    d.text((4, H + 6), "A --> B  (%d cells, %g%% step, %g..%g%%B)  pick the 4 red "
           "corners" % (layout["n_cells"], layout["ramp_step_pct"],
                        layout["start_pct"], layout["end_pct"]),
           fill=(60, 60, 70))
    img.save(path)


# --------------------------------------------------------------------------- #
def _strip_parts(fracs, layout, slot_a, slot_b, dy=0.0, tag=""):
    """Map one gradient strip to bambu_mix3mf parts using slots ``slot_a``/``slot_b``
    for A/B (and their mixes), stacked by ``dy``."""
    cw, ch = layout["cell_w_mm"], layout["cell_h_mm"]
    T = layout["total_thickness_mm"]
    ca, cb = np.array([70, 130, 210]), np.array([220, 140, 60])
    parts = []
    for i, fr in enumerate(fracs):
        box = [(i * cw, dy, 0.0, i * cw + cw, dy + ch, T)]
        nm = "c%02d_%dB%s" % (i, round(100 * fr), tag)
        if fr <= 1e-9:
            parts.append({"name": nm, "boxes": box, "slot": slot_a})
        elif fr >= 1 - 1e-9:
            parts.append({"name": nm, "boxes": box, "slot": slot_b})
        else:
            col = "#%02X%02X%02X" % tuple(int(x) for x in ca * (1 - fr) + cb * fr)
            parts.append({"name": nm, "boxes": box,
                          "mix": {"components": [slot_a, slot_b],
                                  "ratios": [1 - fr, fr], "colour": col}})
    return parts


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-mm", type=float, default=1.0,
                   help="solid strip thickness / light path; Bambu mixes sublayers "
                        "within it (default 1.0)")
    p.add_argument("--cell-w", type=float, default=10.0,
                   help="width of each gradient cell mm (default 10)")
    p.add_argument("--cell-h", type=float, default=20.0,
                   help="height of the strip mm (default 20)")
    p.add_argument("--ramp-step", type=float, default=10.0,
                   help="%%B increment per cell (default 10 -> 0/10/../100%%B)")
    p.add_argument("--start", type=float, default=0.0,
                   help="%%B of the first cell (default 0 = pure A)")
    p.add_argument("--end", type=float, default=100.0,
                   help="%%B of the last cell (default 100 = pure B)")
    p.add_argument("--count", type=int, default=1,
                   help="BATCH: stack this many strips (1-3) on one plate, each on "
                        "its own A/B filament pair -- print several pairs in one job")
    p.add_argument("--gap-mm", type=float, default=6.0,
                   help="gap between stacked strips (default 6)")
    p.add_argument("--plate-mm", type=float, default=250.0,
                   help="printer plate height for the batch fit check (default 250)")
    p.add_argument("--colors", default=None,
                   help="comma-separated #hex per slot A1,B1,A2,B2,... (default "
                        "a preset palette)")
    p.add_argument("--also-stl", action="store_true")
    p.add_argument("--bambu-template", default=None,
                   help="Bambu .3mf export to template from; defaults to the "
                        "bundled P2S template so the ratios open pre-set")
    p.add_argument("--plain", action="store_true",
                   help="write a plain 3MF (no embedded mix ratios; Bambu flags "
                        "it 'not from Bambu Lab')")
    p.add_argument("--out-dir", default=None)
    opts = p.parse_args(argv)

    out_dir = opts.out_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "mixpad")
    os.makedirs(out_dir, exist_ok=True)

    layout, fracs = build_layout(opts)
    tmf = os.path.join(out_dir, "mixture_pad.3mf")
    lay = os.path.join(out_dir, "layout.json")
    prev = os.path.join(out_dir, "mixture_pad_preview.png")
    with open(lay, "w") as f:
        json.dump(layout, f, indent=2)
    write_preview(layout, prev)

    # BATCH: N strips stacked in Y (each on its own A/B pair)
    pitch = layout["pad_h_mm"] + opts.gap_mm
    n = max(1, min(3, opts.count))
    while n > 1 and (n * layout["pad_h_mm"] + (n - 1) * opts.gap_mm) > opts.plate_mm:
        n -= 1
    if n < min(3, max(1, opts.count)):
        sys.stderr.write("note: %d strips don't fit %.0f mm tall; using %d\n"
                         % (opts.count, opts.plate_mm, n))
    palette = ["#66A3D2", "#D2A366", "#66C28A", "#C266A3", "#A3C266", "#8A66C2"]
    cols = ([c.strip() for c in opts.colors.split(",")] if opts.colors else palette)

    template = None if opts.plain else (opts.bambu_template or default_template())
    if template:
        parts, bases = [], []
        for j in range(n):
            sa, sb = 2 * j + 1, 2 * j + 2
            tag = "" if n == 1 else "_s%d" % (j + 1)
            parts += _strip_parts(fracs, layout, sa, sb, dy=j * pitch, tag=tag)
            bases += [{"colour": cols[(2 * j) % len(cols)]},
                      {"colour": cols[(2 * j + 1) % len(cols)]}]
        info = write_bambu_color_mix_3mf(tmf, template, bases, parts)
        kind = ("Bambu project, %d strip%s (%d slot%s), ratios pre-set"
                % (n, "s" if n > 1 else "", info["n_slots"],
                   "s" if info["n_slots"] > 1 else ""))
    else:
        objects = []
        for j in range(n):
            for i, fr in enumerate(fracs):
                cw, ch, T = layout["cell_w_mm"], layout["cell_h_mm"], \
                    layout["total_thickness_mm"]
                objects.append({"boxes": [(i * cw, j * pitch, 0.0,
                                           i * cw + cw, j * pitch + ch, T)],
                                "name": "c%02d_%dB" % (i, round(100 * fr)),
                                "color": _mix_color(fr) + "FF"})
        write_3mf_objects(tmf, objects)
        kind = ("PLAIN 3MF (%d strip%s), no embedded mixes -- Bambu flags 'not from "
                "Bambu Lab'; drop --plain for a real Bambu project"
                % (n, "s" if n > 1 else ""))

    stls = ""
    if opts.also_stl:
        allb = [(i * layout["cell_w_mm"], 0.0, 0.0,
                 i * layout["cell_w_mm"] + layout["cell_w_mm"], layout["cell_h_mm"],
                 layout["total_thickness_mm"]) for i in range(len(fracs))]
        write_stl(os.path.join(out_dir, "mixture_pad_body.stl"), allb)
        stls = "  (+ body STL)\n"

    sys.stderr.write(
        "mixture strip %.1f x %.1f mm | %d cells %g..%g%%B in %g%% steps | "
        "%.2f mm thick | cell %g x %g mm\n"
        "wrote:\n  %s\n     ^ %s\n%s  %s\n  %s\n" % (
            layout["pad_w_mm"], layout["pad_h_mm"], layout["n_cells"],
            layout["start_pct"], layout["end_pct"], layout["ramp_step_pct"],
            layout["total_thickness_mm"], layout["cell_w_mm"], layout["cell_h_mm"],
            tmf, kind, stls, lay, prev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
