#!/usr/bin/env python3
"""Generate a full-layer STACK verification pad for two transparent filaments.

To check the stacking prediction (T = T0*exp(-sum a_i*d_i)) on a real print: each
cell is a physical stack of two filament layers at chosen thicknesses (e.g. 0.2 mm
red under 0.8 mm green).  Photograph it over the white screen and measure each
cell's colour (``analyze_calibration.py measure``), then compare to
``solve_recipe.py predict``.

Same rigid-piece construction as the mixture pad: cells sit directly on the screen
(so the light path is exactly the stack), tied together by a gap-web carrying the
black register markers, with bare-screen window holes for normalisation.

Default stacks: red=0.2,green=0.8 / red=0.4,green=0.6 / red=0.8,green=0.2 (mm).

Output (default ``filament/stackpad/``): ``stack_pad.3mf`` (Bambu project unless
--plain), ``layout.json`` (per-cell composition, for the analyser), preview PNG.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_calibration_pad import _plate_with_holes, write_stl  # noqa: E402
from bambu_mix3mf import write_bambu_color_mix_3mf, default_template  # noqa: E402

# display colours for the base filaments (cosmetic; reassign in the slicer)
BASE_COLOURS = {"red": "#E04030", "green": "#40C040", "blue": "#4060E0",
                "amber": "#E0A030", "black": "#111111"}


def parse_stack(spec):
    out = []
    for part in spec.split(","):
        name, t = part.split("=")
        out.append((name.strip(), float(t)))
    return out


def build_layout(opts):
    stacks = [parse_stack(s) for s in opts.stacks]
    n = len(stacks)
    # base filament order = first appearance across stacks, then black
    fils = []
    for st in stacks:
        for name, _ in st:
            if name not in fils:
                fils.append(name)
    slot = {name: i + 1 for i, name in enumerate(fils)}
    black_slot = len(fils) + 1

    pad_w = opts.screen_w_mm - 2 * opts.margin_mm
    pad_h = opts.screen_h_mm - 2 * opts.margin_mm
    edge, header = opts.edge_mm, opts.header_mm
    cols = opts.cols
    rows = int(np.ceil(n / cols))
    gx0, gx1 = edge, pad_w - edge
    gy0, gy1 = edge, pad_h - edge - header
    pitch_x = (gx1 - gx0) / cols
    pitch_y = (gy1 - gy0) / rows
    csz = max(opts.min_cell_mm, min(pitch_x, pitch_y) * opts.cell_fill)

    boxes_by_slot = {s: [] for s in list(slot.values()) + [black_slot]}
    cells, pad_holes = [], []
    for i, st in enumerate(stacks):
        r, c = i // cols, i % cols
        cx = gx0 + (c + 0.5) * pitch_x
        cy = gy1 - (r + 0.5) * pitch_y
        x0, x1 = cx - csz / 2, cx + csz / 2
        y0, y1 = cy - csz / 2, cy + csz / 2
        z = 0.0
        comp = {}
        for name, dz in st:                              # stack layers bottom->top
            boxes_by_slot[slot[name]].append(
                (x0, y0, round(z, 4), x1, y1, round(z + dz, 4)))
            comp[name] = comp.get(name, 0.0) + dz
            z += dz
        pad_holes.append((x0, y0, x1, y1))
        cells.append({"index": i, "composition_mm": {k: round(v, 4)
                                                      for k, v in comp.items()},
                      "thickness_mm": round(z, 4),
                      "cx": round(cx, 3), "cy": round(cy, 3),
                      "w": round(csz, 3), "h": round(csz, 3)})

    # reference window holes: one in each side margin beside every cell (robust
    # for any grid, unlike grid-intersection windows which vanish for 1 column)
    wr = min(csz * 0.16, 3.0)
    windows, win_holes = [], []
    for cel in cells:
        cx, cy, w = cel["cx"], cel["cy"], cel["w"]
        for wx in ((cx - w / 2) / 2, ((cx + w / 2) + pad_w) / 2):
            if wr + 1 < wx < pad_w - wr - 1:
                windows.append({"cx": round(wx, 3), "cy": round(cy, 3),
                                "r": round(wr, 3)})
                win_holes.append((wx - wr, cy - wr, wx + wr, cy + wr))

    # connective web (first filament, unsampled) + black register caps
    web = opts.web_mm
    boxes_by_slot[slot[fils[0]]] += _plate_with_holes(
        0, 0, pad_w, pad_h, 0.0, web, pad_holes + win_holes)
    mk, inset, cap = opts.marker_mm, opts.marker_inset_mm, opts.marker_h_mm
    mz0, mz1 = web, round(web + cap, 4)
    corners = {"bottom_left": (inset, inset),
               "bottom_right": (pad_w - inset - mk, inset),
               "top_right": (pad_w - inset - mk, pad_h - inset - mk),
               "top_left": (inset, pad_h - inset - mk)}
    reg = {}
    for name, (mx0, my0) in corners.items():
        boxes_by_slot[black_slot].append((mx0, my0, mz0, mx0 + mk, my0 + mk, mz1))
        reg[name] = {"cx": round(mx0 + mk / 2, 3), "cy": round(my0 + mk / 2, 3),
                     "w": mk, "h": mk}
    dot = mk * 0.45
    dx0, dy0 = inset + mk + opts.marker_gap_mm, pad_h - inset - dot
    boxes_by_slot[black_slot].append((dx0, dy0, mz0, dx0 + dot, dy0 + dot, mz1))
    reg["orientation_dot"] = {"cx": round(dx0 + dot / 2, 3),
                              "cy": round(dy0 + dot / 2, 3), "w": round(dot, 3),
                              "h": round(dot, 3),
                              "note": "black dot next to the TOP-LEFT corner"}

    layout = {
        "units": "mm", "mode": "full-layer-stack", "filaments": fils,
        "screen_w_mm": opts.screen_w_mm, "screen_h_mm": opts.screen_h_mm,
        "margin_mm": opts.margin_mm,
        "pad_w_mm": round(pad_w, 3), "pad_h_mm": round(pad_h, 3),
        "web_mm": web, "cols": cols, "rows": rows, "cell_mm": round(csz, 3),
        "origin": "bottom-left",
        "pad_corners": [[0, 0], [pad_w, 0], [pad_w, pad_h], [0, pad_h]],
        "register_markers": {"color": "black opaque cap on the web",
                             "cap_mm": cap, "size_mm": mk, "z_mm": [mz0, mz1],
                             "corners": reg},
        "cells": cells, "reference_windows": windows,
    }
    return layout, fils, slot, black_slot, boxes_by_slot


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
        d.ellipse([X(w["cx"]) - rr, Y(w["cy"]) - rr, X(w["cx"]) + rr,
                   Y(w["cy"]) + rr], fill=(255, 255, 255), outline=(150, 150, 160))
    for cel in layout["cells"]:
        x0, y0 = X(cel["cx"] - cel["w"] / 2), Y(cel["cy"] + cel["h"] / 2)
        x1, y1 = X(cel["cx"] + cel["w"] / 2), Y(cel["cy"] - cel["h"] / 2)
        d.rectangle([x0, y0, x1, y1], fill=(200, 180, 170), outline=(70, 70, 80))
        lab = "+".join("%s%.1f" % (k[:1].upper(), v)
                       for k, v in cel["composition_mm"].items())
        d.text((x0 + 3, y0 + 3), lab, fill=(30, 30, 30))
    for name, m in layout["register_markers"]["corners"].items():
        oc = (230, 40, 40) if name == "orientation_dot" else (0, 0, 0)
        d.rectangle([X(m["cx"] - m["w"] / 2), Y(m["cy"] + m["h"] / 2),
                     X(m["cx"] + m["w"] / 2), Y(m["cy"] - m["h"] / 2)],
                    fill=(15, 15, 15), outline=oc, width=2)
    img.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stacks", action="append",
                   help="a stack, e.g. red=0.2,green=0.8 (mm); repeatable. "
                        "Default: the three red/green stacks")
    p.add_argument("--screen-w-mm", type=float, default=64.0)
    p.add_argument("--screen-h-mm", type=float, default=138.0)
    p.add_argument("--margin-mm", type=float, default=3.0)
    p.add_argument("--cols", type=int, default=1, help="cell grid columns (default 1)")
    p.add_argument("--cell-fill", type=float, default=0.72)
    p.add_argument("--min-cell-mm", type=float, default=10.0)
    p.add_argument("--edge-mm", type=float, default=2.0)
    p.add_argument("--header-mm", type=float, default=9.0)
    p.add_argument("--web-mm", type=float, default=0.3)
    p.add_argument("--marker-mm", type=float, default=6.0)
    p.add_argument("--marker-inset-mm", type=float, default=1.0)
    p.add_argument("--marker-h-mm", type=float, default=0.4)
    p.add_argument("--marker-gap-mm", type=float, default=1.5)
    p.add_argument("--bambu-template", default=None)
    p.add_argument("--plain", action="store_true")
    p.add_argument("--also-stl", action="store_true")
    p.add_argument("--out-dir", default=None)
    opts = p.parse_args(argv)
    if not opts.stacks:
        opts.stacks = ["red=0.2,green=0.8", "red=0.4,green=0.6", "red=0.8,green=0.2"]

    out_dir = opts.out_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "stackpad")
    os.makedirs(out_dir, exist_ok=True)

    layout, fils, slot, black_slot, boxes_by_slot = build_layout(opts)
    lay = os.path.join(out_dir, "layout.json")
    prev = os.path.join(out_dir, "stack_pad_preview.png")
    tmf = os.path.join(out_dir, "stack_pad.3mf")
    with open(lay, "w") as f:
        json.dump(layout, f, indent=2)
    write_preview(layout, prev)

    bases = [{"colour": BASE_COLOURS.get(nm, "#AAAAAA")} for nm in fils]
    bases.append({"colour": BASE_COLOURS["black"]})
    parts = [{"name": nm, "boxes": boxes_by_slot[slot[nm]], "slot": slot[nm]}
             for nm in fils]
    parts.append({"name": "markers", "boxes": boxes_by_slot[black_slot],
                  "slot": black_slot})
    template = None if opts.plain else (opts.bambu_template or default_template())
    if template:
        write_bambu_color_mix_3mf(tmf, template, bases, parts)
        kind = "Bambu project, slots: " + ", ".join(
            "%d=%s" % (slot[nm], nm) for nm in fils) + ", %d=black" % black_slot
    else:
        from make_calibration_pad import write_3mf
        # plain 3MF: body = everything except markers, markers separate
        body = [b for nm in fils for b in boxes_by_slot[slot[nm]]]
        write_3mf(tmf, body, boxes_by_slot[black_slot])
        kind = "PLAIN 3MF (assign filaments in slicer; drop --plain for Bambu)"

    if opts.also_stl:
        body = [b for nm in fils for b in boxes_by_slot[slot[nm]]]
        write_stl(os.path.join(out_dir, "stack_pad_body.stl"), body)

    sys.stderr.write(
        "stack pad %.1f x %.1f mm | %d cells | filaments %s\n"
        "wrote:\n  %s\n     ^ %s\n  %s\n  %s\n" % (
            layout["pad_w_mm"], layout["pad_h_mm"], len(layout["cells"]),
            "+".join(fils), tmf, kind, lay, prev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
