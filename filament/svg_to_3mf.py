#!/usr/bin/env python3
"""Stained-glass SVG panes -> one printable Bambu colour-mix 3MF.

The final stage of the pipeline: take the per-colour pane fragments from the SVG
generator (``png_to_stained_glass_svg.py`` -> ``<img>_fragments/color_NN_<hex>.svg``),
snap each pane's colour to the nearest PRINTABLE recipe in the filament LUT
(``solve_recipe.py``), extrude the panes to a fixed panel thickness, and assemble
them into a single Bambu *Color Mixing* 3MF via ``bambu_mix3mf.py`` -- each pane
carrying the single filament or the 2-filament sub-layer mix the LUT chose.

Colour reduction happens HERE, against the real gamut: many original colours that
map to the same recipe merge, and colours the palette can't reach show a high dE
(add the missing filament).  A fixed thickness keeps the panel flat (colour comes
from the sub-layer mix, not from varying height).

    python3 filament/svg_to_3mf.py --frag-dir sample1_fragments \
        --cal-root filament/calibration --thickness 1.6 --out panel.3mf
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# SVG path parsing (fragments are polygons: M / L / H / V / Z, + C just in case)
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[MmLlHhVvCcZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_path_d(d):
    """SVG path 'd' -> list of closed rings, each an Nx2 array of (x,y) mm."""
    toks = _TOKEN.findall(d)
    i, cur, start = 0, None, None
    rings, ring = [], []

    def nums(k):
        nonlocal i
        v = [float(toks[i + j]) for j in range(k)]
        i += k
        return v

    while i < len(toks):
        c = toks[i]
        if c in "Zz":
            i += 1
            if ring:
                rings.append(np.array(ring, float))
                ring = []
            cur = start
            continue
        if c.isalpha():
            cmd = c
            i += 1
        # implicit repeat: reuse cmd on bare coordinates
        if cmd in "Mm":
            x, y = nums(2)
            if cmd == "m" and cur is not None:
                x, y = cur[0] + x, cur[1] + y
            if ring:
                rings.append(np.array(ring, float))
            ring = [(x, y)]
            cur = start = (x, y)
            cmd = "l" if cmd == "m" else "L"          # subsequent pairs are lines
        elif cmd in "Ll":
            x, y = nums(2)
            if cmd == "l":
                x, y = cur[0] + x, cur[1] + y
            ring.append((x, y))
            cur = (x, y)
        elif cmd in "Hh":
            x = nums(1)[0]
            x = cur[0] + x if cmd == "h" else x
            ring.append((x, cur[1]))
            cur = (x, cur[1])
        elif cmd in "Vv":
            y = nums(1)[0]
            y = cur[1] + y if cmd == "v" else y
            ring.append((cur[0], y))
            cur = (cur[0], y)
        elif cmd in "Cc":                             # flatten cubic to segments
            p = nums(6)
            if cmd == "c":
                p = [p[0] + cur[0], p[1] + cur[1], p[2] + cur[0], p[3] + cur[1],
                     p[4] + cur[0], p[5] + cur[1]]
            p0 = np.array(cur)
            c1, c2, p3 = np.array(p[:2]), np.array(p[2:4]), np.array(p[4:])
            for t in np.linspace(0, 1, 9)[1:]:
                b = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1
                     + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3)
                ring.append((b[0], b[1]))
            cur = (p3[0], p3[1])
        else:
            i += 1
    if ring:
        rings.append(np.array(ring, float))
    # drop degenerate rings + duplicate closing point
    out = []
    for r in rings:
        if len(r) >= 3:
            if np.allclose(r[0], r[-1]):
                r = r[:-1]
            if len(r) >= 3:
                out.append(r)
    return out


def _ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_fragment(path):
    """Return (width_mm, height_mm, hex, rings) for a color_NN_<hex>.svg fragment.
    Rings are all the coloured panes (the 1x1 black register rects are skipped)."""
    txt = open(path).read()
    m = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', txt)
    w, h = (float(m.group(3)), float(m.group(4))) if m else (0.0, 0.0)
    hexm = re.search(r"color_\d+_([0-9a-fA-F]{6})", os.path.basename(path))
    hexc = hexm.group(1).lower() if hexm else "888888"
    rings = []
    for pm in re.finditer(r'<path[^>]*\bd="([^"]+)"', txt):
        rings += _parse_path_d(pm.group(1))
    # ignore the tiny 1x1mm corner registration squares
    rings = [r for r in rings if abs(_ring_area(r)) > 0.5]
    return w, h, hexc, rings


# --------------------------------------------------------------------------- #
# Nest rings into polygons-with-holes, triangulate (ear clip + hole bridging),
# and extrude to a slab mesh.
# --------------------------------------------------------------------------- #
def _pt_in_ring(ring, p):
    x, y = p
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _orient(ring, ccw):
    return ring if (_ring_area(ring) > 0) == ccw else ring[::-1]


def _contains(B, A):
    """Ring B contains ring A (majority of A's vertices inside B -- robust to a
    stray boundary-touching vertex; panes never straddle each other)."""
    inside = sum(1 for p in A if _pt_in_ring(B, p))
    return inside > len(A) // 2


def nest_polygons(rings):
    """Group rings into [{outer, holes}] via immediate-parent nesting.  Each ring's
    parent is the SMALLEST ring that contains it; even nesting depth = solid pane,
    odd = hole, and each hole is attached to its immediate (solid) parent."""
    areas = [abs(_ring_area(r)) for r in rings]
    parent = [None] * len(rings)
    for a in range(len(rings)):
        cands = [b for b in range(len(rings))
                 if b != a and areas[b] > areas[a] and _contains(rings[b], rings[a])]
        if cands:
            parent[a] = min(cands, key=lambda b: areas[b])

    def depth(i):
        d, p = 0, parent[i]
        while p is not None:
            d, p = d + 1, parent[p]
        return d

    polys = []
    for i, r in enumerate(rings):
        if depth(i) % 2 == 0:                         # solid pane
            holes = [rings[h] for h in range(len(rings)) if parent[h] == i]
            polys.append({"outer": r, "holes": holes})
    return polys


def _bridge_holes(outer, holes):
    """Merge holes into the outer ring -> one weakly-simple loop.  For each hole's
    rightmost vertex M, cast a +x ray to the nearest outer edge and bridge to that
    edge's larger-x endpoint (Eberly's method) -- always finds a valid bridge."""
    poly = [tuple(p) for p in outer]
    for hole in sorted(holes, key=lambda h: -float(h[:, 0].max())):
        hp = [tuple(p) for p in hole]
        hi = int(np.argmax([p[0] for p in hp]))
        M = hp[hi]
        bestx, best = 1e18, None
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            if (a[1] > M[1]) != (b[1] > M[1]):        # edge straddles M's row
                x = a[0] + (b[0] - a[0]) * (M[1] - a[1]) / (b[1] - a[1])
                if x >= M[0] - 1e-9 and x < bestx:    # nearest hit to the right
                    bestx = x
                    best = k if a[0] > b[0] else (k + 1) % len(poly)
        if best is None:
            best = int(np.argmax([p[0] for p in poly]))
        seq = hp[hi:] + hp[:hi] + [hp[hi]]
        poly = poly[:best + 1] + seq + [poly[best]] + poly[best + 1:]
    return poly


def _ear_clip(poly):
    """Ear-clipping triangulation of a (weakly-)simple CCW polygon -> (P, tris).
    Robust to the coincident vertices a hole bridge introduces."""
    P = _orient(np.array(poly, float), ccw=True)
    idx = list(range(len(P)))
    tris = []

    def cross(o, a, b):
        return (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - \
               (P[a][1] - P[o][1]) * (P[b][0] - P[o][0])

    def inside(a, b, c, p):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and
                    ((d1 > 0) or (d2 > 0) or (d3 > 0)))

    guard = 0
    while len(idx) > 3 and guard < 20 * len(P) ** 2:
        guard += 1
        n = len(idx)
        found = False
        for k in range(n):
            a, b, c = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            if cross(a, b, c) <= 1e-9:                # not a convex corner
                continue
            bad = False
            for m in idx:
                if m in (a, b, c):
                    continue
                if (np.allclose(P[m], P[a]) or np.allclose(P[m], P[b])
                        or np.allclose(P[m], P[c])):  # skip bridge duplicates
                    continue
                if inside(a, b, c, m):
                    bad = True
                    break
            if not bad:
                tris.append((a, b, c))
                idx.pop(k)
                found = True
                break
        if not found:                                # stuck: fan the remainder
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            return P, tris
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return P, tris


def extrude(rings, z0, z1, flip_h=None):
    """rings of ONE fragment -> (vertices, triangles) for all panes extruded z0..z1.
    flip_h: if given, y -> flip_h - y (SVG y-down -> model y-up)."""
    verts, tris = [], []
    for poly in nest_polygons(rings):
        loop = _bridge_holes(_orient(poly["outer"], True),
                             [_orient(h, False) for h in poly["holes"]])
        P, t2d = _ear_clip(loop)
        b = len(verts)
        for (x, y) in P:                              # bottom then top
            yy = (flip_h - y) if flip_h is not None else y
            verts.append((x, yy, z0))
        for (x, y) in P:
            yy = (flip_h - y) if flip_h is not None else y
            verts.append((x, yy, z1))
        m = len(P)
        for (i, j, k) in t2d:
            tris.append((b + i, b + k, b + j))        # bottom (down normal)
            tris.append((b + m + i, b + m + j, b + m + k))   # top
        for e in range(m):                            # side walls
            i, j = e, (e + 1) % m
            tris += [(b + i, b + j, b + m + j), (b + i, b + m + j, b + m + i)]
    return verts, tris


# --------------------------------------------------------------------------- #
# Colour mapping (LUT) + Bambu colour-mix 3MF assembly
# --------------------------------------------------------------------------- #
def _lut(cal_root, thickness, max_filaments):
    import solve_recipe as SR

    class O:
        pass
    o = O()
    o.cal = o.cal_dir = o.mixcal = None
    o.cal_root = cal_root
    SR._discover_mixcals(o)
    pool = SR._load_pool(o)
    if not pool:
        raise SystemExit("no calibrations under %s (calibrate some filaments first)"
                         % cal_root)
    sigma, pair_sigma = SR.load_sigma(o.mixcal)
    cands = SR.sublayer_candidates(pool, sigma, pair_sigma, thickness, thickness,
                                   thickness, max_filaments)
    return SR, pool, sigma, pair_sigma, cands


def _corner_boxes(path, height_mm, z0, z1):
    """The 4 registration squares (SVG <rect>) as black marker boxes (y-flipped)."""
    txt = open(path).read()
    boxes = []
    for m in re.finditer(r'<rect\s+x="([-\d.]+)"\s+y="([-\d.]+)"\s+width='
                         r'"([-\d.]+)"\s+height="([-\d.]+)"', txt):
        x, y, w, h = map(float, m.groups())
        boxes.append((x, height_mm - (y + h), z0, x + w, height_mm - y, z1))
    return boxes


def _cluster_colors(colw, k, SR):
    """area-weighted k-means over Lab -> {orig_hex: representative_hex}."""
    labs = np.array([SR.linear_to_lab(SR.hex_to_linear(h)) for h, _ in colw])
    wts = np.array([max(w, 1.0) for _, w in colw])
    seeds = labs[np.argsort(-wts)[:k]].copy()
    lab_id = np.zeros(len(colw), int)
    for _ in range(25):
        d = ((labs[:, None, :] - seeds[None, :, :]) ** 2).sum(2)
        lab_id = d.argmin(1)
        for j in range(k):
            m = lab_id == j
            if m.any():
                seeds[j] = (labs[m] * wts[m, None]).sum(0) / wts[m].sum()
    out = {}
    for i, (h, _) in enumerate(colw):
        j = lab_id[i]
        member = min((ii for ii in range(len(colw)) if lab_id[ii] == j),
                     key=lambda ii: ((labs[ii] - seeds[j]) ** 2).sum())
        out[h] = colw[member][0]
    return out


def map_recipes(frag_dir, cal_root, thickness=1.6, max_delta=20.0,
                num_colors=None, max_size_mm=None):
    """Read fragments and map each pane colour to the nearest printable recipe."""
    SR, pool, sigma, pair_sigma, cands2 = _lut(cal_root, thickness, 2)
    cands3 = SR.sublayer_candidates(pool, sigma, pair_sigma, thickness, thickness,
                                    thickness, 3)
    frags = sorted(glob.glob(os.path.join(frag_dir, "color_*.svg")))
    if not frags:
        raise SystemExit("no color_*.svg fragments in %s" % frag_dir)
    items = []
    for f in frags:
        w, h, hexc, rings = read_fragment(f)
        items.append({"file": f, "w": w, "h": h, "hex": hexc, "rings": rings,
                      "area": sum(abs(_ring_area(r)) for r in rings)})
    W, H = items[0]["w"], items[0]["h"]
    scale = (max_size_mm / max(W, H)) if max_size_mm else 1.0
    if scale != 1.0:                                 # resize panel, keep aspect
        for it in items:
            it["rings"] = [r * scale for r in it["rings"]]
        W, H = W * scale, H * scale

    targets = {it["hex"]: it["hex"] for it in items}
    if num_colors and num_colors < len(items):
        targets = _cluster_colors([(it["hex"], it["area"]) for it in items],
                                  num_colors, SR)

    def recipe(hexc):
        rec = SR.solve_target_sublayer(hexc, pool, sigma, pair_sigma,
                                       cands=cands2)["recommended"]
        if rec["delta_e"] > max_delta:               # allow a 3-mix as last resort
            r3 = SR.solve_target_sublayer(hexc, pool, sigma, pair_sigma,
                                          cands=cands3)["recommended"]
            if r3["delta_e"] < rec["delta_e"] - 0.5:
                rec = r3
        return rec
    rec_cache = {t: recipe(t) for t in set(targets.values())}
    return {"SR": SR, "pool": pool, "items": items, "rec_cache": rec_cache,
            "targets": targets, "W": W, "H": H, "scale": scale,
            "names": [m.name for m in pool], "thickness": thickness,
            "max_delta": max_delta}


def render_preview(m, path, ppm=3.0):
    """Gamut preview PNG: each pane painted its PRINTABLE recipe colour (what the
    panel will actually look like), not the original image colour."""
    from PIL import Image, ImageDraw
    W, H = m["W"], m["H"]
    img = Image.new("RGB", (max(1, int(W * ppm)), max(1, int(H * ppm))),
                    (16, 16, 20))
    d = ImageDraw.Draw(img)
    panes = []
    for it in m["items"]:
        hx = m["rec_cache"][m["targets"][it["hex"]]]["predicted_hex"]
        rgb = tuple(int(hx[2 * k:2 * k + 2], 16) for k in range(3))
        for poly in nest_polygons(it["rings"]):       # solid panes only
            panes.append((abs(_ring_area(poly["outer"])), poly["outer"], rgb))
    for _, outer, rgb in sorted(panes, key=lambda x: -x[0]):   # big -> small on top
        d.polygon([(x * ppm, y * ppm) for x, y in outer], fill=rgb)
    img.save(path)
    return path


def _print_table(m):
    rows = [(it["hex"], m["rec_cache"][m["targets"][it["hex"]]], it["area"])
            for it in m["items"]]
    print("\npanel %.0fx%.0f mm @ %.1f mm | filaments: %s + black"
          % (m["W"], m["H"], m["thickness"], ", ".join(m["names"])))
    print("\n%-9s %6s  %-28s %5s" % ("colour", "area", "recipe", "dE"))
    print("-" * 56)
    ngam = 0
    for hexc, rec, area in sorted(rows, key=lambda r: -r[2]):
        mix = " / ".join("%s %d%%" % (n, f)
                         for n, f in zip(rec["filaments"], rec["fracs_pct"]))
        out = rec["delta_e"] > m["max_delta"]
        ngam += out
        print("#%-8s %6.0f  %-28s %5.1f%s" % (hexc, area, mix, rec["delta_e"],
              "  << out of gamut" if out else ""))
    print("\n%d panes, %d out of gamut (dE>%.0f -- add the missing filament)"
          % (len(rows), ngam, m["max_delta"]))


def build_3mf(frag_dir, cal_root, out_path, thickness=1.6, max_delta=20.0,
              num_colors=None, bed_mm=256.0, max_size_mm=None):
    from bambu_mix3mf import write_bambu_color_mix_3mf, default_template
    m = map_recipes(frag_dir, cal_root, thickness, max_delta, num_colors, max_size_mm)
    SR, pool, items = m["SR"], m["pool"], m["items"]
    names, H, scale = m["names"], m["H"], m["scale"]
    rec_cache, targets = m["rec_cache"], m["targets"]
    slot_of = {n: i + 1 for i, n in enumerate(names)}
    black_slot = len(names) + 1

    parts, rows = [], []
    for it in items:
        rec = rec_cache[targets[it["hex"]]]
        v, t = extrude(it["rings"], 0.0, thickness, flip_h=H)
        if not t:
            continue
        part = {"name": "c_%s" % it["hex"], "mesh": (v, t)}
        if rec["n"] == 1:
            part["slot"] = slot_of[rec["filaments"][0]]
        else:
            part["mix"] = {"components": [slot_of[n] for n in rec["filaments"]],
                           "ratios": [f / 100.0 for f in rec["fracs_pct"]],
                           "colour": "#" + rec["predicted_hex"]}
        parts.append(part)
        rows.append((it["hex"], rec, it["area"]))
    cb = _corner_boxes(items[0]["file"], H / scale, 0.0, thickness)
    cb = [tuple(v * scale if k in (0, 1, 3, 4) else v for k, v in enumerate(b))
          for b in cb]
    if cb:
        parts.append({"name": "markers", "boxes": cb, "slot": black_slot})

    bases = [{"colour": "#" + SR.linear_to_hex(SR.predict_linear([fl], [thickness]))}
             for fl in pool] + [{"colour": "#111111"}]
    write_bambu_color_mix_3mf(out_path, default_template(), bases, parts, bed_mm)
    render_preview(m, os.path.splitext(out_path)[0] + "_preview.png")
    _print_table(m)
    print("\n-> %s (+ _preview.png)" % out_path)
    return 0


def _tri_area(verts, tris, zpick):
    a = 0.0
    for (i, j, k) in tris:
        A, B, C = verts[i], verts[j], verts[k]
        if not (A[2] == B[2] == C[2] == zpick):
            continue
        a += abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) / 2
    return a


def _selftest():
    # synthetic: 20x20 square with a 6x6 hole -> net area 400-36=364
    outer = np.array([(0, 0), (20, 0), (20, 20), (0, 20)], float)
    hole = np.array([(7, 7), (13, 7), (13, 13), (7, 13)], float)
    v, t = extrude([outer, hole], 0.0, 1.6)
    top = _tri_area(v, t, 1.6)
    print("synthetic square-with-hole: triangulated top area %.1f (expect 364.0) %s"
          % (top, "OK" if abs(top - 364) < 1 else "FAIL"))

    frag = sorted(glob.glob("sample1_fragments/color_*.svg"))
    if not frag:
        print("(no sample fragments to test on)")
        return 0
    print("\n%-22s %5s %6s %9s %9s" % ("fragment", "rings", "polys", "net_mm2",
                                       "tri_mm2"))
    ok = True
    for f in frag:
        w, h, hexc, rings = read_fragment(f)
        polys = nest_polygons(rings)
        net = sum(abs(_ring_area(p["outer"])) - sum(abs(_ring_area(hh))
                  for hh in p["holes"]) for p in polys)
        v, t = extrude(rings, 0.0, 1.6, flip_h=h)
        tri = _tri_area(v, t, 1.6)
        good = abs(tri - net) < max(2.0, 0.02 * net)
        ok = ok and good
        print("%-22s %5d %6d %9.0f %9.0f  %s" % (os.path.basename(f)[:22],
              len(rings), len(polys), net, tri, "" if good else "<-- MISMATCH"))
    print("\ngeometry %s" % ("OK" if ok else "has triangulation errors"))
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frag-dir", help="folder of color_NN_<hex>.svg fragments")
    p.add_argument("--cal-root", default="filament/calibration",
                   help="calibration root (the LUT source)")
    p.add_argument("--out", default="panel.3mf", help="output Bambu 3MF")
    p.add_argument("--thickness", type=float, default=1.6,
                   help="panel thickness mm (flat; colour via sub-layer mix)")
    p.add_argument("--max-delta", type=float, default=20.0,
                   help="allow a 3-filament mix only if 2-mix exceeds this dE")
    p.add_argument("--num-colors", type=int,
                   help="reduce the palette to this many recipes (default: all)")
    p.add_argument("--max-size-mm", type=float,
                   help="scale the panel so its longest side is this (keep aspect)")
    p.add_argument("--selftest", action="store_true", help="geometry self-test")
    opts = p.parse_args(argv)
    if opts.selftest or not opts.frag_dir:
        return _selftest()
    return build_3mf(opts.frag_dir, opts.cal_root, opts.out, opts.thickness,
                     opts.max_delta, opts.num_colors, max_size_mm=opts.max_size_mm)


if __name__ == "__main__":
    raise SystemExit(main())
