#!/usr/bin/env python3
"""Pick a transparent-filament print recipe for each target glass colour.

Phase B of the filament toolchain.  Given per-filament calibrations from
``analyze_calibration.py`` (how each filament absorbs light per mm, per channel)
and a set of TARGET colours (hex, e.g. the SVG generator's ``color_NN_<hex>``
palette), find for each target the stack of filaments + thicknesses whose
backlit-white appearance best matches it (minimum Lab Delta-E).

Modes (``--mode``):
  * ``full-layer`` (default) -- each filament contributes a block of whole layers;
    a pane is one filament, or a stack of a few, each at its own thickness.  This
    is the finished, print-once-per-filament model below.
  * ``sub-layer``  -- Bambu Studio "Color Mixing": 2-3 same-material filaments
    interleaved as thin alternating sub-layers at a chosen RATIO over a total
    thickness.  On real prints the mix deviates from pure volume-weighted
    absorption by dE ~15-20 mid-ramp (interleaved layers transmit MORE than the
    average), so it needs its OWN mixture calibration -- pass ``--mixcal`` with the
    per-filament sigma from ``mixture.py fit``.  With sigma it predicts to the
    print-repeatability floor (dE ~4).  Recipe = filament pair + ratio + thickness.

Model (per channel c), a stack of filaments i at thicknesses t_i, lit from behind
by white:
    T_c = T0_c * exp(- sum_i a_ic * t_i)          (Beer-Lambert; a from calib)
    predicted linear colour = T_c   (white backlight normalised to 1)
A single filament reproduces its calibration exactly.  For a stack we use one
shared surface term T0 (the fused print has one pair of outer air interfaces),
which is the pre-stack-calibration estimate; a measured 2-stack later refines it.

Constraints (agreed design): a pane is ONE filament at a chosen thickness when it
can match; the solver may mix up to --max-filaments (default 3) from the pool.

Usage:
    # calibrations + explicit targets
    python3 filament/solve_recipe.py solve \
        --cal amber=filament/cal_amber/calibration.json \
        --cal teal=filament/cal_teal/calibration.json \
        --targets 166693,982d24,cfac37 --out-dir filament/recipes

    # read the target palette straight from an SVG-generator fragments folder
    python3 filament/solve_recipe.py solve --cal-dir filament/cals \
        --from-svg-dir sample1_fragments --out-dir filament/recipes

    # no printer / no calibrations yet: synthetic end-to-end check
    python3 filament/solve_recipe.py selftest
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Colour science: sRGB <-> linear <-> Lab, and Delta-E (CIE76)
# --------------------------------------------------------------------------- #
_M_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_linear(h):
    h = h.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    return srgb_to_linear(rgb)


def linear_to_hex(lin):
    srgb = np.clip(linear_to_srgb(lin) * 255.0, 0, 255).round().astype(int)
    return "%02x%02x%02x" % tuple(srgb)


def linear_to_lab(lin):
    xyz = _M_RGB2XYZ @ np.clip(np.asarray(lin, float), 0, 1) / _WHITE_D65

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)

    fx, fy, fz = f(xyz)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def delta_e(lin_a, lin_b):
    """CIE76 Delta-E between two linear-RGB colours (Euclidean in Lab)."""
    return float(np.linalg.norm(linear_to_lab(lin_a) - linear_to_lab(lin_b)))


# --------------------------------------------------------------------------- #
# Filament model
# --------------------------------------------------------------------------- #
class Filament:
    def __init__(self, name, a, T0, max_frac=1.0):
        self.name = name
        self.a = np.asarray(a, float)          # absorption per mm, per channel
        self.T0 = np.asarray(T0, float)        # zero-thickness surface transmittance
        self.max_frac = float(max_frac)        # cap on this filament's sub-layer mix
        #                                        share (a VERY intense filament <0.4)

    def __repr__(self):
        return "Filament(%s, a=%s, T0=%s)" % (
            self.name, np.round(self.a, 3).tolist(), np.round(self.T0, 3).tolist())


def load_filament(name, cal_path):
    """Build a Filament from an analyze_calibration.py calibration.json."""
    with open(cal_path) as f:
        cal = json.load(f)
    prim = cal.get("primary_absorption_per_mm", {})
    white = cal.get("screens", {}).get("white", {}).get("per_channel", {})
    a, T0 = [], []
    for c in ("R", "G", "B"):
        # absorption: prefer the matching-primary-screen value, else white screen
        if c in prim:
            a.append(prim[c])
        elif c in white:
            a.append(white[c]["a"])
        else:
            raise SystemExit("error: %s has no absorption for channel %s" %
                             (cal_path, c))
        T0.append(white[c]["T0"] if c in white else 0.92)
    max_frac = cal.get("reliability", {}).get("recommended_max_mix_fraction", 1.0)
    return Filament(name, a, T0, max_frac=max_frac)


# --------------------------------------------------------------------------- #
# Forward model + per-target solve
# --------------------------------------------------------------------------- #
def predict_linear(models, thicks):
    """Predicted backlit-white linear RGB for a stack of models at thicks (mm)."""
    thicks = np.asarray(thicks, float)
    used = thicks > 1e-9
    absorb = sum(m.a * t for m, t in zip(models, thicks))
    if used.any():                             # one shared surface term (fused)
        T0 = np.exp(np.mean([np.log(m.T0) for m, u in zip(models, used) if u],
                            axis=0))
    else:
        T0 = np.ones(3)
    return np.clip(T0 * np.exp(-np.asarray(absorb, float)), 0, 1)


def _nnls_thickness(models, target_lin):
    """Non-negative thicknesses that hit the target absorbance (active-set)."""
    A = np.array([[m.a[c] for m in models] for c in range(3)])      # 3 x k
    T0ref = np.exp(np.mean([np.log(m.T0) for m in models], axis=0))  # 3
    d = np.log(T0ref) - np.log(np.clip(target_lin, 1e-3, 1.0))       # 3
    active = list(range(len(models)))
    t = np.zeros(len(models))
    while active:
        sol, *_ = np.linalg.lstsq(A[:, active], d, rcond=None)
        if (sol >= -1e-9).all():
            for j, idx in enumerate(active):
                t[idx] = max(sol[j], 0.0)
            break
        active.pop(int(np.argmin(sol)))         # drop most-negative filament
    return t


def solve_subset(models, target_lin, tmin, tmax, layer):
    """Best thicknesses for a fixed filament subset; returns (thicks, deltaE)."""
    t = np.clip(_nnls_thickness(models, target_lin), 0, tmax)
    t = np.round(t / layer) * layer
    t[t < tmin] = 0.0

    def de(tv):
        return delta_e(target_lin, predict_linear(models, tv))

    best = de(t)
    steps = [layer * s for s in (-3, -2, -1, 1, 2, 3)]   # local refine on real DE
    improved = True
    while improved:
        improved = False
        for i in range(len(models)):
            for ds in steps:
                tv = t.copy()
                tv[i] = np.clip(tv[i] + ds, 0, tmax)
                if tv[i] < tmin:
                    tv[i] = 0.0
                tv = np.round(tv / layer) * layer
                d2 = de(tv)
                if d2 < best - 1e-9:
                    best, t, improved = d2, tv, True
    return t, best


def solve_target(target_hex, pool, max_filaments=3, tmin=0.2, tmax=4.0,
                 layer=0.1, tol_de=1.5):
    """Find the best recipe for one target hex; returns a recipe dict."""
    target_lin = hex_to_linear(target_hex)
    cands = []
    for r in range(1, min(max_filaments, len(pool)) + 1):
        for combo in itertools.combinations(range(len(pool)), r):
            models = [pool[i] for i in combo]
            t, de = solve_subset(models, target_lin, tmin, tmax, layer)
            layers = [{"filament": models[j].name, "thickness_mm": round(t[j], 3)}
                      for j in range(len(models)) if t[j] > 0]
            if not layers:
                continue
            cands.append({"delta_e": round(de, 3), "n": len(layers),
                          "layers": layers,
                          "predicted_hex": linear_to_hex(
                              predict_linear(models, t))})
    if not cands:
        return {"target_hex": target_hex.lower(), "delta_e": None, "layers": []}
    # dedupe identical recipes, then rank: prefer fewer filaments within tol_de
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: c["delta_e"]):
        key = tuple(sorted((l["filament"], l["thickness_mm"]) for l in c["layers"]))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    best_de = uniq[0]["delta_e"]
    near = [c for c in uniq if c["delta_e"] <= best_de + tol_de]
    chosen = min(near, key=lambda c: (c["n"], c["delta_e"]))
    single = min((c for c in uniq if c["n"] == 1),
                 key=lambda c: c["delta_e"], default=None)
    return {"target_hex": target_hex.lower(),
            "recommended": chosen,
            "best_match": uniq[0],
            "best_single": single}


# --------------------------------------------------------------------------- #
# Sub-layer mixing (Bambu Studio Color Mixing): 2-3 filaments interleaved as
# thin alternating sublayers at a ratio.  Built on the full-layer calibration
# (absorption a, surface T0) PLUS a per-filament mixture term sigma fitted by
# mixture.py from printed ramps.  Predicted transmittance = mix_tau_multi.
# --------------------------------------------------------------------------- #
def load_sigma(paths):
    """Merge per-filament sigma from one or more mixture_calibration.json files.

    Returns (sigma{name:[3]}, measured_pairs{key:{...}}).  Later files override
    earlier ones for the same filament; each pair's measured ramp is kept for the
    posterior override (a directly-measured pair beats the generalized sigma)."""
    sigma, pairs = {}, {}
    for path in paths or []:
        with open(path) as f:
            mc = json.load(f)
        for n, s in mc.get("sigma", {}).items():
            sigma[n] = np.asarray(s, float)
        fils = mc.get("filaments")
        if fils and mc.get("measured_tau"):
            pairs["+".join(sorted(fils))] = {
                "A": fils[0], "B": fils[1], "ratios": mc["ratios"],
                "tau": mc["measured_tau"], "T": mc.get("thickness_mm", 1.0)}
    return sigma, pairs


def predict_mix_linear(fils, sigs, fracs, thickness):
    """Backlit-white linear RGB of a sub-layer mix (fracs sum to 1)."""
    from mixture import mix_tau_multi          # lazy: mixture imports us
    fr = np.asarray(fracs, float)
    fr = fr / fr.sum()
    return mix_tau_multi(fils, [np.asarray(s, float) for s in sigs], fr, thickness)


def solve_target_sublayer(target_hex, pool, sigma, ratio_step=0.05,
                          tmin=0.4, tmax=3.0, layer=0.2, max_filaments=2,
                          tol_de=1.5):
    """Best sub-layer recipe for one target: a single filament, or 2-3 filaments
    interleaved at a ratio, over a total thickness.  Returns a recipe dict."""
    target_lin = hex_to_linear(target_hex)
    thicks = np.round(np.arange(tmin, tmax + 1e-9, layer), 3)
    ratios = np.round(np.arange(ratio_step, 1.0 - 1e-9, ratio_step), 3)
    cands = []

    def add(lin, rec):
        rec["delta_e"] = round(delta_e(target_lin, lin), 3)
        rec["predicted_hex"] = linear_to_hex(lin)
        cands.append(rec)

    for m in pool:                                       # single (pure) pane
        for T in thicks:
            add(predict_linear([m], [T]),
                {"n": 1, "filaments": [m.name], "fracs_pct": [100],
                 "thickness_mm": float(T), "has_sigma": True})

    combos = []
    if max_filaments >= 2:
        combos += [c for c in itertools.combinations(range(len(pool)), 2)]
    for a, b in combos:                                  # 2-filament mix
        A, B = pool[a], pool[b]
        sA = sigma.get(A.name, np.zeros(3))
        sB = sigma.get(B.name, np.zeros(3))
        have = A.name in sigma and B.name in sigma
        for p in ratios:                                 # p = fraction of B
            if p > B.max_frac + 1e-9 or (1 - p) > A.max_frac + 1e-9:
                continue                                 # intense-filament cap
            for T in thicks:
                add(predict_mix_linear([A, B], [sA, sB], [1 - p, p], T),
                    {"n": 2, "filaments": [A.name, B.name],
                     "fracs_pct": [round((1 - p) * 100), round(p * 100)],
                     "thickness_mm": float(T), "has_sigma": have})

    if max_filaments >= 3:                               # 3-filament mix (coarse)
        r3 = np.round(np.arange(0.2, 0.81, 0.2), 3)
        for a, b, c in itertools.combinations(range(len(pool)), 3):
            mods = [pool[a], pool[b], pool[c]]
            sgs = [sigma.get(m.name, np.zeros(3)) for m in mods]
            have = all(m.name in sigma for m in mods)
            for fa in r3:
                for fb in r3:
                    fc = round(1 - fa - fb, 3)
                    if fc < 0.19:
                        continue
                    if (fa > mods[0].max_frac + 1e-9 or fb > mods[1].max_frac + 1e-9
                            or fc > mods[2].max_frac + 1e-9):
                        continue                         # intense-filament cap
                    for T in thicks:
                        add(predict_mix_linear(mods, sgs, [fa, fb, fc], T),
                            {"n": 3, "filaments": [m.name for m in mods],
                             "fracs_pct": [round(fa * 100), round(fb * 100),
                                           round(fc * 100)],
                             "thickness_mm": float(T), "has_sigma": have})

    if not cands:
        return {"target_hex": target_hex.lower(), "delta_e": None}
    cands.sort(key=lambda c: c["delta_e"])
    best_de = cands[0]["delta_e"]
    near = [c for c in cands if c["delta_e"] <= best_de + tol_de]
    chosen = min(near, key=lambda c: (c["n"], c["delta_e"]))    # prefer simpler
    single = min((c for c in cands if c["n"] == 1),
                 key=lambda c: c["delta_e"], default=None)
    return {"target_hex": target_hex.lower(), "recommended": chosen,
            "best_match": cands[0], "best_single": single}


def _fmt_mix(rec):
    return " / ".join("%s %d%%" % (n, f)
                      for n, f in zip(rec["filaments"], rec["fracs_pct"]))


def print_table_sublayer(recipes):
    print("\n%-9s %-9s %5s  recipe (sub-layer mix, total mm)" %
          ("target", "predict", "dE"))
    print("-" * 66)
    for r in recipes:
        rec = r.get("recommended")
        if not rec:
            print("#%-8s  (no recipe)" % r["target_hex"])
            continue
        warn = "" if rec.get("has_sigma", True) else "  [!no-sigma: baseline]"
        print("#%-8s #%-8s %5.1f  %s @ %.1fmm%s"
              % (r["target_hex"], rec["predicted_hex"], rec["delta_e"],
                 _fmt_mix(rec), rec["thickness_mm"], warn))


def write_swatches_sublayer(recipes, path, sw=120, hh=64):
    rows = [r for r in recipes if r.get("recommended")]
    if not rows:
        return
    img = Image.new("RGB", (sw * 2 + 320, hh * len(rows) + 2), (250, 250, 252))
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        y = i * hh + 1
        rec = r["recommended"]
        for j, hx in enumerate((r["target_hex"], rec["predicted_hex"])):
            rgb = tuple(int(hx[k:k + 2], 16) for k in (0, 2, 4))
            d.rectangle([j * sw + 1, y, j * sw + sw, y + hh - 2], fill=rgb)
        d.text((2 * sw + 8, y + 4), "target #%s" % r["target_hex"], fill=(40, 40, 40))
        d.text((2 * sw + 8, y + 22), "-> #%s  dE=%.1f" %
               (rec["predicted_hex"], rec["delta_e"]), fill=(40, 40, 40))
        d.text((2 * sw + 8, y + 40), "%s @ %.1fmm" % (_fmt_mix(rec),
               rec["thickness_mm"]), fill=(90, 90, 110))
    img.save(path)


# --------------------------------------------------------------------------- #
# Palette input + reporting
# --------------------------------------------------------------------------- #
_HEX = re.compile(r"([0-9a-fA-F]{6})")


def targets_from_svg_dir(d):
    """Extract hex codes from ``color_NN_<hex>.*`` filenames in a folder."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, "color_*"))):
        m = _HEX.search(os.path.splitext(os.path.basename(p))[0])
        if m and m.group(1).lower() not in out:
            out.append(m.group(1).lower())
    return out


def write_swatches(recipes, path, sw=120, hh=64):
    """target-vs-predicted colour strip so you can eyeball the matches."""
    rows = [r for r in recipes if r.get("recommended")]
    img = Image.new("RGB", (sw * 2 + 260, hh * len(rows) + 2), (250, 250, 252))
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        y = i * hh + 1
        rec = r["recommended"]
        for j, hx in enumerate((r["target_hex"], rec["predicted_hex"])):
            rgb = tuple(int(hx[k:k + 2], 16) for k in (0, 2, 4))
            d.rectangle([j * sw + 1, y, j * sw + sw, y + hh - 2], fill=rgb)
        d.text((2 * sw + 8, y + 6), "target #%s" % r["target_hex"], fill=(40, 40, 40))
        recipe = " + ".join("%s %.2fmm" % (l["filament"], l["thickness_mm"])
                            for l in rec["layers"])
        d.text((2 * sw + 8, y + 24), "-> #%s  dE=%.1f" %
               (rec["predicted_hex"], rec["delta_e"]), fill=(40, 40, 40))
        d.text((2 * sw + 8, y + 42), recipe, fill=(90, 90, 110))
    img.save(path)


def print_table(recipes):
    print("\n%-9s %-9s %5s  %-2s  recipe" % ("target", "predict", "dE", "n"))
    print("-" * 64)
    for r in recipes:
        rec = r.get("recommended")
        if not rec:
            print("#%-8s  (no recipe)" % r["target_hex"])
            continue
        recipe = " + ".join("%s %.2f" % (l["filament"], l["thickness_mm"])
                            for l in rec["layers"])
        print("#%-8s #%-8s %5.1f  %d   %s"
              % (r["target_hex"], rec["predicted_hex"], rec["delta_e"],
                 rec["n"], recipe))


# --------------------------------------------------------------------------- #
# Self-test: synthetic filaments + targets -> solve -> verify
# --------------------------------------------------------------------------- #
def _synthetic_pool():
    return [
        Filament("amber",   a=[0.25, 0.55, 1.70], T0=[0.94, 0.94, 0.92]),
        Filament("cyan",    a=[1.60, 0.35, 0.40], T0=[0.93, 0.94, 0.94]),
        Filament("magenta", a=[0.35, 1.55, 0.45], T0=[0.94, 0.92, 0.94]),
        Filament("smoke",   a=[0.75, 0.72, 0.70], T0=[0.93, 0.93, 0.93]),
    ]


def run_selftest(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pool = _synthetic_pool()
    byname = {m.name: m for m in pool}

    # plant known recipes, render their target hex, then see if we recover them
    planted = [
        ("amber", [("amber", 1.0)]),
        ("cyan",  [("cyan", 0.8)]),
        ("amber+cyan", [("amber", 0.6), ("cyan", 0.5)]),
        ("magenta+smoke", [("magenta", 0.7), ("smoke", 0.3)]),
    ]
    ok = True
    recipes = []
    print("self-test: recover planted recipes (synthetic filaments)")
    print("  %-16s %-9s %-9s %5s  recovered" % ("planted", "target", "predict", "dE"))
    for label, mix in planted:
        models = [byname[n] for n, _ in mix]
        thicks = [t for _, t in mix]
        target_hex = linear_to_hex(predict_linear(models, thicks))
        r = solve_target(target_hex, pool, max_filaments=3, tmin=0.1, layer=0.1)
        recipes.append(r)
        rec = r["recommended"]
        recipe = " + ".join("%s %.2f" % (l["filament"], l["thickness_mm"])
                            for l in rec["layers"])
        good = rec["delta_e"] <= 2.0
        ok = ok and good
        print("  %-16s #%-8s #%-8s %5.1f  %s%s"
              % (label, target_hex, rec["predicted_hex"], rec["delta_e"],
                 recipe, "" if good else "   <-- FAIL"))

    write_swatches(recipes, os.path.join(out_dir, "swatches.png"))
    with open(os.path.join(out_dir, "recipes.json"), "w") as f:
        json.dump(recipes, f, indent=2)
    print("\nwrote recipes.json + swatches.png to %s" % out_dir)
    print("SELF-TEST %s (all planted recipes recovered to dE<=2)"
          % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def _load_pool(opts):
    pool = []
    for spec in opts.cal or []:
        if "=" not in spec:
            raise SystemExit("error: --cal expects name=path.json, got %r" % spec)
        name, path = spec.split("=", 1)
        pool.append(load_filament(name, path))
    if opts.cal_dir:
        for path in sorted(glob.glob(os.path.join(opts.cal_dir, "*",
                                                  "calibration.json"))):
            name = os.path.basename(os.path.dirname(path))
            pool.append(load_filament(name, path))
    return pool


def parse_stack(spec):
    """'red=0.2,green=0.8' -> {'red': 0.2, 'green': 0.8} (thicknesses in mm)."""
    out = {}
    for part in spec.split(","):
        name, t = part.split("=")
        out[name.strip()] = float(t)
    return out


def predict_stack(byname, stack):
    """stack: {filament_name: thickness_mm}.  Returns (linear, hex, Lab)."""
    models = [byname[n] for n in stack]
    thicks = [stack[n] for n in stack]
    lin = predict_linear(models, thicks)
    return lin, linear_to_hex(lin), linear_to_lab(lin)


def write_predict_swatches(rows, path, sw=140, hh=60):
    img = Image.new("RGB", (sw + 300, hh * len(rows) + 2), (250, 250, 252))
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        y = i * hh + 1
        rgb = tuple(int(r["predicted_hex"][k:k + 2], 16) for k in (0, 2, 4))
        d.rectangle([1, y, sw, y + hh - 2], fill=rgb)
        d.text((sw + 10, y + 8), r["label"], fill=(40, 40, 40))
        d.text((sw + 10, y + 28), "-> #%s   L*%.0f a*%.0f b*%.0f"
               % (r["predicted_hex"], *r["lab"]), fill=(90, 90, 110))
    img.save(path)


def run_predict(opts):
    pool = _load_pool(opts)
    if not pool:
        raise SystemExit("error: no calibrations (use --cal or --cal-dir)")
    byname = {m.name: m for m in pool}
    if not opts.stack and not opts.mix:
        raise SystemExit("error: pass --stack (full-layer) or --mix (sub-layer)")
    sigma = {}
    if opts.mix:
        sigma, _ = load_sigma(opts.mixcal)
        if not sigma:
            sys.stderr.write("warning: --mix with no --mixcal -> sigma=0 "
                             "(baseline; off by dE ~15 mid-ramp)\n")
    rows = []
    print("\n%-34s %-9s  Lab" % ("recipe", "predict"))
    print("-" * 60)
    for spec in opts.stack:
        stack = parse_stack(spec)
        miss = [n for n in stack if n not in byname]
        if miss:
            raise SystemExit("error: no calibration for %s (have %s)"
                             % (miss, list(byname)))
        lin, hx, lab = predict_stack(byname, stack)
        label = "stack: " + " + ".join("%s %.2f" % (n, t) for n, t in stack.items())
        rows.append({"label": label, "stack": stack, "predicted_hex": hx,
                     "lab": [round(x, 1) for x in lab]})
        print("%-34s #%-8s  L*%.0f a*%.0f b*%.0f" % (label, hx, *lab))
    for spec in opts.mix:
        mix = parse_stack(spec)                     # name=fraction
        miss = [n for n in mix if n not in byname]
        if miss:
            raise SystemExit("error: no calibration for %s (have %s)"
                             % (miss, list(byname)))
        fils = [byname[n] for n in mix]
        sigs = [sigma.get(n, np.zeros(3)) for n in mix]
        lin = predict_mix_linear(fils, sigs, list(mix.values()), opts.thickness)
        hx, lab = linear_to_hex(lin), linear_to_lab(lin)
        tot = sum(mix.values())
        label = "mix: " + " / ".join("%s %d%%" % (n, round(f / tot * 100))
                                     for n, f in mix.items()) + " @%.1fmm" % opts.thickness
        rows.append({"label": label, "mix": mix, "thickness_mm": opts.thickness,
                     "predicted_hex": hx, "lab": [round(x, 1) for x in lab]})
        print("%-34s #%-8s  L*%.0f a*%.0f b*%.0f" % (label, hx, *lab))
    os.makedirs(opts.out_dir, exist_ok=True)
    with open(os.path.join(opts.out_dir, "predictions.json"), "w") as f:
        json.dump({"filaments": [m.name for m in pool], "stacks": rows}, f, indent=2)
    write_predict_swatches(rows, os.path.join(opts.out_dir,
                                              "predicted_swatches.png"))
    sys.stderr.write("\nwrote predictions.json + predicted_swatches.png to %s\n"
                     % opts.out_dir)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("solve", help="solve recipes for a target palette")
    sv.add_argument("--mode", choices=["full-layer", "sub-layer"],
                    default="full-layer",
                    help="full-layer (default): whole-layer filament blocks. "
                         "sub-layer: Bambu Studio Color Mixing (experimental).")
    sv.add_argument("--cal", action="append",
                    help="filament calibration: name=path/to/calibration.json "
                         "(repeatable)")
    sv.add_argument("--cal-dir",
                    help="folder of <name>/calibration.json calibrations")
    sv.add_argument("--targets", help="comma-separated target hex codes")
    sv.add_argument("--from-svg-dir",
                    help="read targets from color_NN_<hex> filenames in a folder")
    sv.add_argument("--mixcal", action="append",
                    help="[sub-layer] mixture_calibration.json with per-filament "
                         "sigma (repeatable; merges pairs for generalization)")
    sv.add_argument("--max-filaments", type=int, default=3)
    sv.add_argument("--min-mm", type=float, default=0.2)
    sv.add_argument("--max-mm", type=float, default=4.0)
    sv.add_argument("--layer", type=float, default=0.1)
    sv.add_argument("--tol-de", type=float, default=1.5,
                    help="prefer fewer filaments when within this dE of the best")
    sv.add_argument("--out-dir", default="filament/recipes")

    pr = sub.add_parser("predict",
                        help="forward: predict a filament stack's backlit colour")
    pr.add_argument("--cal", action="append",
                    help="filament calibration: name=path/to/calibration.json")
    pr.add_argument("--cal-dir", help="folder of <name>/calibration.json")
    pr.add_argument("--stack", action="append", default=[],
                    help="full-layer stack, e.g. red=0.2,green=0.8 (mm); repeatable")
    pr.add_argument("--mix", action="append", default=[],
                    help="sub-layer mix, e.g. red=0.4,green=0.6 (fractions) "
                         "@thickness in --thickness; repeatable")
    pr.add_argument("--thickness", type=float, default=1.0,
                    help="[--mix] total thickness mm for sub-layer mixes")
    pr.add_argument("--mixcal", action="append",
                    help="[--mix] mixture_calibration.json (per-filament sigma)")
    pr.add_argument("--out-dir", default="filament/predictions")

    st = sub.add_parser("selftest", help="synthetic recover-the-recipe check")
    st.add_argument("--out-dir", default="/tmp/recipes_selftest")

    opts = p.parse_args(argv)
    if opts.cmd == "selftest":
        return run_selftest(opts.out_dir)
    if opts.cmd == "predict":
        return run_predict(opts)

    pool = _load_pool(opts)
    if not pool:
        raise SystemExit("error: no calibrations (use --cal or --cal-dir)")
    targets = []
    if opts.targets:
        targets += [h.strip().lstrip("#").lower() for h in opts.targets.split(",")
                    if h.strip()]
    if opts.from_svg_dir:
        targets += targets_from_svg_dir(opts.from_svg_dir)
    if not targets:
        raise SystemExit("error: no targets (use --targets or --from-svg-dir)")
    os.makedirs(opts.out_dir, exist_ok=True)

    if opts.mode == "sub-layer":
        sigma, mpairs = load_sigma(opts.mixcal)
        if not sigma:
            raise SystemExit(
                "error: sub-layer mode needs mixture calibration -- pass "
                "--mixcal <mixture_calibration.json> (from `mixture.py fit`). "
                "Without sigma, mixes are off by dE ~15 mid-ramp.")
        missing = [m.name for m in pool if m.name not in sigma]
        if missing:
            sys.stderr.write("warning: no sigma for %s -- mixes involving them "
                             "fall back to baseline (unreliable)\n"
                             % ", ".join(missing))
        recipes = [solve_target_sublayer(
            h, pool, sigma, tmin=opts.min_mm, tmax=opts.max_mm,
            layer=opts.layer, max_filaments=opts.max_filaments,
            tol_de=opts.tol_de) for h in targets]
        with open(os.path.join(opts.out_dir, "recipes.json"), "w") as f:
            json.dump({"mode": opts.mode, "filaments": [m.name for m in pool],
                       "sigma_for": sorted(sigma), "recipes": recipes}, f, indent=2)
        write_swatches_sublayer(recipes, os.path.join(opts.out_dir,
                                                      "swatches.png"))
        print_table_sublayer(recipes)
        sys.stderr.write("\npool: %s | sigma: %s | %d targets -> %s\n" %
                         (", ".join(m.name for m in pool), ", ".join(sorted(sigma)),
                          len(targets), os.path.join(opts.out_dir, "recipes.json")))
        return 0

    recipes = [solve_target(h, pool, max_filaments=opts.max_filaments,
                            tmin=opts.min_mm, tmax=opts.max_mm, layer=opts.layer,
                            tol_de=opts.tol_de) for h in targets]
    with open(os.path.join(opts.out_dir, "recipes.json"), "w") as f:
        json.dump({"mode": opts.mode, "filaments": [m.name for m in pool],
                   "recipes": recipes}, f, indent=2)
    write_swatches(recipes, os.path.join(opts.out_dir, "swatches.png"))
    print_table(recipes)
    sys.stderr.write("\npool: %s | %d targets -> %s\n" %
                     (", ".join(m.name for m in pool), len(targets),
                      os.path.join(opts.out_dir, "recipes.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
