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
from fractions import Fraction
from math import gcd

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
        # Surface term T0 is physically ~0.92 (two air/plastic interfaces lose ~8%).
        # A measured T0 well below that means the fit's INTERCEPT absorbed print-line
        # SCATTERING -- pronounced on a weakly-absorbing filament whose curve is
        # nearly flat, so the intercept soaks up the dimming (e.g. light-blue read
        # ~0.5).  That's not a surface term; it wrongly darkens thin panes and made
        # 4-colour disagree with 3-colour (which used the 0.92 default).  Bound it;
        # out of the physical range -> use the default.
        t0 = white[c]["T0"] if c in white else 0.92
        T0.append(t0 if 0.82 <= t0 <= 1.0 else 0.92)
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
    """Load sigma from one or more mixture_calibration.json files.

    Returns (sigma{name:[3]}, pair_sigma{(A,B): {name:[3]}}).  `sigma` is the
    generalized per-filament value (last file wins) used to predict UNSEEN pairs;
    `pair_sigma` holds each DIRECTLY-calibrated pair's own fit, used as a POSTERIOR
    for that exact pair (a measured pair beats the generalized prediction)."""
    sigma, pair_sigma = {}, {}
    for path in paths or []:
        with open(path) as f:
            mc = json.load(f)
        sig = {n: np.asarray(s, float) for n, s in mc.get("sigma", {}).items()}
        for n, s in sig.items():
            sigma[n] = s
        fils = mc.get("filaments")
        if fils and len(fils) == 2 and all(n in sig for n in fils):
            pair_sigma[tuple(sorted(fils))] = {n: sig[n] for n in fils}
    return sigma, pair_sigma


def printable_ratios(base=10, nice_denoms=(3, 4)):
    """Ratios a slicer can actually build as integer sub-layer counts: multiples
    of 1/base (10% steps) PLUS simple fractions (1/3, 2/3, 1/4, 3/4).  A printer
    can't interleave an arbitrary 11% or 57%, but 1/3 = 1:2 sub-layers is clean."""
    rs = {round(i / base, 4) for i in range(base + 1)}
    for dn in nice_denoms:
        rs.update(round(k / dn, 4) for k in range(dn + 1))
    return sorted(rs)


PRINTABLE_RATIOS = printable_ratios()          # {0,.1,.2,.25,.3,.333,.4,.5,...,1}


def sublayer_counts(fracs, max_denom=10):
    """Smallest integer sub-layer counts realizing `fracs` (e.g. [0.667,0.333] ->
    [2,1], [0.3,0.7] -> [3,7]).  This is what the slicer actually interleaves."""
    frs = [Fraction(f).limit_denominator(max_denom) for f in fracs]
    L = 1
    for fr in frs:
        L = L * fr.denominator // gcd(L, fr.denominator)
    counts = [int(fr * L) for fr in frs]
    g = 0
    for c in counts:
        g = gcd(g, c)
    return [c // g for c in counts] if g > 1 else counts


def _pair_sigmas(A, B, sigma, pair_sigma):
    """(sigma_A, sigma_B, source) for a filament pair -- the pair's OWN direct fit
    if it was calibrated (posterior), else the generalized per-filament sigma."""
    key = tuple(sorted((A.name, B.name)))
    if pair_sigma and key in pair_sigma:
        return pair_sigma[key][A.name], pair_sigma[key][B.name], "direct"
    z = np.zeros(3)
    return sigma.get(A.name, z), sigma.get(B.name, z), "generalized"


def predict_mix_linear(fils, sigs, fracs, thickness):
    """Backlit-white linear RGB of a sub-layer mix (fracs sum to 1)."""
    from mixture import mix_tau_multi          # lazy: mixture imports us
    fr = np.asarray(fracs, float)
    fr = fr / fr.sum()
    return mix_tau_multi(fils, [np.asarray(s, float) for s in sigs], fr, thickness)


def sublayer_candidates(pool, sigma, pair_sigma=None, tmin=0.4, tmax=3.0,
                        layer=0.2, max_filaments=2):
    """The full set of PRINTABLE sub-layer recipes with their predicted backlit
    colour -- this IS the colour LUT.  Singles, plus 2-/3-filament mixes at
    printable ratios over a thickness grid, honouring each intense filament's mix
    cap and using a directly-calibrated pair's posterior sigma when available.
    Each entry keeps ``_lin`` (linear RGB) for downstream Delta-E / Lab."""
    thicks = np.round(np.arange(tmin, tmax + 1e-9, layer), 3)
    ratios = [r for r in PRINTABLE_RATIOS if 0 < r < 1]   # slicer-buildable only
    out = []

    def emit(lin, rec):
        rec["_lin"] = lin
        rec["predicted_hex"] = linear_to_hex(lin)
        out.append(rec)

    for m in pool:                                       # single (pure) pane
        for T in thicks:
            emit(predict_linear([m], [T]),
                 {"n": 1, "filaments": [m.name], "fracs_pct": [100],
                  "sublayer_ratio": "1", "thickness_mm": float(T),
                  "has_sigma": True, "sigma_source": "pure"})

    if max_filaments >= 2:
        for a, b in itertools.combinations(range(len(pool)), 2):
            A, B = pool[a], pool[b]
            sA, sB, src = _pair_sigmas(A, B, sigma, pair_sigma)
            have = (src == "direct") or (A.name in sigma and B.name in sigma)
            for p in ratios:                             # p = fraction of B
                if p > B.max_frac + 1e-9 or (1 - p) > A.max_frac + 1e-9:
                    continue                             # intense-filament cap
                counts = sublayer_counts([1 - p, p])
                for T in thicks:
                    emit(predict_mix_linear([A, B], [sA, sB], [1 - p, p], T),
                         {"n": 2, "filaments": [A.name, B.name],
                          "fracs_pct": [round((1 - p) * 100), round(p * 100)],
                          "sublayer_ratio": ":".join(map(str, counts)),
                          "thickness_mm": float(T), "has_sigma": have,
                          "sigma_source": src})

    if max_filaments >= 3:
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
                    counts = sublayer_counts([fa, fb, fc])
                    for T in thicks:
                        emit(predict_mix_linear(mods, sgs, [fa, fb, fc], T),
                             {"n": 3, "filaments": [m.name for m in mods],
                              "fracs_pct": [round(fa * 100), round(fb * 100),
                                            round(fc * 100)],
                              "sublayer_ratio": ":".join(map(str, counts)),
                              "thickness_mm": float(T), "has_sigma": have,
                              "sigma_source": "generalized"})
    return out


def solve_target_sublayer(target_hex, pool, sigma, pair_sigma=None, ratio_step=0.05,
                          tmin=0.4, tmax=3.0, layer=0.2, max_filaments=2,
                          tol_de=1.5, cands=None, max_delta=None):
    """Best sub-layer recipe for one target.  Pass a precomputed ``cands`` (from
    sublayer_candidates) to reuse the LUT across many targets."""
    target_lin = hex_to_linear(target_hex)
    if cands is None:
        cands = sublayer_candidates(pool, sigma, pair_sigma, tmin, tmax, layer,
                                    max_filaments)
    scored = []
    for c in cands:
        d = {k: v for k, v in c.items() if k != "_lin"}
        d["delta_e"] = round(delta_e(target_lin, c["_lin"]), 3)
        scored.append(d)
    if not scored:
        return {"target_hex": target_hex.lower(), "delta_e": None}
    scored.sort(key=lambda c: c["delta_e"])
    if max_delta is not None:
        # tiered: use the FEWEST filaments whose best match is already within
        # max_delta (a good-enough single beats any mix); else the overall best.
        chosen = None
        for n in sorted({c["n"] for c in scored}):
            best_n = min((c for c in scored if c["n"] == n),
                         key=lambda c: c["delta_e"])
            if best_n["delta_e"] <= max_delta:
                chosen = best_n
                break
        if chosen is None:
            chosen = scored[0]
    else:
        best_de = scored[0]["delta_e"]
        near = [c for c in scored if c["delta_e"] <= best_de + tol_de]
        chosen = min(near, key=lambda c: (c["n"], c["delta_e"]))  # prefer simpler
    single = min((c for c in scored if c["n"] == 1),
                 key=lambda c: c["delta_e"], default=None)
    return {"target_hex": target_hex.lower(), "recommended": chosen,
            "best_match": scored[0], "best_single": single}


def _fmt_mix(rec):
    s = " / ".join("%s %d%%" % (n, f)
                   for n, f in zip(rec["filaments"], rec["fracs_pct"]))
    if rec.get("sublayer_ratio"):
        s += "  [%s sublayers]" % rec["sublayer_ratio"]
    return s


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
# Calibration folder layout (the natural workflow):
#   filament/calibration/<name>/calibration.json           single-filament cals
#   filament/calibration/mix/<A>+<B>/mixture_calibration.json   pair sigmas
# Point any command at the root with --cal-root (default) and it discovers both.
CAL_ROOT = "filament/calibration"


def _discover_mixcals(opts):
    """Extend opts.mixcal with every mix/*/mixture_calibration.json under the
    calibration root, so `--cal-root` alone wires up all the pair sigmas.  Skipped
    when the user is explicit with --cal/--cal-dir (they manage --mixcal then)."""
    root = getattr(opts, "cal_root", None)
    if not root or getattr(opts, "cal", None) or getattr(opts, "cal_dir", None):
        return
    found = sorted(glob.glob(os.path.join(root, "mix", "*",
                                          "mixture_calibration.json")))
    have = list(opts.mixcal or [])
    opts.mixcal = have + [p for p in found if p not in have]


def _load_pool(opts):
    pool = []
    cal_dir = getattr(opts, "cal_dir", None)
    root = getattr(opts, "cal_root", None)
    if root and not cal_dir and not (opts.cal):      # single cals in <root>/<name>/
        cal_dir = root
    for spec in opts.cal or []:
        if "=" not in spec:
            raise SystemExit("error: --cal expects name=path.json, got %r" % spec)
        name, path = spec.split("=", 1)
        pool.append(load_filament(name, path))
    if cal_dir:
        for path in sorted(glob.glob(os.path.join(cal_dir, "*",
                                                  "calibration.json"))):
            pool.append(load_filament(os.path.basename(os.path.dirname(path)), path))
    sel = getattr(opts, "filaments", None)           # restrict to a chosen subset
    if sel:
        want = [s.strip() for s in (sel.split(",") if isinstance(sel, str) else sel)]
        order = {n: i for i, n in enumerate(want)}
        pool = sorted((m for m in pool if m.name in order), key=lambda m: order[m.name])
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
    _discover_mixcals(opts)
    pool = _load_pool(opts)
    if not pool:
        raise SystemExit("error: no calibrations (use --cal-root/--cal-dir/--cal)")
    byname = {m.name: m for m in pool}
    if not opts.stack and not opts.mix:
        raise SystemExit("error: pass --stack (full-layer) or --mix (sub-layer)")
    sigma, pair_sigma = {}, {}
    if opts.mix:
        sigma, pair_sigma = load_sigma(opts.mixcal)
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
        if len(fils) == 2:                          # posterior for a direct pair
            sA, sB, src = _pair_sigmas(fils[0], fils[1], sigma, pair_sigma)
            sigs = [sA, sB]
        else:
            sigs = [sigma.get(n, np.zeros(3)) for n in mix]
            src = "generalized"
        lin = predict_mix_linear(fils, sigs, list(mix.values()), opts.thickness)
        hx, lab = linear_to_hex(lin), linear_to_lab(lin)
        tot = sum(mix.values())
        label = "mix: " + " / ".join("%s %d%%" % (n, round(f / tot * 100))
                                     for n, f in mix.items()) + " @%.1fmm[%s]" % (opts.thickness, src)
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


def _draw_filament_map(pool, ths, path):
    """Palette overview: each filament as backlit swatches at several thicknesses,
    with its absorption + intensity class alongside."""
    sw, hh, lw = 96, 66, 96
    W, H = lw + sw * len(ths) + 250, hh * len(pool) + 34
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)
    for j, t in enumerate(ths):
        d.text((lw + j * sw + 10, 8), "%.1fmm" % t, fill=(60, 60, 70))
    for i, m in enumerate(pool):
        y = i * hh + 28
        d.text((6, y + hh // 2 - 10), m.name, fill=(25, 25, 35))
        for j, t in enumerate(ths):
            hexc = linear_to_hex(predict_linear([m], [t]))
            rgb = tuple(int(hexc[k:k + 2], 16) for k in (0, 2, 4))
            d.rectangle([lw + j * sw + 2, y, lw + j * sw + sw - 4, y + hh - 6], fill=rgb)
        x = lw + sw * len(ths) + 14
        d.text((x, y + 10), "a  R%.2f  G%.2f  B%.2f"
               % (m.a[0], m.a[1], m.a[2]), fill=(60, 60, 70))
        if m.max_frac < 1.0:
            d.text((x, y + 30), "INTENSE -- keep mix < %d%%"
                   % round(m.max_frac * 100), fill=(170, 60, 60))
        else:
            d.text((x, y + 30), "normal transparent", fill=(60, 130, 70))
    img.save(path)


def _draw_gamut(cands, path):
    """Reachable colours in the a*b* (hue/chroma) plane -- each printable recipe a
    dot painted its own predicted colour.  Shows how much of colour space the
    palette can hit; the hole in the middle is the un-saturatable greys."""
    labs = [linear_to_lab(c["_lin"]) for c in cands]
    A = [l[1] for l in labs] or [0]
    B = [l[2] for l in labs] or [0]
    amin, amax = min(A + [0]) - 8, max(A + [0]) + 8
    bmin, bmax = min(B + [0]) - 8, max(B + [0]) + 8
    W = H = 560
    pad = 44
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)

    def X(a):
        return pad + (W - 2 * pad) * (a - amin) / (amax - amin)

    def Y(b):
        return H - pad - (H - 2 * pad) * (b - bmin) / (bmax - bmin)

    d.line([X(0), pad, X(0), H - pad], fill=(220, 220, 226))
    d.line([pad, Y(0), W - pad, Y(0)], fill=(220, 220, 226))
    d.text((W - pad - 46, Y(0) + 5), "+a* red", fill=(150, 150, 160))
    d.text((X(0) + 5, pad - 2), "+b* yellow", fill=(150, 150, 160))
    for c, l in zip(cands, labs):
        hx = c["predicted_hex"]
        rgb = tuple(int(hx[k:k + 2], 16) for k in (0, 2, 4))
        x, y = X(l[1]), Y(l[2])
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=rgb)
    d.text((8, 8), "reachable gamut  (a*b* plane, %d printable recipes)" % len(cands),
           fill=(40, 40, 50))
    img.save(path)


def run_lut(opts):
    """Build the full colour LUT: every printable recipe -> its predicted colour,
    written to color_lut.json (+ a gamut image).  With --match, look up targets."""
    _discover_mixcals(opts)
    pool = _load_pool(opts)
    if not pool:
        raise SystemExit("error: no calibrations (use --cal-root/--cal-dir/--cal)")
    sigma, pair_sigma = load_sigma(opts.mixcal)
    if not sigma:
        sys.stderr.write("warning: no --mixcal -> mixes use sigma=0 (baseline; "
                         "single-filament entries are still exact)\n")
    cands = sublayer_candidates(pool, sigma, pair_sigma, opts.min_mm, opts.max_mm,
                                opts.layer, opts.max_filaments)
    entries = []
    for c in cands:
        e = {k: v for k, v in c.items() if k != "_lin"}
        e["lab"] = [round(float(x), 1) for x in linear_to_lab(c["_lin"])]
        entries.append(e)
    os.makedirs(opts.out_dir, exist_ok=True)
    with open(os.path.join(opts.out_dir, "color_lut.json"), "w") as f:
        json.dump({"filaments": [m.name for m in pool], "count": len(entries),
                   "layer_mm": opts.layer, "entries": entries}, f)
    _draw_gamut(cands, os.path.join(opts.out_dir, "gamut.png"))
    n_single = sum(1 for c in cands if c["n"] == 1)
    print("colour LUT: %d recipes (%d single, %d mix) over %d filaments"
          % (len(cands), n_single, len(cands) - n_single, len(pool)))
    if opts.match:
        print("\n%-9s %-9s %5s  recipe" % ("target", "predict", "dE"))
        print("-" * 60)
        for hx in opts.match.split(","):
            hx = hx.strip().lstrip("#")
            r = solve_target_sublayer(hx, pool, sigma, pair_sigma, cands=cands,
                                      tol_de=opts.tol_de)
            rec = r.get("recommended")
            print("#%-8s #%-8s %5.1f  %s @ %.1fmm" %
                  (hx, rec["predicted_hex"], rec["delta_e"], _fmt_mix(rec),
                   rec["thickness_mm"]))
    sys.stderr.write("\nwrote %s/color_lut.json + gamut.png\n" % opts.out_dir)
    return 0


def run_map(opts):
    """Full palette map: filament table + pair-coverage matrix + swatch image."""
    _discover_mixcals(opts)
    pool = _load_pool(opts)
    if not pool:
        raise SystemExit("error: no calibrations (use --cal-root/--cal-dir/--cal)")
    sigma, pair_sigma = load_sigma(opts.mixcal)
    names = [m.name for m in pool]
    print("\nFILAMENT MAP  (%d filaments)\n" % len(pool))
    print("%-10s %-20s %-8s %-6s %s" %
          ("name", "a  R / G / B  (per mm)", "class", "mix<", "surface@1mm"))
    print("-" * 62)
    for m in pool:
        intense = m.max_frac < 1.0
        print("%-10s %5.2f %5.2f %5.2f      %-8s %-6s #%s"
              % (m.name, m.a[0], m.a[1], m.a[2],
                 "INTENSE" if intense else "normal",
                 ("%d%%" % round(m.max_frac * 100)) if intense else "-",
                 linear_to_hex(predict_linear([m], [1.0]))))
    print("\nPAIR COVERAGE  (D = direct-calibrated pair, g = generalizable, . = none)")
    print("      " + " ".join("%-3s" % n[:3] for n in names))
    for i, ni in enumerate(names):
        cells = []
        for j, nj in enumerate(names):
            if i == j:
                cells.append(" - ")
            elif tuple(sorted((ni, nj))) in pair_sigma:
                cells.append(" D ")
            elif ni in sigma and nj in sigma:
                cells.append(" g ")
            else:
                cells.append(" . ")
        print("%-5s " % ni[:5] + " ".join(cells))
    if pair_sigma:
        print("\ndirect pairs (posterior): %s"
              % ", ".join("+".join(k) for k in sorted(pair_sigma)))
    os.makedirs(opts.out_dir, exist_ok=True)
    ths = [float(x) for x in opts.thicknesses.split(",")]
    _draw_filament_map(pool, ths, os.path.join(opts.out_dir, "filament_map.png"))
    sys.stderr.write("\nwrote %s/filament_map.png\n" % opts.out_dir)
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
    sv.add_argument("--cal-root", default=CAL_ROOT,
                    help="calibration root; auto-discovers <name>/calibration.json "
                         "+ mix/*/mixture_calibration.json (default %(default)s)")
    sv.add_argument("--filaments",
                    help="restrict to these filament names (comma-separated)")
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
    pr.add_argument("--cal-root", default=CAL_ROOT,
                    help="calibration root (auto-discovers cals + mix sigmas)")
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

    mp = sub.add_parser("map", help="overview of the whole filament palette")
    mp.add_argument("--cal", action="append",
                    help="filament calibration: name=path/to/calibration.json")
    mp.add_argument("--cal-dir", help="folder of <name>/calibration.json")
    mp.add_argument("--cal-root", default=CAL_ROOT,
                    help="calibration root (auto-discovers cals + mix sigmas)")
    mp.add_argument("--filaments",
                    help="restrict to these filament names (comma-separated)")
    mp.add_argument("--mixcal", action="append",
                    help="mixture_calibration.json (per-filament sigma); repeatable")
    mp.add_argument("--thicknesses", default="0.6,1.2,2.4",
                    help="swatch thicknesses in mm (comma-separated)")
    mp.add_argument("--out-dir", default=CAL_ROOT)

    lt = sub.add_parser("lut", help="build the full printable colour LUT + gamut")
    lt.add_argument("--cal", action="append",
                    help="filament calibration: name=path/to/calibration.json")
    lt.add_argument("--cal-dir", help="folder of <name>/calibration.json")
    lt.add_argument("--cal-root", default=CAL_ROOT,
                    help="calibration root (auto-discovers cals + mix sigmas)")
    lt.add_argument("--filaments",
                    help="restrict to these filament names (comma-separated)")
    lt.add_argument("--mixcal", action="append",
                    help="mixture_calibration.json (per-filament sigma); repeatable")
    lt.add_argument("--max-filaments", type=int, default=2,
                    help="1=singles, 2=+pairs (default), 3=+triples")
    lt.add_argument("--min-mm", type=float, default=0.4)
    lt.add_argument("--max-mm", type=float, default=3.0)
    lt.add_argument("--layer", type=float, default=0.2)
    lt.add_argument("--tol-de", type=float, default=1.5)
    lt.add_argument("--match", help="comma-separated target hex codes to look up")
    lt.add_argument("--out-dir", default=CAL_ROOT)

    st = sub.add_parser("selftest", help="synthetic recover-the-recipe check")
    st.add_argument("--out-dir", default="/tmp/recipes_selftest")

    opts = p.parse_args(argv)
    if opts.cmd == "selftest":
        return run_selftest(opts.out_dir)
    if opts.cmd == "predict":
        return run_predict(opts)
    if opts.cmd == "map":
        return run_map(opts)
    if opts.cmd == "lut":
        return run_lut(opts)

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
        _discover_mixcals(opts)
        sigma, pair_sigma = load_sigma(opts.mixcal)
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
        if pair_sigma:
            sys.stderr.write("directly-calibrated pairs (posterior): %s\n"
                             % ", ".join("+".join(k) for k in sorted(pair_sigma)))
        recipes = [solve_target_sublayer(
            h, pool, sigma, pair_sigma, tmin=opts.min_mm, tmax=opts.max_mm,
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
