# Filament calibration & mixing (transparent filaments)

Tools for picking transparent-filament print recipes (which filament + how thick)
to hit each stained-glass pane's target colour. Transparent filaments have no
standard CMYW set and slicer colour-mix previews are inaccurate versus the real
backlit print, so we calibrate each filament empirically.

Everything here depends on **numpy + PIL only** (no scipy / OpenCV) — homography,
blob detection, and the diagnostic plots are all hand-rolled, so it runs on a
stock Python with just those two packages.

## Workflow

### 1. Generate a calibration pad — `make_calibration_pad.py`

A continuous transparent **base plate** (default 0.4 mm) with a grid of square
cells built on top, so the whole thing is **one rigid piece** — the cells stay
fixed relative to the markers however you set it on the screen. Each cell's total
light path is `base_plate + increment` (increments 0.1 → 2.0 mm in 0.1 steps = 20
cells → totals 0.5 → 2.4 mm); we fit transmittance against that total, so the base
plate is just part of every slab and doesn't bias the fit. **Opaque black register
markers** sit as a thin cap on the 4 corners (a separate part for your black
filament) plus an orientation dot; because the black is top-layers-only, the
slicer needs a single filament change near the end of the print. **Reference
windows are real holes** through the plate, giving true bare-screen samples.

```bash
python3 filament/make_calibration_pad.py          # -> filament/pad/{calibration_pad.3mf, layout.json, preview.png}
```

Sized to lie *inside* a phone screen (default 64×138 mm active area → 58×132 mm
pad). Print at 0.1 mm layer height (divides both the base plate and the step) so
thicknesses are exact. `layout.json` records every cell / reference-window /
marker position in millimetres and each cell's total thickness — the analyser
reads it.

### 2. Photograph the printed pad

Lay the pad on a phone/tablet showing a **full-screen solid colour** (the screen
is both the backlight and a programmable RGB source).

**How many shots?** `analyze` classifies each filament and prints the exact
**capture requirements**:
- **Normal transparent filament** → **one white shot is enough.** White carries
  all three channels and each fits cleanly.
- **Intense filament** (absorbs a channel so hard that white blacks it out — e.g.
  a deep blue kills red) → the report says *which extra single-colour screens to
  shoot* (e.g. "WHITE + RED + GREEN"). Over a full-brightness single-colour screen
  you can over-expose that one channel without clipping the others, turning a
  floored/noisy channel into a real fit. A channel that stays black even on its own
  screen is genuinely opaque there — its absorption is reported as a lower bound
  (fine: it reads ~0 in any mix).

The report also states the **exposure requirement**: the bare screen (reference
windows) must read **~85–95 %** of full — bright but not clipped — and flags yours
as OK / TOO DIM / TOO BRIGHT.

Display the bundled standard primaries `filament/screens/{white,red,green,blue}.png`
(pure sRGB #FFFFFF/#FF0000/#00FF00/#0000FF; regenerate at your screen's resolution
with `python3 filament/make_screens.py --width W --height H`).

Practical must-dos: turn OFF True Tone / Night Shift / auto-brightness, avoid
glare, shoot RAW. **Expose each screen colour on its own** — see below.

**Shoot RAW (DNG/ARW/CR2/…) if you can** — it's linear and skips the phone's tone
curve, which otherwise inflates the absorption ~40–60%. rawpy decodes it
automatically (`pip install rawpy`). Reference exposure: ISO 100, dark room, screen
at max **fixed** brightness, shutter set so **that screen's** bare windows read
~85–95% (bright but not clipped). **Set the shutter per colour, NOT one shutter for
all four** — transmittance is cell ÷ bare-screen *within a single photo*, so the
absolute level cancels and each colour should be exposed on its own; a blue screen
is far dimmer than white, so a white-tuned shutter under-exposes it (lengthen the
shutter, keep ISO low). WB is irrelevant — it cancels in the ratio, and RAW makes
it moot. `analyze` prints **shot-quality warnings** and a per-screen exposure check
(under/over-exposure, noisy fits) with re-shoot tips. The bare screen showing
through the gaps between cells is what normalises out exposure/brightness.

### 3. Analyse the photos — `analyze_calibration.py`

```bash
python3 filament/analyze_calibration.py analyze \
    --layout filament/pad/layout.json --name "PolyTerra Teal" \
    --white white.jpg --red red.jpg --green green.jpg --blue blue.jpg \
    --out-dir filament/cal_polyterra_teal
```

Per photo it detects the 4 black markers (+ dot for orientation), fits a
homography from pad-mm to image pixels, samples every cell and every bare-screen
reference window, divides the cells by a plane fitted through the reference
windows (removes screen brightness gradient / vignetting), and fits

    ln T = b − a·t        a = absorption per mm,  T0 = exp(b) = surface term

per channel. Output: `calibration.json` (headline `primary_absorption_per_mm` =
the red/green/blue-screen diagonal), plus `detect_*.png` overlays and a
`curves.png` transmittance plot.

### 4. Solve print recipes — `solve_recipe.py`

Given the per-filament calibrations and a set of target colours, find the stack of
filaments + thicknesses that best matches each target under a white backlight
(minimum Lab Delta-E). A pane is one filament when that suffices; the solver mixes
up to `--max-filaments` (default 3) from the pool.

```bash
python3 filament/solve_recipe.py solve \
    --cal-dir filament/cals \
    --from-svg-dir sample1_fragments \
    --out-dir filament/recipes
```

Reads targets straight from the SVG generator's `color_NN_<hex>` fragment
filenames (or `--targets a1b2c3,ffcc00`). Writes `recipes.json` and a
`swatches.png` (target vs. predicted colour). The reported Delta-E is honest about
**gamut**: transparent filaments can only subtract light, so saturated targets a
given pool can't reach show a high Delta-E — add a filament that passes the
missing primary.

Model: `T_c = T0_c * exp(-sum_i a_ic * t_i)` per channel; predicted linear colour
= transmittance under white. Stacks use one shared surface term `T0` (the fused
print has one pair of outer interfaces) — a measured 2-stack will later refine the
cross-term. A single filament reproduces its calibration exactly.

## Sub-layer colour mixing (experimental, Bambu Studio Color Mixing)

An alternative to stacking whole-layer blocks: within one solid part, Bambu Studio
interleaves thin alternating sublayers of 2–3 filaments at a ratio, and the backlit
eye blends them. Same physics baseline as full-layer (Beer–Lambert on the mix
fractions), plus an interface-scatter term calibrated separately.

**Calibrate a filament pair** — `make_mixture_pad.py` emits an 11-pad ramp (pad 0 =
pure A … pad 10 = pure B). Each pad is a solid block **tagged with a mix ratio**;
Bambu does the sublayer slicing. The output `mixture_pad.3mf` opens as a genuine
Bambu project with all ratios pre-set (no color-triangle dragging):

```bash
python3 filament/make_mixture_pad.py
# -> filament/mixpad/mixture_pad.3mf  (12 filament slots: 1=A, 2=B, 3=black,
#    4-12 = the 9 mixes; enable_mixed_color_sublayer=1)
```

It templates from the bundled `filament/templates/bambu_p2s.3mf` (a sanitised
Bambu Lab P2S / PETG-Transparent skeleton). For a different machine, pass your own
Bambu `.3mf` export via `--bambu-template <export>.3mf`; `--plain` writes a plain
core-3MF instead (Bambu flags it "not from Bambu Lab"). The single-filament
`make_calibration_pad.py` takes the same flags.

**The mixture model** — `mixture.py`. Requires the full-layer calibration of each
filament as the baseline (`ln T = b − a·t`, exact at the pure ends). It fits a
per-filament scatter `σ_X` from each pair's ramp asymmetry, solved jointly across
pairs so a shared filament stays consistent. Then `T_c = T0·exp(−[base + scatter])`
with `scatter = T·Σ_i σ_i·f_i²(1−f_i)` predicts **any** pair (incl. one never
printed — A–C from A–B + B–C) and **3-filament** mixes from the same pair-calibrated
σ. A pair's own measured ramp overrides the generalized prediction when they differ.

```bash
python3 filament/mixture.py fit --layout filament/mixpad/layout.json --white mix.dng \
    --a red=filament/cals/red/calibration.json --b green=filament/cals/green/calibration.json \
    --out-dir filament/mixcal        # -> mixture_calibration.json (per-filament σ)
python3 filament/mixture.py selftest # recovers σ; predicts unseen A-C + 3-mixes
```

**Solve / predict sub-layer recipes** — the fitted σ feeds straight into
`solve_recipe.py --mode sub-layer` (pass one `--mixcal` per calibrated pair; they
merge, so a shared filament's σ covers unseen pairs A-C from A-B + B-C):

```bash
# forward: predict a specific mix's backlit colour
python3 filament/solve_recipe.py predict --cal-dir filament/cals \
    --mixcal filament/mixcal/mixture_calibration.json \
    --mix red=0.5,green=0.5 --thickness 1.0

# solve: best (filament pair + ratio + thickness) per target palette
python3 filament/solve_recipe.py solve --mode sub-layer --cal-dir filament/cals \
    --mixcal filament/mixcal/mixture_calibration.json \
    --from-svg-dir sample1_fragments --layer 0.2 --out-dir filament/recipes_sub
```

Both fall back to σ=0 (pure absorption) for any filament without a mixture
calibration and warn -- that baseline is off by dE ~15 mid-ramp, so calibrate the
pair first.

**Real-print finding.** With red+green both calibrated at the print's 0.2 mm layer
height, a printed 11-step sub-layer ramp confirmed the model: baseline (single-
filament absorption only) mean ΔE **11.7** (up to ~20 mid-ramp), σ-model mean ΔE
**4.1** -- at the print-repeatability floor (the pure endpoints alone sit at ΔE ~6
from print-to-print variation).  So single-filament calibration is **not** enough
for sub-layer mixes; you need the pair-ramp σ, but it generalises per filament.
Absorption is also layer-height dependent -- green's red-absorption rose +46 %
from 0.1 to 0.2 mm -- so calibrate at the production layer height.

**`bambu_mix3mf.py`** — the reusable writer that turns parts (each tagged with a
base filament or a mix ratio) into a Bambu color-mix 3MF: de-duplicates distinct
mixes into virtual filament slots, extends every per-filament config array to the
new slot count (handling the dual-nozzle ×2-variant arrays), and preserves the
template's Bambu markers so it loads as a genuine project. The stained-glass model
generator will reuse this to emit per-pane recipes.

### Self-tests (no printer needed)

The analyser can render synthetic backlit-pad photos from a known ground-truth
filament (perspective warp + brightness gradient + noise) and check it recovers
the coefficients — this is how the pipeline was validated before any real print:

```bash
python3 filament/analyze_calibration.py selftest --out-dir /tmp/cal
# -> SELF-TEST PASSED, recovered absorption within <0.01/mm of ground truth
python3 filament/solve_recipe.py selftest
# -> SELF-TEST PASSED, planted recipes recovered to dE<=2
python3 filament/mixture.py selftest
# -> SELF-TEST PASSED, unseen A-C pair + 3-filament mixes predicted to dE<1.5
```

## Roadmap

- [x] Calibration-pad generator (`make_calibration_pad.py`) — one rigid base-plate part
- [x] Photo analyser (`analyze_calibration.py`), synthetic-validated
- [x] Full-layer solver (`solve_recipe.py`): palette → per-pane recipe
      (filament + thickness + predicted colour + ΔE), stacking ≤3 filaments
- [x] Validate on a real printed + photographed pad (red+green, 0.2 mm)
- [x] Sub-layer mixture calibration (`mixture.py fit`) + solver
      (`solve_recipe.py --mode sub-layer`), real-print validated (ΔE 11.7→4.1)
- [ ] Third filament (blue) to test σ generalisation on an unseen pair (A-C)
- [ ] Upgrade Delta-E CIE76 → CIEDE2000; integrate directly into the SVG generator
