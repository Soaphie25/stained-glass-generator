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
is both the backlight and a programmable RGB source). Take one photo over each of
**white, red, green, blue** → 4 photos per filament. The single-colour screens
isolate one channel each; white calibrates all three.

Practical must-dos: turn OFF True Tone / Night Shift / auto-brightness, lock camera
exposure & white balance (or shoot RAW), avoid glare. The bare screen showing
through the gaps between cells is used to normalise out exposure/brightness.

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

### Self-test (no printer needed)

The analyser can render synthetic backlit-pad photos from a known ground-truth
filament (perspective warp + brightness gradient + noise) and check it recovers
the coefficients — this is how the pipeline was validated before any real print:

```bash
python3 filament/analyze_calibration.py selftest --out-dir /tmp/cal
# -> SELF-TEST PASSED, recovered absorption within <0.01/mm of ground truth
```

## Roadmap

- [x] Calibration-pad generator (`make_calibration_pad.py`) — one rigid base-plate part
- [x] Photo analyser (`analyze_calibration.py`), synthetic-validated
- [ ] Validate on a real printed + photographed pad
- [ ] Mixture solver: SVG palette (`color_NN_<hex>`) → per-pane recipe
      (filament + thickness + predicted colour + ΔE), stacking ≤3 filaments
