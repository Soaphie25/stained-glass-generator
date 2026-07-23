#!/usr/bin/env python3
"""Generate the standard solid backlight images for calibration photography.

Display one of these FULL-SCREEN on a phone/tablet, lay the calibration pad on
top, and photograph it -- one photo per colour (white / red / green / blue). The
solid primaries are what ``analyze_calibration.py`` expects behind the pad.

Do it right: max brightness; turn OFF True Tone / Night Shift / auto-brightness
and any blue-light filter; lock the camera's exposure & white balance (or shoot
RAW) so all four photos share one exposure.

Colours are pure sRGB primaries (white #FFFFFF, red #FF0000, green #00FF00,
blue #0000FF). Solid fills, so the resolution only needs to cover your screen.
"""
import argparse
import os

from PIL import Image

COLOURS = {"white": (255, 255, 255), "red": (255, 0, 0),
           "green": (0, 255, 0), "blue": (0, 0, 255)}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=1440, help="pixels (default 1440)")
    p.add_argument("--height", type=int, default=2560, help="pixels (default 2560)")
    p.add_argument("--out-dir", default=None,
                   help="output folder (default: <script dir>/screens)")
    opts = p.parse_args(argv)

    out_dir = opts.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "filament", "screens")
    os.makedirs(out_dir, exist_ok=True)
    for name, rgb in COLOURS.items():
        Image.new("RGB", (opts.width, opts.height), rgb).save(
            os.path.join(out_dir, name + ".png"))
    print("wrote %d backlight images (%dx%d) to %s"
          % (len(COLOURS), opts.width, opts.height, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
