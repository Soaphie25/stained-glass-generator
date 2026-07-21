#!/usr/bin/env python3
"""A tiny local browser GUI for the filament calibration + LUT toolchain.

No new dependencies: it's a stdlib ``http.server`` that serves ONE page and shells
out to the existing CLI tools (analyze_calibration.py / mixture.py / solve_recipe.py),
then shows their output + preview images.  The browser only builds the commands and
uploads the picked photos (as base64 JSON) -- all the real work stays in the CLI.

    python3 filament/gui.py            # -> http://127.0.0.1:8000  (opens a browser)
    python3 filament/gui.py --port 9000 --no-open

Panels:
  * Calibrate -- 4 file pickers (white / red / green / blue), capture requirements
    shown ABOVE them, and after analysing: absorption, filament class + mix cap,
    per-photo exposure status, and any shot-quality problems (over/under-exposure,
    pad mismatch, fully-absorbed channels).
  * Map       -- the palette overview + pair-coverage.
  * LUT       -- build the colour LUT + gamut, and look a target hex up -> recipe.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # repo root (run CLIs from here)
CALROOT = "filament/calibration"
PAD_LAYOUT = "filament/pad/layout.json"
MIX_LAYOUT = "filament/mixpad/layout.json"
UPLOADS = "filament/calibration/_uploads"

sys.path.insert(0, HERE)
try:
    from analyze_calibration import CAPTURE_TIPS      # noqa: F401 (kept for ref)
except Exception:
    CAPTURE_TIPS = ""

# Bilingual capture requirements (EN + 简体中文) shown above the pickers.
REQ = (
    "CAPTURE REQUIREMENTS  /  拍摄要求\n"
    "• Shoot RAW (DNG/ARW/…).  拍 RAW 格式，数据线性更准确。\n"
    "• Dark room; screen at MAX, FIXED brightness — turn off auto-brightness / "
    "True Tone / Night Shift.\n"
    "  暗室拍摄；屏幕调到最大且固定亮度，关闭自动亮度 / 原彩 / 夜览。\n"
    "• Expose EACH colour on its own so THAT screen's bare windows read ~85–95% "
    "(blue is dimmer → longer shutter).\n"
    "  每种背光单独曝光，使该屏的裸露参考窗读数约 85–95%（蓝色更暗，需更长快门）。\n"
    "• ISO 100; adjust with the SHUTTER, not ISO; white balance doesn't matter "
    "(it cancels in the ratio).\n"
    "  ISO 保持 100；用快门而非 ISO 调整；白平衡无关（比值中会抵消）。\n"
    "• Don't clip the cells or the windows. Lay the pad FLAT, square-on, filling "
    "the frame.\n"
    "  不要让格子或参考窗过曝削顶；标定板铺平、正对镜头并充满画面。\n"
    "• Most filaments: WHITE only. Pale or intense filament: add its RED/GREEN/BLUE "
    "screen(s) to sharpen the hue (each colour screen measures its channel at higher "
    "SNR; used automatically when cleaner).\n"
    "  多数耗材：只需白屏。淡色或强吸收耗材：另加对应红/绿/蓝屏以校正色相"
    "（彩色屏高信噪比测量该通道，拟合更干净时自动采用）。"
)


def sh(args):
    """Run a CLI tool from the repo root; return (rc, stdout, stderr)."""
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _read_cal(name):
    """Read the most-recent result for <name> -- prefer a fresh INVALID (pad
    mismatch) shot so the GUI shows it, without clobbering a good calibration."""
    d = os.path.join(ROOT, CALROOT, name)
    cands = [os.path.join(d, f) for f in
             ("calibration.json", "calibration_INVALID.json")]
    cands = [p for p in cands if os.path.isfile(p)]
    if not cands:
        return None
    return json.load(open(max(cands, key=os.path.getmtime)))


def _filaments():
    """Names of calibrated filaments (folders with a calibration.json)."""
    out = []
    d = os.path.join(ROOT, CALROOT)
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if os.path.isfile(os.path.join(d, n, "calibration.json")):
                out.append(n)
    return out


# --------------------------------------------------------------------------- #
# POST handlers -- each returns a JSON-able dict
# --------------------------------------------------------------------------- #
def do_analyze(data):
    name = (data.get("name") or "filament").strip() or "filament"
    layer = str(data.get("layer") or "0.2")
    layout = data.get("layout") or PAD_LAYOUT
    stage = os.path.join(UPLOADS, name)
    os.makedirs(os.path.join(ROOT, stage), exist_ok=True)
    args = ["python3", "filament/analyze_calibration.py", "analyze",
            "--layout", layout, "--name", name, "--layer-mm", layer,
            "--cal-root", CALROOT]
    used = []
    for scr in ("white", "red", "green", "blue"):
        f = (data.get("files") or {}).get(scr)
        if not f:
            continue
        ext = os.path.splitext(f.get("filename", ""))[1] or ".dng"
        rel = os.path.join(stage, scr + ext)
        with open(os.path.join(ROOT, rel), "wb") as fh:
            fh.write(base64.b64decode(f["b64"].split(",")[-1]))
        args += ["--" + scr, rel]
        used.append(scr)
    if not used:
        return {"ok": False, "stderr": "pick at least a WHITE photo", "cmd": ""}
    mk = data.get("markers") or {}                    # {screen: 4 corners (orig px)}
    if isinstance(mk, list):                          # back-compat: bare list = white
        mk = {"white": mk}
    for scr in ("white", "red", "green", "blue"):
        pts = mk.get(scr)
        if pts and len(pts) == 4:
            flag = "--markers" if scr == "white" else "--markers-" + scr
            args += [flag, ";".join("%.1f,%.1f" % (p[0], p[1]) for p in pts)]
    rc, out, err = sh(args)
    res = {"ok": rc == 0, "cmd": " ".join(args), "stdout": out, "stderr": err,
           "used": used}
    res.update(_cal_payload(name))
    return res


def do_decode(data):
    """Decode an uploaded RAW/image -> a small JPEG the browser can show, for
    hand-picking corner markers.  Returns a data URL + original + display sizes."""
    f = data.get("file")
    if not f:
        return {"ok": False, "stderr": "no file"}
    import io
    import base64 as b64
    import analyze_calibration as A
    from PIL import Image
    os.makedirs(os.path.join(ROOT, UPLOADS), exist_ok=True)
    ext = os.path.splitext(f.get("filename", ""))[1] or ".dng"
    tmp = os.path.join(ROOT, UPLOADS, "_decode" + ext)
    with open(tmp, "wb") as fh:
        fh.write(b64.b64decode(f["b64"].split(",")[-1]))
    try:
        arr = A._load_photo(tmp)
    except Exception as e:
        return {"ok": False, "stderr": "decode failed: %s" % e}
    H, W = arr.shape[:2]
    s = 900.0 / max(W, H)
    im = Image.fromarray(arr).resize((max(1, int(W * s)), max(1, int(H * s))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return {"ok": True, "orig": [W, H], "disp": [im.width, im.height],
            "jpeg": "data:image/jpeg;base64," + b64.b64encode(buf.getvalue()).decode()}


def _cal_payload(name):
    """Display payload for a stored single-filament calibration -- absorption, the
    absorb curve first, class + exposure."""
    cal = _read_cal(name)
    if not cal:
        return {}
    d0 = "%s/%s" % (CALROOT, name)
    imgs = []
    for f in ("absorption.png", "curves.png", "detect_white.png",
              "detect_red.png", "detect_green.png", "detect_blue.png"):
        p = "%s/%s" % (d0, f)
        if os.path.isfile(os.path.join(ROOT, p)):
            imgs.append(p)
    return {"name": name, "primary": cal.get("primary_absorption_per_mm"),
            "reliability": cal.get("reliability"),
            "warnings": cal.get("warnings", []),
            "screens": {s: {"max_ref": d.get("max_ref"),
                            "clip_frac": d.get("clip_frac"),
                            "marker_aspect": d.get("marker_aspect"),
                            "expected_aspect": d.get("expected_aspect")}
                        for s, d in cal.get("screens", {}).items()},
            "images": imgs}


def do_loadcal(data):
    name = (data.get("name") or "").strip()
    p = _cal_payload(name)
    if not p:
        return {"ok": False, "stderr": "no calibration for '%s'" % name}
    p["ok"] = True
    p["loaded"] = True
    return p


def _mixtures():
    """List of directly-calibrated pair names (e.g. 'blue+red')."""
    out = []
    d = os.path.join(ROOT, CALROOT, "mix")
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if os.path.isfile(os.path.join(d, n, "mixture_calibration.json")):
                out.append(n)
    return out


def do_mixtures(data):
    return {"mixtures": _mixtures()}


def do_loadmix(data):
    pair = (data.get("pair") or "").strip()
    j = os.path.join(ROOT, CALROOT, "mix", pair, "mixture_calibration.json")
    if not os.path.isfile(j):
        return {"ok": False, "stderr": "no mixture calibration for '%s'" % pair}
    mc = json.load(open(j))
    img = "%s/mix/%s/mixture_fit.png" % (CALROOT, pair)
    res = {"ok": True, "loaded": True, "pair": pair,
           "filaments": mc.get("filaments"), "sigma": mc.get("sigma")}
    if os.path.isfile(os.path.join(ROOT, img)):
        res["images"] = [img]
    return res


def do_mixfit(data):
    a, b = (data.get("a") or "").strip(), (data.get("b") or "").strip()
    if not a or not b or a == b:
        return {"ok": False, "stderr": "pick two DIFFERENT calibrated filaments "
                                       "请选择两种不同的已校准耗材"}
    f = data.get("file")
    if not f:
        return {"ok": False, "stderr": "pick the mixture-pad WHITE photo "
                                       "请选择混色标定板的白屏照片"}
    stage = os.path.join(UPLOADS, "mix_%s_%s" % (a, b))
    os.makedirs(os.path.join(ROOT, stage), exist_ok=True)
    ext = os.path.splitext(f.get("filename", ""))[1] or ".dng"
    rel = os.path.join(stage, "white" + ext)
    with open(os.path.join(ROOT, rel), "wb") as fh:
        fh.write(base64.b64decode(f["b64"].split(",")[-1]))
    args = ["python3", "filament/mixture.py", "fit", "--layout", MIX_LAYOUT,
            "--white", rel, "--cal-root", CALROOT,
            "--a", "%s=%s/%s/calibration.json" % (a, CALROOT, a),
            "--b", "%s=%s/%s/calibration.json" % (b, CALROOT, b)]
    mk = data.get("markers")
    if mk and len(mk) == 4:
        args += ["--markers", ";".join("%.1f,%.1f" % (p[0], p[1]) for p in mk)]
    rc, out, err = sh(args)
    pair = "+".join(sorted((a, b)))
    res = {"ok": rc == 0, "cmd": " ".join(args), "stdout": out, "stderr": err,
           "pair": pair}
    p = "%s/mix/%s/mixture_fit.png" % (CALROOT, pair)
    if os.path.isfile(os.path.join(ROOT, p)):
        res["images"] = [p]
    return res


def do_filaments(data):
    return {"filaments": _filaments()}


def _attach_pad(res, d):
    import glob as _g
    tmf = sorted(_g.glob(os.path.join(ROOT, d, "*.3mf")))
    if tmf:
        res["download"] = os.path.relpath(tmf[0], ROOT)
    pv = sorted(_g.glob(os.path.join(ROOT, d, "*preview*.png")))
    if pv:
        res["images"] = [os.path.relpath(pv[0], ROOT)]
    return res


def do_genpad(data):
    args = ["python3", "filament/make_calibration_pad.py",
            "--screen-w-mm", str(data.get("w") or 64),
            "--screen-h-mm", str(data.get("h") or 138),
            "--step-mm", str(data.get("step") or 0.2)]
    rc, out, err = sh(args)
    return _attach_pad({"ok": rc == 0, "cmd": " ".join(args), "stdout": out,
                        "stderr": err}, "filament/pad")


def do_genmixpad(data):
    args = ["python3", "filament/make_mixture_pad.py",
            "--screen-w-mm", str(data.get("w") or 64),
            "--screen-h-mm", str(data.get("h") or 138)]
    rc, out, err = sh(args)
    return _attach_pad({"ok": rc == 0, "cmd": " ".join(args), "stdout": out,
                        "stderr": err}, "filament/mixpad")


def do_map(data):
    rc, out, err = sh(["python3", "filament/solve_recipe.py", "map"])
    return {"ok": rc == 0, "stdout": out, "stderr": err,
            "images": ["%s/filament_map.png" % CALROOT]}


def do_lut(data):
    args = ["python3", "filament/solve_recipe.py", "lut"]
    match = (data.get("match") or "").strip()
    if match:
        args += ["--match", match]
    rc, out, err = sh(args)
    return {"ok": rc == 0, "stdout": out, "stderr": err,
            "images": ["%s/gamut.png" % CALROOT]}


POST = {"/analyze": do_analyze, "/decode": do_decode,
        "/mixfit": do_mixfit, "/filaments": do_filaments,
        "/loadcal": do_loadcal, "/mixtures": do_mixtures, "/loadmix": do_loadmix,
        "/genpad": do_genpad, "/genmixpad": do_genmixpad,
        "/map": do_map, "/lut": do_lut}


# --------------------------------------------------------------------------- #
PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Filament calibration & LUT</title><style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#222;background:#f6f6f9}
 header{background:#1f2430;color:#eee;padding:12px 20px;font-size:18px}
 .tabs{display:flex;gap:2px;background:#2a2f3a;padding:0 12px}
 .tabs button{background:none;border:0;color:#aab;padding:10px 18px;cursor:pointer;font-size:14px}
 .tabs button.on{background:#f6f6f9;color:#222;border-radius:6px 6px 0 0}
 .panel{display:none;padding:20px;max-width:1000px;margin:0 auto}
 .panel.on{display:block}
 .req{background:#fff8e6;border:1px solid #f0d98a;border-radius:8px;padding:10px 14px;white-space:pre-wrap;font-size:12.5px;color:#5b4a1a}
 fieldset{border:1px solid #ddd;border-radius:8px;margin:14px 0;padding:12px 14px;background:#fff}
 legend{font-weight:600;padding:0 6px}
 label{display:inline-block;min-width:70px;color:#444}
 .row{margin:6px 0}
 input[type=text],input[type=number]{padding:5px 7px;border:1px solid #ccc;border-radius:6px}
 button.go{background:#2d6cdf;color:#fff;border:0;border-radius:7px;padding:9px 18px;cursor:pointer;font-size:14px}
 button.go:disabled{opacity:.5}
 .cmd{font-family:ui-monospace,monospace;font-size:12px;background:#20242e;color:#b7c6ea;padding:8px 10px;border-radius:6px;overflow-x:auto;white-space:pre-wrap}
 pre.out{font-family:ui-monospace,monospace;font-size:12px;background:#fafafc;border:1px solid #eee;padding:8px 10px;border-radius:6px;overflow-x:auto}
 .warn{background:#fdeaea;border:1px solid #e8a1a1;color:#8a1f1f;border-radius:8px;padding:8px 12px;margin:8px 0}
 .info{background:#eef3fb;border:1px solid #b8cdec;color:#274a7a;border-radius:8px;padding:8px 12px;margin:8px 0}
 .done{background:#e7f6ec;border:1px solid #94cea9;color:#1c5c30;font-weight:600;border-radius:8px;padding:9px 12px;margin:8px 0}
 .ok{background:#eaf7ee;border:1px solid #a7d8b6;color:#1c5c30;border-radius:8px;padding:8px 12px;margin:8px 0}
 .chips span{display:inline-block;padding:4px 8px;margin:3px;border-radius:6px;font-size:12px}
 .exp-ok{background:#e5f5ea;color:#1c5c30}.exp-bad{background:#fdeaea;color:#8a1f1f}
 img.prev{max-width:320px;border:1px solid #ddd;border-radius:6px;margin:6px 8px 0 0;vertical-align:top}
 .sw{display:inline-block;width:64px;height:44px;border:1px solid #bbb;border-radius:5px;vertical-align:middle;margin-right:6px}
</style></head><body>
<header>Filament calibration &amp; colour&nbsp;LUT&nbsp;&nbsp;·&nbsp;&nbsp;耗材校准与色彩查找表</header>
<div class=tabs>
 <button class="on" onclick="tab('cal',this)">1 · Calibrate 单色校准</button>
 <button onclick="tab('mix',this)">2 · Mixture 混色校准</button>
 <button onclick="tab('map',this)">3 · Palette map 色板</button>
 <button onclick="tab('lut',this)">4 · Colour LUT 查找表</button>
</div>

<div id=cal class="panel on">
 <fieldset><legend>① Generate calibration pad&nbsp;·&nbsp;生成标定板</legend>
  <div class=row>screen 屏幕 <input type=number id=cp_w value=64 style="width:60px"> × <input type=number id=cp_h value=138 style="width:60px"> mm
   &nbsp;&nbsp; step 步长 / 层高 <input type=number id=cp_step value=0.2 step=0.1 style="width:60px"> mm</div>
  <div class=row style="color:#777;font-size:12px">Sized to fit inside your phone/tablet screen; the step MUST equal your print layer height.<br>尺寸需适配你的手机/平板屏幕；步长必须等于打印层高。</div>
  <div class=row><button class=go onclick="genPad('/genpad','cp')">Generate 3MF 生成</button> <span id=cp_status></span></div>
  <div id=cp_result></div>
 </fieldset>
 <div class=req>__REQ__</div>
 <fieldset><legend>② Calibrate a filament&nbsp;·&nbsp;校准一种耗材</legend>
  <div class=row><label>name 名称</label><input type=text id=c_name placeholder="e.g. amber 例：琥珀">
    <label style="min-width:110px">layer 层高 (mm)</label><input type=number id=c_layer value="0.2" step="0.1" style="width:70px"></div>
  <div class=row><label>white 白 *</label><input type=file id=f_white accept="image/*,.dng,.arw,.cr2,.nef,.raf"></div>
  <div class=row><label>red 红</label><input type=file id=f_red accept="image/*,.dng,.arw,.cr2,.nef,.raf"><span style="color:#888;font-size:12px;margin-left:8px">try ISO 160 · 1/40s 起始参考</span></div>
  <div class=row><label>green 绿</label><input type=file id=f_green accept="image/*,.dng,.arw,.cr2,.nef,.raf"><span style="color:#888;font-size:12px;margin-left:8px">try ISO 100 · 1/30s 起始参考</span></div>
  <div class=row><label>blue 蓝</label><input type=file id=f_blue accept="image/*,.dng,.arw,.cr2,.nef,.raf"><span style="color:#888;font-size:12px;margin-left:8px">try ISO 125 · 1/40s · max screen brightness 最大亮度</span></div>
  <div class=row style="color:#888;font-size:12px">Suggested START settings only — then fine-tune by the <b>lit channel's</b> R/G/B histogram (~85%), not luma.<br>仅为起始参考——再依据<b>被点亮通道</b>的 R/G/B 直方图（约 85%）微调，而非亮度直方图。</div>
  <div class=row style="color:#777;font-size:12px">White alone works for most filaments. Add the matching colour screen(s) to sharpen the hue of a <b>pale</b> or <b>intense</b> filament — each colour screen measures its channel (R/G/B) at far higher SNR, and is used automatically when it's the cleaner fit.<br>大多数耗材只需白屏。<b>淡色</b>或<b>强吸收</b>耗材可另加对应彩色背光以校正色相——彩色屏能高信噪比地测量该通道，拟合更干净时会自动采用。</div>
  <div class=row style="color:#777;font-size:12px">◈ Pick markers manually per screen — use if auto-detect grabs the wrong squares, e.g. a colour backlight washes out the black corners · 手动标记角点（逐屏）——自动识别选错方块时使用，例如彩色背光下黑角标被冲淡</div>
  <div class=row>
    <button onclick="pickMarkers('cal_white','f_white','mk_area')">◈ white 白</button>
    <button onclick="pickMarkers('cal_red','f_red','mk_area')">◈ red 红</button>
    <button onclick="pickMarkers('cal_green','f_green','mk_area')">◈ green 绿</button>
    <button onclick="pickMarkers('cal_blue','f_blue','mk_area')">◈ blue 蓝</button>
  </div>
  <div id=mk_area></div>
  <div class=row><button class=go id=c_go onclick="analyze()">Analyze 分析</button> <span id=c_status></span></div>
 </fieldset>
 <fieldset><legend>③ View a calibrated filament&nbsp;·&nbsp;查看已校准耗材</legend>
  <select id=cv_name></select> <button class=go onclick="loadCal()">Load 加载</button>
  <button onclick="loadFils()" style="margin-left:6px">↻</button>
  <span style="color:#777;font-size:12px">&nbsp;shows its absorption curve 显示吸收曲线</span>
 </fieldset>
 <div id=c_cmd></div><div id=c_result></div>
</div>

<div id=mix class="panel">
 <div class=req>Calibrates the sub-layer mixing of a PAIR (fits per-filament σ).  校准两种耗材的分层混色（拟合每种耗材的 σ）。
• Print the 11-pad mixture ramp with filament A in slot 1 and B in slot 2, then photograph it over WHITE (same exposure rules as above).
  用 A 放 1 号、B 放 2 号打印 11 格混色渐变板，再在白屏下拍摄（曝光要求同上）。
• Both filaments must already be calibrated (tab 1).  两种耗材都需先在「单色校准」完成。</div>
 <fieldset><legend>① Generate mixture pad&nbsp;·&nbsp;生成混色渐变板</legend>
  <div class=row>screen 屏幕 <input type=number id=mp_w value=64 style="width:60px"> × <input type=number id=mp_h value=138 style="width:60px"> mm</div>
  <div class=row style="color:#777;font-size:12px">11-pad ramp A→B; load filament A in slot 1, B in slot 2 in the slicer.<br>11 格 A→B 渐变；切片时 A 放 1 号槽、B 放 2 号槽。</div>
  <div class=row><button class=go onclick="genPad('/genmixpad','mp')">Generate 3MF 生成</button> <span id=mp_status></span></div>
  <div id=mp_result></div>
 </fieldset>
 <fieldset><legend>② Calibrate a 2-colour mixture&nbsp;·&nbsp;校准双色混合</legend>
  <div class=row><label>A (slot 1)</label><select id=mx_a></select>
    <label style="min-width:90px">B (slot 2)</label><select id=mx_b></select>
    <button onclick="loadFils()" style="margin-left:8px">↻ refresh 刷新</button></div>
  <div class=row><label>white 白 *</label><input type=file id=mx_file accept="image/*,.dng,.arw,.cr2,.nef,.raf"> <span style="color:#777;font-size:12px">mixture-pad photo over white 混色板白屏照片</span></div>
  <div class=row><button onclick="pickMarkers('mix','mx_file','mx_mk_area')">◈ Pick markers manually 手动标记角点</button>
    <span style="color:#777;font-size:12px">if auto-detect fails · 自动识别失败时使用</span></div>
  <div id=mx_mk_area></div>
  <div class=row><button class=go id=mx_go onclick="mixfit()">Fit σ 拟合</button> <span id=mx_status></span></div>
 </fieldset>
 <fieldset><legend>③ View a calibrated mixture&nbsp;·&nbsp;查看已校准混色</legend>
  <select id=mv_pair></select> <button class=go onclick="loadMix()">Load 加载</button>
  <button onclick="loadFils()" style="margin-left:6px">↻</button>
  <span style="color:#777;font-size:12px">&nbsp;shows the measured-vs-predicted ramp 显示实测/预测对比</span>
 </fieldset>
 <div id=mv_result></div>
 <div id=mx_cmd></div><div id=mx_result></div>
</div>

<div id=map class="panel">
 <fieldset><legend>Palette map&nbsp;·&nbsp;色板总览</legend>
  <button class=go onclick="runMap()">Refresh map 刷新</button> <span id=m_status></span>
 </fieldset>
 <div id=m_result></div>
</div>

<div id=lut class="panel">
 <fieldset><legend>Colour LUT &amp; gamut&nbsp;·&nbsp;色彩查找表与色域</legend>
  <button class=go onclick="runLut('')">Build LUT + gamut 生成</button>
  <span style="margin-left:16px">match target 匹配目标 <input type=text id=l_hex placeholder="ff8800" style="width:90px">
  <button class=go onclick="matchHex()">Look up 查找</button></span>
  <span id=l_status></span>
 </fieldset>
 <div id=l_match></div><div id=l_result></div>
</div>

<script>
function tab(id,b){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
 document.getElementById(id).classList.add('on');
 document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));b.classList.add('on');}
function f2b64(f){return new Promise(r=>{const x=new FileReader();x.onload=()=>r({filename:f.name,b64:x.result});x.readAsDataURL(f);});}
function img(p){return '<img class=prev src="/img?path='+encodeURIComponent(p)+'&t='+Date.now()+'">';}
async function post(url,body){const r=await fetch(url,{method:'POST',body:JSON.stringify(body||{})});return r.json();}
async function genPad(url,pfx){const st=document.getElementById(pfx+'_status');st.textContent='running…';
 document.getElementById(pfx+'_result').innerHTML='';
 const body={w:document.getElementById(pfx+'_w').value,h:document.getElementById(pfx+'_h').value};
 const se=document.getElementById(pfx+'_step');if(se)body.step=se.value;
 const r=await post(url,body);st.textContent='';
 let h=r.ok?'<div class=done>✓ pad generated · 已生成标定板</div>':'<div class=warn><pre class=out>'+(r.stderr||r.stdout||'')+'</pre></div>';
 if(r.download){const fn=r.download.split('/').pop();
   h+='<div class=row style="margin-top:6px"><a class=go style="text-decoration:none;padding:8px 14px" href="/file?path='+encodeURIComponent(r.download)+'" download>⬇ Download '+fn+' 下载</a> &nbsp; <span style="color:#555">open in Bambu Studio &amp; print · 用 Bambu Studio 打开并打印</span></div>';}
 if(r.images)r.images.forEach(p=>h+=img(p));
 document.getElementById(pfx+'_result').innerHTML=h;}

const MKC={};   // context ('cal'/'mix') -> {MK, orig, disp, img}
async function pickMarkers(ctx,fileId,areaId){const el=document.getElementById(fileId);
 if(!el.files[0]){alert('pick the photo first 请先选择照片');return;}
 document.getElementById(areaId).innerHTML='decoding… 解码中…';
 const r=await post('/decode',{file:await f2b64(el.files[0])});
 if(!r.ok){document.getElementById(areaId).innerHTML='<div class=warn>'+(r.stderr||'decode failed')+'</div>';return;}
 const st={MK:[],orig:r.orig,disp:r.disp,img:null};MKC[ctx]=st;
 const cid='mkc_'+ctx,iid='mki_'+ctx;
 document.getElementById(areaId).innerHTML='<div class=info>Click the 4 black corner markers (any order) · 点击 4 个黑色角标（顺序任意）</div>'+
  '<canvas id="'+cid+'" style="border:1px solid #ccc;max-width:100%;cursor:crosshair"></canvas> <button onclick="clearMk(&quot;'+ctx+'&quot;)">clear 清除</button> <span id="'+iid+'" style="color:#555"></span>';
 const c=document.getElementById(cid),im=new Image();
 im.onload=()=>{c.width=im.width;c.height=im.height;st.img=im;c.getContext('2d').drawImage(im,0,0);
  c.onclick=(e)=>{const b=c.getBoundingClientRect();const x=(e.clientX-b.left)*c.width/b.width,y=(e.clientY-b.top)*c.height/b.height;
   if(st.MK.length<4){st.MK.push([x,y]);drawMk(ctx);}};};
 im.src=r.jpeg;}
function drawMk(ctx){const st=MKC[ctx],c=document.getElementById('mkc_'+ctx),g=c.getContext('2d');g.drawImage(st.img,0,0);
 st.MK.forEach((p,i)=>{g.fillStyle='#e0f';g.beginPath();g.arc(p[0],p[1],7,0,7);g.fill();g.fillStyle='#fff';g.font='12px sans-serif';g.fillText(i+1,p[0]-3,p[1]+4);});
 document.getElementById('mki_'+ctx).textContent=st.MK.length+'/4'+(st.MK.length===4?' ✓ ready 就绪':'');}
function clearMk(ctx){const st=MKC[ctx];st.MK=[];if(st.img)document.getElementById('mkc_'+ctx).getContext('2d').drawImage(st.img,0,0);document.getElementById('mki_'+ctx).textContent='0/4';}
function mkFor(ctx){const st=MKC[ctx];return (st&&st.MK.length===4)?st.MK.map(p=>[p[0]/st.disp[0]*st.orig[0],p[1]/st.disp[1]*st.orig[1]]):null;}
async function analyze(){
 const btn=document.getElementById('c_go');btn.disabled=true;
 document.getElementById('c_status').textContent='running…';
 document.getElementById('c_result').innerHTML='';document.getElementById('c_cmd').innerHTML='';
 const files={};
 for(const s of ['white','red','green','blue']){const el=document.getElementById('f_'+s);if(el.files[0])files[s]=await f2b64(el.files[0]);}
 const markers={white:mkFor('cal_white'),red:mkFor('cal_red'),green:mkFor('cal_green'),blue:mkFor('cal_blue')};
 const res=await post('/analyze',{name:document.getElementById('c_name').value,layer:document.getElementById('c_layer').value,files,markers});
 btn.disabled=false;document.getElementById('c_status').textContent='';
 if(res.cmd)document.getElementById('c_cmd').innerHTML='<div class=cmd>'+res.cmd+'</div>';
 renderCal(res,'c_result');
}
async function loadCal(){const n=document.getElementById('cv_name').value;if(!n)return;
 document.getElementById('c_cmd').innerHTML='';const r=await post('/loadcal',{name:n});renderCal(r,'c_result');}
function renderCal(res,target){
 let h='';
 const badpad=(res.warnings||[]).some(w=>/PAD MISMATCH/i.test(w));
 const nm=res.name||'filament';
 if(!res.ok){h+='<div class=warn><b>failed 失败:</b><br><pre class=out>'+(res.stderr||'')+'</pre></div>';}
 else if(badpad){h+='<div class=warn><b>✗ INVALID — pad/shot doesn\\'t match the layout · 无效：标定板/拍摄与布局不匹配</b><br>'+
   'Most likely the pad isn\\'t lying FLAT or the shot is TILTED, so the cells were sampled in the wrong spots (numbers below are bogus). Press the pad flat against the screen, shoot square-on, and re-analyse. Only if it persists is the pad a different make_calibration_pad version (reprint).<br>'+
   '多半是标定板没铺平或拍摄倾斜，导致采样位置错误（下方数值无效）。请把板压平贴屏、正对镜头重拍后再分析。若仍不匹配，才是标定板版本不同（需重打）。</div>';}
 else if(res.primary){h+='<div class=done>'+(res.loaded?'📂 loaded · 已加载 ':'✓ calibrated · 校准成功 ')+'&nbsp;→ filament/calibration/'+nm+'/</div>';}
 const rel=res.reliability;
 if(res.primary&&!badpad){h+='<div class=ok><b>absorption /mm · 吸收系数</b> &nbsp; R '+res.primary.R+' &nbsp; G '+res.primary.G+' &nbsp; B '+res.primary.B+'</div>';}
 if(rel){const CLS={'intense':'INTENSE 强吸收','normal-transparent':'normal 普通透明'};
   h+='<p><b>class 类别:</b> '+(CLS[rel.filament_class]||rel.filament_class);
   if(rel.mix_advice)h+='<br><span style="color:#b33">⚠ '+rel.mix_advice+'<br>该耗材吸收极强，混色占比请低于上限，否则会盖过其它颜色。</span>';h+='</p>';}
 // per-photo exposure status (bilingual)
 if(res.screens){h+='<div class=chips><b>exposure 曝光&nbsp;</b>';
   for(const s in res.screens){const d=res.screens[s];let bad='',cls='exp-ok';
     if(d.max_ref!=null&&d.max_ref<0.75){bad=' TOO DIM 偏暗';cls='exp-bad';}
     else if(d.clip_frac!=null&&d.clip_frac>0.12){bad=' CLIPPED 过曝';cls='exp-bad';}
     if(d.marker_aspect&&d.expected_aspect&&Math.abs(d.marker_aspect-d.expected_aspect)/d.expected_aspect>0.02){bad+=' PAD-MISMATCH 板不匹配';cls='exp-bad';}
     h+='<span class="'+cls+'">'+s+' '+Math.round((d.max_ref||0)*100)+'%'+(bad||' ✓')+'</span>';}
   h+='</div>';}
 // split EXPECTED skips (colour screen washed out -> ignored, fine) from real problems
 const ws=res.warnings||[],skips=ws.filter(w=>/SKIPPED/i.test(w)),probs=ws.filter(w=>!/SKIPPED/i.test(w));
 if(skips.length){h+='<div class=info><b>skipped shots (expected, ignored) · 已跳过的照片（正常，可忽略）:</b><ul>';
   skips.forEach(w=>{const g=cnGloss(w);h+='<li>'+w.replace(/SKIPPED.*?:/,'skipped:')+(g?'<br><span style="color:#456">〔'+g+'〕</span>':'')+'</li>';});
   h+='</ul>A colour screen can\\'t be read when the filament passes/blocks that colour — the white shot covers it.<br>当耗材透过/挡住该颜色时其背光无法读取标记，白屏已足够。</div>';}
 if(probs.length){h+='<div class=warn><b>shot-quality problems 拍摄问题:</b><ul>';
   probs.forEach(w=>{const g=cnGloss(w);h+='<li>'+w+(g?'<br><span style="color:#a33">〔'+g+'〕</span>':'')+'</li>';});h+='</ul></div>';}
 if(res.images)res.images.forEach(p=>h+=img(p));
 document.getElementById(target).innerHTML=h;
}
async function loadMix(){const p=document.getElementById('mv_pair').value;if(!p)return;
 const r=await post('/loadmix',{pair:p});let h='';
 if(!r.ok){h='<div class=warn><pre class=out>'+(r.stderr||'')+'</pre></div>';}
 else{h='<div class=done>📂 loaded · 已加载 &nbsp;'+r.pair+'</div>';
   if(r.images)r.images.forEach(x=>h+=img(x));
   if(!r.images)h+='<div class=info>no fit image saved (re-run the fit to regenerate) · 无拟合图，请重新拟合生成</div>';}
 document.getElementById('mv_result').innerHTML=h;}
const WARN_CN=[['SKIPPED','已跳过该照片：未找到标记点（该背光下标记被冲淡，或标定板出框/过度倾斜）。已用其它可用照片完成校准；普通耗材只需白屏即可'],
 ['OVER-EXPOSED','过曝：参考窗被削顶，请缩短快门/降低亮度'],
 ['UNDER-EXPOSED','曝光不足：背光太暗，请调亮屏幕/延长快门'],
 ['TOO DIM','该屏偏暗：延长其快门（不要加 ISO）'],
 ['CLIPPED','该屏过曝：缩短其快门'],
 ['PAD MISMATCH','标定板与布局不匹配：版本不对或未铺平，请用当前板重打或摆正重拍'],
 ['FULLY ABSORBED','该通道完全吸收：数值为下界（正常，混色中读数≈0）'],
 ['NOISY','拟合噪声过大：疑似 ISO 高/抖动/反光，请暗室、低 ISO、稳定拍摄']];
function cnGloss(w){const u=w.toUpperCase();for(const [k,v] of WARN_CN)if(u.includes(k))return v;return '';}
async function loadFils(){const r=await post('/filaments',{});const fs=r.filaments||[];
 for(const id of ['mx_a','mx_b','cv_name']){const s=document.getElementById(id);if(!s)continue;const cur=s.value;
   s.innerHTML=fs.map(f=>'<option'+(f==cur?' selected':'')+'>'+f+'</option>').join('');}
 if(fs.length>1&&document.getElementById('mx_b').selectedIndex==document.getElementById('mx_a').selectedIndex)document.getElementById('mx_b').selectedIndex=1;
 const rm=await post('/mixtures',{});const ms=rm.mixtures||[];const mv=document.getElementById('mv_pair');
 if(mv){const cur=mv.value;mv.innerHTML=ms.map(m=>'<option'+(m==cur?' selected':'')+'>'+m+'</option>').join('');}}
async function mixfit(){const btn=document.getElementById('mx_go');btn.disabled=true;
 document.getElementById('mx_status').textContent='running…';
 document.getElementById('mx_result').innerHTML='';document.getElementById('mx_cmd').innerHTML='';
 const el=document.getElementById('mx_file');const file=el.files[0]?await f2b64(el.files[0]):null;
 const res=await post('/mixfit',{a:document.getElementById('mx_a').value,b:document.getElementById('mx_b').value,file,markers:mkFor('mix')});
 btn.disabled=false;document.getElementById('mx_status').textContent='';
 if(res.cmd)document.getElementById('mx_cmd').innerHTML='<div class=cmd>'+res.cmd+'</div>';
 let h='';
 if(!res.ok){h+='<div class=warn><b>failed 失败:</b><br><pre class=out>'+(res.stderr||'')+'</pre></div>';}
 else{h+='<div class=done>✓ σ fitted · σ 拟合完成 &nbsp;→ filament/calibration/mix/'+res.pair+'/</div>';
   // pull the model vs baseline dE summary lines
   const t=res.stdout||'',m=t.match(/model.*dE.*/i),b=t.match(/baseline.*dE.*/i);
   if(m||b)h+='<div class=ok>'+[m,b].filter(Boolean).map(x=>x[0]).join('<br>')+'<br><span style="color:#555">lower model ΔE = better; the pair is now a direct posterior in the LUT. 模型 ΔE 越低越好，该组合已作为直接后验进入查找表。</span></div>';
   if(res.images)res.images.forEach(p=>h+=img(p));
   h+='<details><summary>raw table 原始数据</summary><pre class=out>'+t+'</pre></details>';}
 document.getElementById('mx_result').innerHTML=h;}
async function runMap(){document.getElementById('m_status').textContent='running…';
 const r=await post('/map',{});document.getElementById('m_status').textContent='';
 let h='<pre class=out>'+(r.stdout||r.stderr||'')+'</pre>';if(r.images)r.images.forEach(p=>h+=img(p));
 document.getElementById('m_result').innerHTML=h;}
async function runLut(m){document.getElementById('l_status').textContent='running…';
 const r=await post('/lut',{match:m});document.getElementById('l_status').textContent='';
 let h='<pre class=out>'+(r.stdout||r.stderr||'')+'</pre>';if(r.images)r.images.forEach(p=>h+=img(p));
 document.getElementById('l_result').innerHTML=h;}
async function matchHex(){const hx=document.getElementById('l_hex').value.trim().replace('#','');
 if(!hx)return;document.getElementById('l_status').textContent='running…';
 const r=await post('/lut',{match:hx});document.getElementById('l_status').textContent='';
 // parse "#target #predict  dE  recipe"
 let h='';const re=/#([0-9a-fA-F]{6})\\s+#([0-9a-fA-F]{6})\\s+([\\d.]+)\\s+(.+)/g,txt=r.stdout||'';let m2;
 while((m2=re.exec(txt))){h+='<div style="margin:8px 0"><span class=sw style="background:#'+m2[1]+'"></span>'+
   '<span class=sw style="background:#'+m2[2]+'"></span> target #'+m2[1]+' → #'+m2[2]+' &nbsp; ΔE '+m2[3]+' &nbsp; <b>'+m2[4]+'</b></div>';}
 document.getElementById('l_match').innerHTML=h||'<pre class=out>'+txt+'</pre>';}
loadFils();   // populate mixture dropdowns on load
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       PAGE.replace("__REQ__", REQ).encode())
        elif u.path == "/img":
            rel = parse_qs(u.query).get("path", [""])[0]
            full = os.path.normpath(os.path.join(ROOT, rel))
            base = os.path.join(ROOT, "filament")
            if full.startswith(base) and os.path.isfile(full):
                self._send(200, "image/png", open(full, "rb").read())
            else:
                self._send(404, "text/plain", b"not found")
        elif u.path == "/file":                      # download (e.g. the .3mf)
            rel = parse_qs(u.query).get("path", [""])[0]
            full = os.path.normpath(os.path.join(ROOT, rel))
            base = os.path.join(ROOT, "filament")
            if full.startswith(base) and os.path.isfile(full):
                body = open(full, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition",
                                 'attachment; filename="%s"' % os.path.basename(full))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        fn = POST.get(u.path)
        if not fn:
            self._send(404, "text/plain", b"not found")
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}
        try:
            out = fn(data)
        except Exception as e:                       # never 500 silently
            out = {"ok": False, "stderr": "server error: %r" % e}
        self._send(200, "application/json", json.dumps(out).encode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    opts = ap.parse_args(argv)
    url = "http://127.0.0.1:%d" % opts.port
    srv = HTTPServer(("127.0.0.1", opts.port), Handler)
    sys.stderr.write("filament GUI on %s   (Ctrl-C to stop)\n" % url)
    if not opts.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\nbye\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
