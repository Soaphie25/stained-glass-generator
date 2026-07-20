#!/usr/bin/env python3
"""Browser GUI for the stained-glass 3MF generator (separate from the filament
calibration GUI).  Workflow:

  1. Check the calibration LUT/gamut.  If nothing is calibrated yet, it points you
     to the filament GUI (``python3 filament/gui.py``) to calibrate first.
  2. Pick a JPEG/PNG.
  3. Vectorise it to stained-glass panes (a few key options + a "More" panel for
     all of them) and show the GAMUT preview -- each pane painted the printable
     recipe colour, i.e. what the panel will actually look like.
  4. When happy, generate the Bambu colour-mix 3MF (options: depth + size, aspect
     kept) and download it.

Stdlib http.server, no new deps; shells out to png_to_stained_glass_svg.py and
imports svg_to_3mf in-process.  Bilingual EN + 简体中文.

    python3 filament/glass_gui.py            # -> http://127.0.0.1:8010
"""
import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))     # repo root (this file lives here)
CALROOT = "filament/calibration"
WORK = "filament/calibration/_glass"          # gitignored scratch (under calibration/)
sys.path.insert(0, os.path.join(ROOT, "filament"))    # import svg_to_3mf/solve_recipe


def sh(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _filaments():
    d = os.path.join(ROOT, CALROOT)
    return sorted(n for n in (os.listdir(d) if os.path.isdir(d) else [])
                  if os.path.isfile(os.path.join(d, n, "calibration.json")))


# --------------------------------------------------------------------------- #
def do_lutstatus(data):
    fils = _filaments()
    gam = "%s/gamut.png" % CALROOT
    return {"filaments": fils, "n": len(fils),
            "gamut": gam if os.path.isfile(os.path.join(ROOT, gam)) else None,
            "ready": len(fils) >= 1}


# allowed SVG-generator options (key -> --key), passed through from the browser
_SVG_OPTS = ("max-size-mm", "px-mm", "num-colors", "black-block-mm",
             "line-width", "line-width-scale", "lum-threshold", "alpha-min",
             "fit-tolerance", "simplify-tolerance", "smooth-tolerance",
             "min-fragment-area", "color-merge-tol", "min-line-width")


def do_convert(data):
    work = os.path.join(ROOT, WORK)
    os.makedirs(work, exist_ok=True)
    f = data.get("image")
    if not f:
        return {"ok": False, "stderr": "pick an image 请选择图片"}
    ext = os.path.splitext(f.get("filename", ""))[1] or ".png"
    img = os.path.join(WORK, "input" + ext)
    with open(os.path.join(ROOT, img), "wb") as fh:
        fh.write(base64.b64decode(f["b64"].split(",")[-1]))
    fragdir = os.path.join(WORK, "frag")
    args = ["python3", "png_to_stained_glass_svg.py", img,
            "--fragments-dir", fragdir, "--fragment-color", "original"]
    for k in _SVG_OPTS:
        v = (data.get("svg") or {}).get(k)
        if v not in (None, "", "auto-default"):
            args += ["--" + k, str(v)]
    rc, out, err = sh(args)
    if rc != 0:
        return {"ok": False, "cmd": " ".join(args), "stderr": err or out}
    return {"ok": True, "cmd": " ".join(args), "frag_dir": fragdir,
            "colors": _frag_colors(fragdir)}


def _frag_colors(fragdir):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, fragdir, "color_*.svg"))):
        import re
        m = re.search(r"color_\d+_([0-9a-fA-F]{6})", os.path.basename(f))
        if m:
            out.append(m.group(1))
    return out


def _map(data):
    import svg_to_3mf as V
    fragdir = os.path.join(WORK, "frag")
    return V, V.map_recipes(fragdir, CALROOT,
                            thickness=float(data.get("depth") or 1.6),
                            max_delta=float(data.get("max_delta") or 20),
                            num_colors=(int(data["colors"]) if data.get("colors")
                                        else None),
                            max_size_mm=(float(data["size"]) if data.get("size")
                                         else None))


def _table(m):
    rows = []
    for it in sorted(m["items"], key=lambda x: -x["area"]):
        rec = m["rec_cache"][m["targets"][it["hex"]]]
        rows.append({"hex": it["hex"], "predicted": rec["predicted_hex"],
                     "recipe": " / ".join("%s %d%%" % (n, f) for n, f
                                          in zip(rec["filaments"], rec["fracs_pct"])),
                     "dE": round(rec["delta_e"], 1),
                     "out": rec["delta_e"] > m["max_delta"]})
    return rows


def do_preview(data):
    try:
        V, m = _map(data)
    except SystemExit as e:
        return {"ok": False, "stderr": str(e)}
    prev = os.path.join(WORK, "preview.png")
    V.render_preview(m, os.path.join(ROOT, prev))
    return {"ok": True, "image": prev, "table": _table(m),
            "dims": [round(m["W"]), round(m["H"])],
            "n_out": sum(1 for r in _table(m) if r["out"])}


def do_gen3mf(data):
    try:
        V, m = _map(data)
    except SystemExit as e:
        return {"ok": False, "stderr": str(e)}
    out = os.path.join(WORK, "panel.3mf")
    import svg_to_3mf as V2
    V2.build_3mf(os.path.join(WORK, "frag"), CALROOT, os.path.join(ROOT, out),
                 thickness=float(data.get("depth") or 1.6),
                 max_delta=float(data.get("max_delta") or 20),
                 num_colors=(int(data["colors"]) if data.get("colors") else None),
                 max_size_mm=(float(data["size"]) if data.get("size") else None))
    return {"ok": True, "download": out, "image": os.path.join(WORK, "panel_preview.png"),
            "table": _table(m), "dims": [round(m["W"]), round(m["H"])]}


POST = {"/lutstatus": do_lutstatus, "/convert": do_convert,
        "/preview": do_preview, "/gen3mf": do_gen3mf}


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>Stained-glass 3MF</title><style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f6f9;color:#222}
 header{background:#241f30;color:#eee;padding:12px 20px;font-size:18px}
 .wrap{max-width:1000px;margin:0 auto;padding:18px}
 fieldset{border:1px solid #ddd;border-radius:8px;margin:12px 0;padding:12px 14px;background:#fff}
 legend{font-weight:600;padding:0 6px}
 .row{margin:6px 0}label{color:#444}
 input[type=text],input[type=number],select{padding:5px 7px;border:1px solid #ccc;border-radius:6px}
 button.go{background:#7a3db2;color:#fff;border:0;border-radius:7px;padding:9px 18px;cursor:pointer}
 button.go:disabled{opacity:.5}
 a.dl{background:#2d6cdf;color:#fff;border-radius:7px;padding:9px 16px;text-decoration:none}
 .warn{background:#fdeaea;border:1px solid #e8a1a1;color:#8a1f1f;border-radius:8px;padding:8px 12px;margin:8px 0}
 .info{background:#eef3fb;border:1px solid #b8cdec;color:#274a7a;border-radius:8px;padding:8px 12px;margin:8px 0}
 .ok{background:#e7f6ec;border:1px solid #94cea9;color:#1c5c30;border-radius:8px;padding:8px 12px;margin:8px 0}
 img.prev{max-width:360px;border:1px solid #ccc;border-radius:6px;background:#111}
 table{border-collapse:collapse;font-size:12.5px}td,th{padding:3px 8px;text-align:left}
 .sw{display:inline-block;width:26px;height:16px;border:1px solid #999;border-radius:3px;vertical-align:middle}
 details summary{cursor:pointer;color:#555}
 .grid{display:flex;gap:20px;flex-wrap:wrap}
</style></head><body>
<header>Stained-glass → 3MF&nbsp;&nbsp;·&nbsp;&nbsp;彩色玻璃 → 3MF</header>
<div class=wrap>

<fieldset><legend>① Palette / gamut&nbsp;·&nbsp;色板 / 色域</legend>
 <div id=lut>checking… 检查中…</div>
</fieldset>

<fieldset><legend>② Image → panes&nbsp;·&nbsp;图片 → 玻璃块</legend>
 <div class=row><input type=file id=img accept="image/*"></div>
 <div class=row>size 尺寸(最长边) <input type=number id=o_size value=200 style="width:70px"> mm
   &nbsp; colours 颜色数 <input type=number id=o_colors placeholder="all 全部" style="width:70px">
   &nbsp; leading 铅线 <input type=text id=o_linewidth value="tier" style="width:70px"></div>
 <details><summary>More options 更多选项</summary>
  <div class=row style="font-size:12px;color:#555">px-mm <input type=number id=o_pxmm value=0.4 step=0.1 style="width:60px">
   black-block-mm <input type=number id=o_blackblock value=3 step=0.5 style="width:60px">
   lum-threshold <input type=number id=o_lum value=90 style="width:60px">
   min-fragment-area <input type=number id=o_minfrag value=32 style="width:60px">
   color-merge-tol <input type=number id=o_mergetol value=8 style="width:60px"></div>
 </details>
 <div class=row><button class=go onclick="convert()">Convert 转换</button> <span id=c_status></span></div>
</fieldset>

<fieldset><legend>③ Gamut preview&nbsp;·&nbsp;色域预览 <span style="font-weight:400;color:#777;font-size:12px">— printable colours 实际可打印颜色</span></legend>
 <div class=row>depth 厚度 <input type=number id=o_depth value=1.6 step=0.2 style="width:60px"> mm
   &nbsp; max-ΔE (allow 3-mix) 三色阈值 <input type=number id=o_maxdelta value=20 style="width:60px">
   &nbsp; <button class=go onclick="preview()">Update preview 更新预览</button> <span id=p_status></span></div>
 <div class=grid><div id=p_img></div><div id=p_table></div></div>
</fieldset>

<fieldset><legend>④ Generate 3MF&nbsp;·&nbsp;生成 3MF</legend>
 <div class=row><button class=go onclick="gen()">Generate 生成 3MF</button> <span id=g_status></span></div>
 <div id=g_result></div>
</fieldset>
</div>

<script>
function $(i){return document.getElementById(i);}
function f2b64(f){return new Promise(r=>{const x=new FileReader();x.onload=()=>r({filename:f.name,b64:x.result});x.readAsDataURL(f);});}
function img(p){return '<img class=prev src="/img?path='+encodeURIComponent(p)+'&t='+Date.now()+'">';}
async function post(u,b){const r=await fetch(u,{method:'POST',body:JSON.stringify(b||{})});return r.json();}
function svgOpts(){return {'max-size-mm':$('o_size').value,'num-colors':$('o_colors').value,
 'line-width':$('o_linewidth').value,'px-mm':$('o_pxmm').value,'black-block-mm':$('o_blackblock').value,
 'lum-threshold':$('o_lum').value,'min-fragment-area':$('o_minfrag').value,'color-merge-tol':$('o_mergetol').value};}
function params(){return {depth:$('o_depth').value,size:$('o_size').value,colors:$('o_colors').value,max_delta:$('o_maxdelta').value};}
async function lut(){const r=await post('/lutstatus',{});let h='';
 if(!r.ready){h='<div class=warn><b>No calibrated filaments yet · 尚无已校准耗材</b><br>Calibrate filaments first in the filament GUI: run <code>python3 filament/gui.py</code> · 请先用耗材 GUI 校准：运行 <code>python3 filament/gui.py</code></div>';}
 else{h='<div class=ok>'+r.n+' filament(s) 已校准: '+r.filaments.join(', ')+'</div>';if(r.gamut)h+=img(r.gamut);}
 $('lut').innerHTML=h;}
async function convert(){const el=$('img');if(!el.files[0]){alert('pick an image 请选择图片');return;}
 $('c_status').textContent='vectorising… 矢量化中…';
 const image=await f2b64(el.files[0]);
 const r=await post('/convert',{image,svg:svgOpts()});$('c_status').textContent='';
 if(!r.ok){$('c_status').innerHTML='<span style="color:#a33">'+(r.stderr||'').slice(-200)+'</span>';return;}
 $('c_status').innerHTML='<span style="color:#2a7">✓ '+(r.colors||[]).length+' colours 种颜色</span>';
 preview();}
async function preview(){$('p_status').textContent='mapping… 映射中…';
 const r=await post('/preview',params());$('p_status').textContent='';
 if(!r.ok){$('p_img').innerHTML='<div class=warn>'+(r.stderr||'')+'</div>';return;}
 $('p_img').innerHTML=img(r.image)+'<div style="color:#777;font-size:12px">'+r.dims[0]+'×'+r.dims[1]+' mm · '+r.n_out+' out of gamut 超出色域</div>';
 let t='<table><tr><th>colour</th><th>→ print</th><th>recipe 配方</th><th>ΔE</th></tr>';
 r.table.forEach(x=>{t+='<tr><td><span class=sw style="background:#'+x.hex+'"></span></td><td><span class=sw style="background:#'+x.predicted+'"></span></td><td>'+x.recipe+'</td><td'+(x.out?' style="color:#c33"':'')+'>'+x.dE+(x.out?' ⚠':'')+'</td></tr>';});
 $('p_table').innerHTML=t+'</table>';}
async function gen(){$('g_status').textContent='building… 生成中…';
 const r=await post('/gen3mf',params());$('g_status').textContent='';
 if(!r.ok){$('g_result').innerHTML='<div class=warn><pre>'+(r.stderr||'')+'</pre></div>';return;}
 let h='<div class=ok>✓ panel '+r.dims[0]+'×'+r.dims[1]+' mm 已生成</div>';
 h+='<div class=row><a class=dl href="/file?path='+encodeURIComponent(r.download)+'" download>⬇ Download panel.3mf 下载</a> &nbsp; open in Bambu Studio 用 Bambu Studio 打开</div>';
 h+=img(r.image);
 $('g_result').innerHTML=h;}
lut();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path in ("/img", "/file"):
            rel = parse_qs(u.query).get("path", [""])[0]
            full = os.path.normpath(os.path.join(ROOT, rel))
            if full.startswith(os.path.join(ROOT, "filament")) and os.path.isfile(full):
                if u.path == "/file":
                    self._send(200, "application/octet-stream", open(full, "rb").read(),
                               {"Content-Disposition": 'attachment; filename="%s"'
                                % os.path.basename(full)})
                else:
                    self._send(200, "image/png", open(full, "rb").read())
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
        except Exception as e:
            import traceback
            out = {"ok": False, "stderr": "server error: %s\n%s"
                   % (e, traceback.format_exc()[-500:])}
        self._send(200, "application/json", json.dumps(out).encode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--no-open", action="store_true")
    opts = ap.parse_args(argv)
    url = "http://127.0.0.1:%d" % opts.port
    srv = HTTPServer(("127.0.0.1", opts.port), Handler)
    sys.stderr.write("stained-glass 3MF GUI on %s\n" % url)
    if not opts.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\nbye\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
