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
             "line-width", "line-width-scale", "tier-thin", "tier-bold",
             "lum-threshold", "alpha-min", "fit-tolerance", "simplify-tolerance",
             "smooth-tolerance", "min-fragment-area", "color-merge-tol",
             "min-line-width", "link-angle", "link-width-ratio")
_SVG_FLAGS = ("smooth-curves", "merge-leading")


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
            "--fragments-dir", fragdir, "--leading-svg", os.path.join(WORK, "leading.svg"),
            "--fragment-color", "original"]
    for k in _SVG_OPTS:
        v = (data.get("svg") or {}).get(k)
        if v not in (None, "", "auto-default"):
            args += ["--" + k, str(v)]
    for k in _SVG_FLAGS:                              # boolean flags (no value)
        if (data.get("flags") or {}).get(k):
            args += ["--" + k]
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


def do_gamut(data):
    os.makedirs(os.path.join(ROOT, WORK), exist_ok=True)
    sel = data.get("filaments") or []
    args = ["python3", "filament/solve_recipe.py", "lut", "--cal-root", CALROOT,
            "--out-dir", WORK]
    if sel:
        args += ["--filaments", ",".join(sel)]
    rc, out, err = sh(args)
    return {"ok": rc == 0, "gamut": "%s/gamut.png" % WORK if rc == 0 else None,
            "stderr": err, "selected": sel}


def _map(data):
    import svg_to_3mf as V
    fragdir = os.path.join(WORK, "frag")
    return V, V.map_recipes(fragdir, CALROOT,
                            thickness=float(data.get("depth") or 1.6),
                            max_delta=float(data.get("max_delta") or 20),
                            num_colors=(int(data["colors"]) if data.get("colors")
                                        else None),
                            max_size_mm=(float(data["size"]) if data.get("size")
                                         else None),
                            filaments=(data.get("filaments") or None))


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
    lead = os.path.join(ROOT, WORK, "leading.svg")
    V.render_preview(m, os.path.join(ROOT, prev), leading_svg=lead)
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
                 max_size_mm=(float(data["size"]) if data.get("size") else None),
                 filaments=(data.get("filaments") or None))
    return {"ok": True, "download": out, "image": os.path.join(WORK, "panel_preview.png"),
            "table": _table(m), "dims": [round(m["W"]), round(m["H"])]}


def do_suggest(data):
    """Rank filament subsets that best cover the picked image (area-weighted mean
    Delta-E over its colours), so you pick the best N of your calibrated filaments."""
    f = data.get("image")
    if not f:
        return {"ok": False, "stderr": "pick an image in step ② first "
                                       "请先在②选择图片"}
    work = os.path.join(ROOT, WORK)
    os.makedirs(work, exist_ok=True)
    ext = os.path.splitext(f.get("filename", ""))[1] or ".png"
    p = os.path.join(work, "_suggest" + ext)
    with open(p, "wb") as fh:
        fh.write(base64.b64decode(f["b64"].split(",")[-1]))
    import solve_recipe as SR
    names = _filaments()
    slots = int(data.get("slots") or 4)
    if len(names) < slots:
        return {"ok": False, "stderr": "need >= %d calibrated filaments (have %d)"
                % (slots, len(names))}
    mixcals = glob.glob(os.path.join(ROOT, CALROOT, "mix", "*",
                                     "mixture_calibration.json"))
    sigma, pair = SR.load_sigma(mixcals)
    pool = [SR.load_filament(n, os.path.join(ROOT, CALROOT, n, "calibration.json"))
            for n in names]
    hexes, wts = SR.image_colors(p, 24)
    ranked = SR.suggest_palette(hexes, wts, pool, sigma, pair, slots, 3.0,
                                float(data.get("max_delta") or 20))
    return {"ok": True, "ranked": ranked[:6], "n_colors": len(hexes), "slots": slots}


POST = {"/lutstatus": do_lutstatus, "/gamut": do_gamut, "/convert": do_convert,
        "/preview": do_preview, "/gen3mf": do_gen3mf, "/suggest": do_suggest}


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

<fieldset><legend>① Filaments &amp; gamut&nbsp;·&nbsp;耗材与色域</legend>
 <div id=lut>checking… 检查中…</div>
 <div id=gamut style="margin-top:8px"></div>
 <div id=sug_result style="margin-top:6px"></div>
</fieldset>

<fieldset><legend>② Image → panes&nbsp;·&nbsp;图片 → 玻璃块</legend>
 <div class=row><input type=file id=img accept="image/*"></div>
 <div class=row>size 尺寸(最长边) <input type=number id=o_size value=200 style="width:70px"> mm
   &nbsp; colours 颜色数 <input type=number id=o_colors placeholder="all 全部" style="width:70px"></div>
 <div class=row>leading 铅线
   <select id=o_leadmode onchange="leadUI()">
     <option value="tier">tier 分级</option><option value="auto">auto 自动</option><option value="width">width= 固定宽度</option></select>
   <span id=lead_tier>&nbsp; bold 粗 <input type=number id=o_tierbold value=0 step=0.1 style="width:55px"> thin 细 <input type=number id=o_tierthin value=0 step=0.1 style="width:55px"> mm <span style="color:#999;font-size:11px">(0 = auto 默认)</span></span>
   <span id=lead_width style="display:none">&nbsp; <input type=number id=o_leadwidth value=1 step=0.1 style="width:55px"> mm</span></div>
 <details><summary>More options 更多选项</summary>
  <div class=row style="font-size:12px;color:#555">px-mm <input type=number id=o_pxmm value=0.4 step=0.1 style="width:60px">
   black-block-mm <input type=number id=o_blackblock value=3 step=0.5 style="width:60px">
   lum-threshold <input type=number id=o_lum value=90 style="width:60px">
   min-fragment-area <input type=number id=o_minfrag value=32 style="width:60px">
   color-merge-tol <input type=number id=o_mergetol value=8 style="width:60px">
   alpha-min <input type=number id=o_alpha value=128 style="width:60px">
   link-angle <input type=number id=o_linkangle value=35 style="width:60px">
   link-width-ratio <input type=number id=o_linkratio value=1.7 step=0.1 style="width:60px">
   &nbsp; <label><input type=checkbox id=o_smooth> smooth-curves 平滑曲线铅线</label>
   <label><input type=checkbox id=o_mergelead> merge-leading 合并铅线</label></div>
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
function leadUI(){const m=$('o_leadmode').value;
 $('lead_tier').style.display=(m==='tier')?'inline':'none';
 $('lead_width').style.display=(m==='width')?'inline':'none';}
function svgOpts(){const m=$('o_leadmode').value;
 const o={'max-size-mm':$('o_size').value,'num-colors':$('o_colors').value,
  'line-width':(m==='width'?$('o_leadwidth').value:m),'px-mm':$('o_pxmm').value,
  'black-block-mm':$('o_blackblock').value,'lum-threshold':$('o_lum').value,
  'min-fragment-area':$('o_minfrag').value,'color-merge-tol':$('o_mergetol').value,
  'alpha-min':$('o_alpha').value,'link-angle':$('o_linkangle').value,'link-width-ratio':$('o_linkratio').value};
 if(m==='tier'){o['tier-bold']=$('o_tierbold').value;o['tier-thin']=$('o_tierthin').value;}
 return o;}
function svgFlags(){return {'smooth-curves':$('o_smooth').checked,'merge-leading':$('o_mergelead').checked};}
let ALLFIL=[], SEL=[], SLOTS=4;
function params(){return {depth:$('o_depth').value,size:$('o_size').value,colors:$('o_colors').value,max_delta:$('o_maxdelta').value,filaments:SEL};}
async function lut(){const r=await post('/lutstatus',{});
 if(!r.ready){$('lut').innerHTML='<div class=warn><b>No calibrated filaments yet · 尚无已校准耗材</b><br>Calibrate filaments first: run <code>python3 filament/gui.py</code> · 请先用耗材 GUI 校准：运行 <code>python3 filament/gui.py</code></div>';return;}
 ALLFIL=r.filaments; if(SEL.length===0)SEL=ALLFIL.slice(0,Math.min(SLOTS,ALLFIL.length));
 renderFil(); genGamut();}
function renderFil(){
 let h='<div style="color:#555;font-size:12px;margin-bottom:4px">Select the filaments loaded in your AMS (max '+SLOTS+' slots) · 勾选 AMS 中已装载的耗材（最多 '+SLOTS+' 槽）</div>';
 ALLFIL.forEach(f=>{const on=SEL.includes(f);const dis=(!on&&SEL.length>=SLOTS);
   h+='<label style="margin-right:16px;color:'+(dis?'#bbb':'#222')+'"><input type=checkbox '+(on?'checked':'')+(dis?' disabled':'')+' onchange="togg(\''+f+'\')"> '+f+'</label>';});
 h+=' &nbsp;<button onclick="addAms()">+ Filament 加耗材</button> &nbsp;<span style="color:#777;font-size:12px">'+SEL.length+'/'+SLOTS+' slots 槽</span>';
 h+=' &nbsp;<button onclick="suggest()">◎ Suggest for image 为图片推荐</button> <span id=sug_status style="color:#777;font-size:12px"></span>';
 $('lut').innerHTML=h;}
async function suggest(){const el=$('img');
 if(!el.files[0]){alert('pick an image in step ② first · 请先在②选择图片');return;}
 $('sug_status').textContent='scoring… 评分中…';
 const image=await f2b64(el.files[0]);
 const r=await post('/suggest',{image,slots:SLOTS,max_delta:$('o_maxdelta').value});
 $('sug_status').textContent='';
 if(!r.ok){$('sug_result').innerHTML='<div class=warn>'+(r.stderr||'')+'</div>';return;}
 let h='<div style="color:#555;font-size:12px">Best '+r.slots+'-filament palettes for this image ('+r.n_colors+' colours) — click to apply · 点击应用最佳组合：</div><table style="font-size:13px">';
 r.ranked.forEach(x=>{h+='<tr><td><button onclick="applyPal(\''+x.filaments.join(',')+'\')">'+x.filaments.join(' + ')+'</button></td><td>&nbsp;mean ΔE '+x.mean_de.toFixed(1)+'</td><td>&nbsp;'+Math.round(x.oog_frac*100)+'% out-of-gamut 超色域</td></tr>';});
 h+='</table>';$('sug_result').innerHTML=h;}
function applyPal(csv){SEL=csv.split(',').slice(0,SLOTS);renderFil();genGamut();}
function togg(f){const i=SEL.indexOf(f);if(i>=0)SEL.splice(i,1);else if(SEL.length<SLOTS)SEL.push(f);renderFil();genGamut();}
function addAms(){SLOTS=Math.min(8,SLOTS+4);renderFil();}
async function genGamut(){$('gamut').innerHTML='<span style="color:#777">building gamut… 生成色域…</span>';
 const r=await post('/gamut',{filaments:SEL});
 $('gamut').innerHTML=r.gamut?('<div style="color:#555;font-size:12px">reachable gamut with ['+SEL.join(', ')+'] · 可达色域</div>'+img(r.gamut)):('<div class=warn>'+(r.stderr||'')+'</div>');}
async function convert(){const el=$('img');if(!el.files[0]){alert('pick an image 请选择图片');return;}
 $('c_status').textContent='vectorising… 矢量化中…';
 const image=await f2b64(el.files[0]);
 const r=await post('/convert',{image,svg:svgOpts(),flags:svgFlags()});$('c_status').textContent='';
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
