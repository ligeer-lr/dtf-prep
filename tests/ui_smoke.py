"""اختبار دخان للواجهة في متصفح حقيقي (اختياري، ليس جزءاً من pytest).

يحتاج:  pip install playwright && playwright install chromium
ويحتاج خادماً شغّالاً:  uvicorn app.main:app --port 8000
التشغيل:  DTF_URL=http://127.0.0.1:8000 python tests/ui_smoke.py
اللقطات تُحفظ في مجلد مؤقت يُطبع مساره.
"""
import io, json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.make_sample import make_logo
OUT = Path(tempfile.mkdtemp(prefix="dtf-ui-"))
LOGO = str(OUT / "logo1500.jpg")
make_logo(1500, 1000).save(LOGO, "JPEG", quality=88)
print("output dir:", OUT)
from playwright.sync_api import sync_playwright
from PIL import Image
import numpy as np

BASE = os.environ.get("DTF_URL", "http://127.0.0.1:8000")
res = []
def ok(name, cond, extra=""):
    res.append((name, bool(cond), extra)); print(("PASS " if cond else "FAIL ") + name, extra, flush=True)

def set_range(pg, sel, val):
    pg.eval_on_selector(sel, "(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));}", str(val))

def wait_after(pg, timeout=20000):
    pg.wait_for_function("()=>{const i=document.querySelector('#afterImg');return i&&i.complete&&i.naturalWidth>0&&document.querySelector('#liveBadge').hidden&&document.querySelector('#afterOverlay').hidden}", timeout=timeout)

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1366, "height": 820}, locale="ar", accept_downloads=True)
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
    reqs = []
    pg.on("request", lambda r: reqs.append((r.method, r.url.replace(BASE, ""))) if "/api/" in r.url else None)

    pg.goto(BASE)
    ok("upload stage visible, workspace hidden", pg.is_visible("#uploadStage") and not pg.is_visible("#workspace"))

    t0 = time.time()
    pg.set_input_files("#file", LOGO)
    pg.wait_for_selector("#workspace:not([hidden])")
    ok("upload stage hidden after pick", not pg.is_visible("#uploadStage"))
    wait_after(pg)
    ok("first preview shown", True, f"{time.time()-t0:.1f}s")
    ok("default width 30", pg.input_value("#w") == "30", pg.input_value("#w"))
    ok("height from ratio 20", pg.input_value("#h") == "20", pg.input_value("#h"))
    ok("DPI line", "DPI" in pg.inner_text("#dpiLine"), pg.inner_text("#dpiLine"))
    ok("px line 3543x2362", "3543 × 2362" in pg.inner_text("#pxLine"), pg.inner_text("#pxLine"))
    ok("bg label auto #ffffff", "#ffffff" in pg.inner_text("#bgLabel"), pg.inner_text("#bgLabel"))
    pg.screenshot(path=str(OUT / "ui_desktop.png"))

    # الصورة الناتجة: الشفافية وبقاء النجمة البيضاء
    data = pg.evaluate("""async()=>{const i=document.querySelector('#afterImg');const c=document.createElement('canvas');c.width=i.naturalWidth;c.height=i.naturalHeight;const x=c.getContext('2d');x.drawImage(i,0,0);const d=x.getImageData(0,0,c.width,c.height).data;
      const px=(X,Y)=>{const k=(Y*c.width+X)*4;return [d[k],d[k+1],d[k+2],d[k+3]]};
      return {w:c.width,h:c.height,corner:px(3,3),center:px(Math.floor(c.width/2),Math.floor(c.height/2)-Math.floor(c.height*0.05)),circle:px(Math.floor(c.width/2),Math.floor(c.height*0.5-c.height*0.27))}}""")
    ok("after corner transparent", data["corner"][3] == 0, str(data["corner"]))
    ok("after star center white+opaque", data["center"][3] == 255 and min(data["center"][:3]) > 235, str(data["center"]))

    # سلايدر: زمن التحديث
    lat = []
    for e in (0, 2, 3, 1):
        ms = pg.evaluate("""(v)=>new Promise(res=>{const i=document.querySelector('#afterImg'),r=document.querySelector('#erode');const t=performance.now();
            const h=()=>{i.removeEventListener('load',h);res(performance.now()-t)};i.addEventListener('load',h);r.value=v;r.dispatchEvent(new Event('input',{bubbles:true}));})""", e)
        lat.append(ms)
        wait_after(pg)
    ok("erode slider -> preview image updated < 500ms", max(lat) < 500, "ms=" + ",".join(str(int(x)) for x in lat))
    nprev = len([r for r in reqs if r[1].endswith("/preview")])
    ok("preview requests sent for slider moves", nprev >= 4, str(nprev))

    # اختيار لون الخلفية بالضغط
    vb = pg.locator("#viewBefore").bounding_box()
    cb = pg.evaluate("()=>{const l=document.querySelector('#layerBefore');const v=document.querySelector('#viewBefore');return {left:l.offsetLeft,top:l.offsetTop,w:l.offsetWidth,h:l.offsetHeight}}")
    pg.locator("#viewBefore").click(position={"x": cb["left"] + 12, "y": cb["top"] + 12})
    pg.wait_for_function("()=>document.querySelector('#bgLabel').textContent.includes('يدوي')", timeout=5000)
    ok("manual bg color picked", "يدوي" in pg.inner_text("#bgLabel"), pg.inner_text("#bgLabel"))
    wait_after(pg)
    pg.click("#bgAuto")
    pg.wait_for_function("()=>document.querySelector('#bgLabel').textContent.includes('تلقائي')", timeout=5000)
    ok("reset to auto bg", True)
    wait_after(pg)

    # تكبير وتحريك متزامن
    box = pg.locator("#viewBefore").bounding_box()
    pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    pg.mouse.wheel(0, -600)
    pg.wait_for_timeout(200)
    z = pg.inner_text("#zoomLabel")
    ok("wheel zoom > 100%", int(z.replace("%", "")) > 100, z)
    tr = pg.evaluate("()=>[document.querySelector('#layerBefore').style.transform,document.querySelector('#layerAfter').style.transform]")
    ok("zoom synced both panes", tr[0] == tr[1] and "scale(1)" not in tr[0], tr[0])
    box2 = pg.locator("#viewAfter").bounding_box()
    sx, sy = box2["x"] + box2["width"] / 2, box2["y"] + box2["height"] / 2
    pg.mouse.move(sx, sy); pg.mouse.down(); pg.mouse.move(sx + 60, sy + 30, steps=6); pg.mouse.up()
    tr2 = pg.evaluate("()=>[document.querySelector('#layerBefore').style.transform,document.querySelector('#layerAfter').style.transform]")
    ok("pan from AFTER pane moves BEFORE too", tr2[0] == tr2[1] and tr2[0] != tr[0], tr2[0])
    pg.screenshot(path=str(OUT / "ui_zoom.png"))
    pg.dblclick("#viewBefore")
    ok("double click resets zoom", pg.inner_text("#zoomLabel") == "100%")

    # لون خلفية المعاينة
    pg.click('.sw[data-color="#c1121f"]')
    ok("swatch sets after bg", "rgb(193, 18, 31)" in pg.evaluate("()=>getComputedStyle(document.querySelector('#viewAfter')).backgroundColor"))
    pg.click('.sw[data-color=""]')

    # قفل النسبة
    pg.fill("#w", "45")
    ok("lock: width 45 -> height 30", pg.input_value("#h") == "30", pg.input_value("#h"))
    pg.fill("#h", "20")
    ok("lock: height 20 -> width 30", pg.input_value("#w") == "30", pg.input_value("#w"))
    pg.click("label.lock")
    pg.fill("#h", "10")
    ok("unlock: ratio note shown", pg.is_visible("#ratioNote"))
    pg.click("label.lock")
    ok("lock on: height recomputed", pg.input_value("#h") == "20", pg.input_value("#h"))
    pg.fill("#w", "50")
    ok("low DPI red warning", pg.is_visible("#dpiWarn") and "low" in (pg.get_attribute("#dpiLine", "class") or ""), pg.inner_text("#dpiLine"))
    pg.fill("#w", "٣٠")
    ok("arabic digits accepted", pg.input_value("#h") == "20", pg.input_value("#h"))
    pg.fill("#w", "300")
    ok("too-big size error + download disabled", pg.is_visible("#sizeError") and pg.is_disabled("#download"))
    pg.fill("#w", "30")

    # وضع الذكاء الاصطناعي
    pg.click('label:has(input[value="ai"])')
    t = time.time()
    pg.wait_for_function("()=>!document.querySelector('#afterOverlay').hidden", timeout=5000)
    ok("AI first-run overlay shown", True)
    wait_after(pg, 40000)
    ok("AI preview done", True, f"{time.time()-t:.1f}s")
    t = time.time(); set_range(pg, "#erode", 2); pg.wait_for_timeout(80); wait_after(pg);
    ok("AI second preview fast (cached mask)", time.time() - t < 1.5, f"{(time.time()-t)*1000:.0f}ms")
    pg.click('label:has(input[value="none"])')
    wait_after(pg)
    ok("mode none hides solid box", not pg.is_visible("#solidBox"))
    pg.click("#reset")
    ok("reset restores solid mode", pg.is_checked('input[name="bgMode"][value="solid"]') and pg.input_value("#tol") == "25" and pg.input_value("#erode") == "1")
    wait_after(pg)

    # التحميل النهائي
    pg.fill("#w", "20")
    with pg.expect_download(timeout=90000) as dl:
        pg.click("#download")
        pg.wait_for_selector("#finalModal:not([hidden])")
    d = dl.value
    path = str(OUT / "final_from_ui.png"); d.save_as(path)
    im = Image.open(path)
    ok("final download is 2362x1574 (20x13.33cm) RGBA 300dpi", im.size == (2362, 1574) and im.mode == "RGBA" and round(im.info["dpi"][0]) == 300, f"{im.size} {im.mode} {im.info.get('dpi')}")
    pg.wait_for_selector("#finalDone:not([hidden])", timeout=60000)
    ok("final modal shows chips + link", pg.locator("#chips li").count() >= 4 and "/result" in pg.get_attribute("#finalLink", "href"))
    a = np.asarray(im)
    ok("final star white opaque, corner transparent", a[3, 3, 3] == 0 and a[a.shape[0] // 2 - 80, a.shape[1] // 2, 3] == 255 and a[a.shape[0] // 2 - 80, a.shape[1] // 2, :3].min() > 235)
    pg.screenshot(path=str(OUT / "ui_modal.png"))
    pg.click("#closeModal")
    ok("single upload to /api/sessions (final used session_id)", len([r for r in reqs if r == ("POST", "/api/sessions")]) == 1, str([r for r in reqs if r[1] == "/api/sessions"]))

    ok("no JS errors on desktop", not errs, "; ".join(errs[:3]))

    # جوال
    m = b.new_context(viewport={"width": 390, "height": 844}, locale="ar", has_touch=True, is_mobile=True)
    mp = m.new_page(); merrs = []
    mp.on("pageerror", lambda e: merrs.append(str(e)))
    mp.on("console", lambda x: merrs.append(x.text) if x.type == "error" else None)
    mp.goto(BASE); mp.set_input_files("#file", LOGO); mp.wait_for_selector("#workspace:not([hidden])"); wait_after(mp)
    sw = mp.evaluate("()=>[document.documentElement.scrollWidth,document.documentElement.clientWidth]")
    ok("mobile: no horizontal scroll", sw[0] <= sw[1], str(sw))
    pos = mp.evaluate("()=>{const a=document.querySelector('#viewBefore').getBoundingClientRect(),b=document.querySelector('#viewAfter').getBoundingClientRect(),c=document.querySelector('.controls').getBoundingClientRect();return {beforeTop:a.top,afterTop:b.top,ctrlTop:c.top+scrollY,afterBottom:b.bottom+scrollY}}")
    ok("mobile: panes stacked then controls below", pos["afterTop"] > pos["beforeTop"] + 100 and pos["ctrlTop"] >= pos["afterBottom"] - 5, str(pos))
    mp.screenshot(path=str(OUT / "ui_mobile_top.png"))
    mp.evaluate("()=>document.querySelector('.controls').scrollIntoView()")
    mp.wait_for_timeout(300)
    mp.screenshot(path=str(OUT / "ui_mobile_ctrl.png"))
    fixed = mp.evaluate("()=>{const r=document.querySelector('.cta').getBoundingClientRect();return [Math.round(r.bottom),innerHeight]}")
    ok("mobile: CTA bar pinned to bottom", fixed[0] == fixed[1], str(fixed))
    ok("no JS errors on mobile", not merrs, "; ".join(merrs[:3]))
    b.close()

bad = [r for r in res if not r[1]]
print(f"\n{len(res)-len(bad)}/{len(res)} passed")
sys.exit(1 if bad else 0)
