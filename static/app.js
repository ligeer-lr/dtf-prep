(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
  const ALLOWED_EXT = /\.(png|jpe?g|webp)$/i;
  const DEFAULTS = { mode: "solid", tol: 25, erode: 1, alpha: 50, alphaKeep: false };
  const DEFAULT_WIDTH_CM = 30;
  const MAX_PREVIEW_SIDE = 1000;
  const MAX_ZOOM = 16;

  const state = {
    cfg: { max_upload_mb: 20, dpi: 300, min_side_cm: 1, max_side_cm: 120, max_output_mp: 80, low_dpi_warning: 100 },
    file: null, sid: null, uploadToken: 0,
    iw: 0, ih: 0,            // مقاس الصورة الأصلية (بعد تصحيح الاتجاه)
    cw: 0, ch: 0,            // مقاس لوحة «قبل» (النسخة المصغّرة)
    transparent: false, autoBg: "", usedBg: "", bgColor: "", pickedOnce: false,
    aiReady: false, hasAfter: false,
    k: 1, x: 0, y: 0,
    seq: 0, ctrl: null, timer: null, afterUrl: null,
    jobId: null, poll: null, finalRunning: false,
  };

  const el = {
    uploadStage: $("#uploadStage"), drop: $("#drop"), file: $("#file"), fileError: $("#fileError"), maxMb: $("#maxMb"),
    workspace: $("#workspace"), changeImg: $("#changeImg"),
    viewBefore: $("#viewBefore"), viewAfter: $("#viewAfter"), layerBefore: $("#layerBefore"), layerAfter: $("#layerAfter"),
    beforeCanvas: $("#beforeCanvas"), afterImg: $("#afterImg"), pickHint: $("#pickHint"),
    liveBadge: $("#liveBadge"), liveText: $("#liveText"), afterOverlay: $("#afterOverlay"), overlayText: $("#overlayText"), afterError: $("#afterError"),
    zoomIn: $("#zoomIn"), zoomOut: $("#zoomOut"), zoomFit: $("#zoomFit"), zoomLabel: $("#zoomLabel"),
    swatches: $("#swatches"), customColor: $("#customColor"), customDot: $("#customDot"), customSw: $("#customSw"),
    w: $("#w"), h: $("#h"), lock: $("#lock"), sizeError: $("#sizeError"), dpiLine: $("#dpiLine"), pxLine: $("#pxLine"),
    dpiWarn: $("#dpiWarn"), ratioNote: $("#ratioNote"),
    transNote: $("#transNote"), solidBox: $("#solidBox"), aiHint: $("#aiHint"),
    bgChip: $("#bgChip"), bgLabel: $("#bgLabel"), bgAuto: $("#bgAuto"),
    tol: $("#tol"), tolOut: $("#tolOut"), erode: $("#erode"), erodeOut: $("#erodeOut"),
    alpha: $("#alpha"), alphaOut: $("#alphaOut"), alphaKeep: $("#alphaKeep"),
    download: $("#download"), reset: $("#reset"),
    modal: $("#finalModal"), finalTitle: $("#finalTitle"), finalProgress: $("#finalProgress"), bar: $("#bar"), barFill: $("#barFill"),
    stage: $("#stage"), pct: $("#pct"), finalError: $("#finalError"), finalDone: $("#finalDone"), finalNotes: $("#finalNotes"),
    chips: $("#chips"), finalLink: $("#finalLink"), closeModal: $("#closeModal"),
  };

  /* ====================== أدوات ====================== */
  const show = (n, on = true) => { n.hidden = !on; };
  const fmt = (n) => String(Math.round(n * 100) / 100);
  const hex = (r, g, b) => "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
  const fmtBytes = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " ميجابايت" : Math.max(1, Math.round(n / 1024)) + " كيلوبايت");
  const mode = () => document.querySelector('input[name="bgMode"]:checked').value;

  // يقبل الأرقام العربية/الفارسية والفاصلة العربية: ٦٠٫٥ → 60.5
  function parseNum(raw) {
    let s = String(raw || "").trim();
    if (!s) return NaN;
    s = s.replace(/[٠-٩]/g, (d) => d.charCodeAt(0) - 0x0660).replace(/[۰-۹]/g, (d) => d.charCodeAt(0) - 0x06f0).replace(/[٫,،]/g, ".");
    return /^\d*\.?\d+$/.test(s) ? parseFloat(s) : NaN;
  }

  function toast(msg) {
    let t = $("#toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; t.setAttribute("role", "alert"); document.body.append(t); }
    t.textContent = msg; t.classList.add("on");
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), 6000);
  }

  function del(url) { try { fetch(url, { method: "DELETE", keepalive: true }).catch(() => {}); } catch (_) {} }

  /* ====================== المقاس ====================== */
  function sizeState() {
    const { min_side_cm: mn, max_side_cm: mx, max_output_mp: mp, dpi } = state.cfg;
    const w = parseNum(el.w.value), h = parseNum(el.h.value);
    if (!el.w.value.trim() || !el.h.value.trim()) return { ok: false, msg: "" };
    if (!(w > 0) || !(h > 0)) return { ok: false, msg: "المقاس غير صحيح. أدخل العرض والطول بالسنتيمتر كأرقام." };
    if (w < mn || h < mn) return { ok: false, msg: `المقاس صغير جداً. أقل مقاس مسموح ${mn} سم لكل جهة.` };
    if (w > mx || h > mx) return { ok: false, msg: `المقاس كبير جداً. أقصى مقاس مسموح ${mx} سم لكل جهة.` };
    const tw = Math.max(1, Math.round((w / 2.54) * dpi)), th = Math.max(1, Math.round((h / 2.54) * dpi));
    if (tw * th > mp * 1e6) {
      const side = Math.round((Math.sqrt(mp * 1e6) / dpi) * 2.54);
      return { ok: false, msg: `مقاس الطباعة كبير على الخادم: الملف الناتج سيكون ${tw}×${th} بكسل والحد الأقصى ${mp} ميجابكسل (يعادل تقريباً ${side}×${side} سم). صغّر المقاس.` };
    }
    return { ok: true, w, h, tw, th };
  }

  function setDefaultSize() {
    const { min_side_cm: mn, max_side_cm: mx } = state.cfg;
    let w = DEFAULT_WIDTH_CM, h = (w * state.ih) / state.iw;
    if (h > mx) { h = mx; w = (h * state.iw) / state.ih; }
    if (h < mn) { h = mn; w = (h * state.iw) / state.ih; }
    el.w.value = fmt(w); el.h.value = fmt(h);
  }

  function updateSizeInfo() {
    const s = sizeState();
    const { dpi, low_dpi_warning: low } = state.cfg;
    el.sizeError.textContent = !s.ok ? s.msg : "";
    show(el.sizeError, !s.ok && !!s.msg);
    el.dpiLine.textContent = ""; el.pxLine.textContent = "";
    el.dpiLine.classList.remove("low");
    show(el.dpiWarn, false); show(el.ratioNote, false);
    if (s.ok && state.iw) {
      const scale = Math.min(s.tw / state.iw, s.th / state.ih);
      const eff = dpi / scale;
      el.dpiLine.replaceChildren("الدقة الفعلية عند هذا المقاس: ", Object.assign(document.createElement("b"), { textContent: Math.round(eff) + " DPI" }),
        eff > dpi * 1.02 ? ` (أعلى من ${dpi}؛ ستُصغَّر إلى ${dpi})` : "");
      el.pxLine.replaceChildren("حجم الملف الناتج: ", Object.assign(document.createElement("b"), { textContent: `${s.tw} × ${s.th}` }),
        ` بكسل (≈ ${((s.tw * s.th) / 1e6).toFixed(1)} ميجابكسل)`);
      if (eff < low) {
        el.dpiLine.classList.add("low");
        el.dpiWarn.textContent = `الصورة ضعيفة لهذا المقاس: دقتها الفعلية ${Math.round(eff)} DPI فقط (أقل من ${low}) والمطلوب ${dpi}. ستُكبَّر حوالي ${scale.toFixed(1)} مرة وقد تظهر التفاصيل ناعمة. ارفع صورة أكبر أو صغّر المقاس.`;
        show(el.dpiWarn, true);
      }
      if (!el.lock.checked && Math.abs(state.iw / state.ih / (s.w / s.h) - 1) > 0.02) {
        el.ratioNote.textContent = "نسبة المقاس تختلف عن نسبة الصورة: ستحافظ الصورة على نسبتها بدون تشويه، وتُضاف مساحة شفافة حولها ليكون الملف بالمقاس المطلوب بالضبط.";
        show(el.ratioNote, true);
      }
    }
    updateDownloadEnabled();
  }

  function onSizeInput(which) {
    if (el.lock.checked && state.iw) {
      const v = parseNum(el[which].value);
      if (v > 0) {
        if (which === "w") el.h.value = fmt((v * state.ih) / state.iw);
        else el.w.value = fmt((v * state.iw) / state.ih);
      }
    }
    updateSizeInfo();
    schedulePreview(350); // مقاس الطباعة يؤثر على مقدار «تصغير الحواف» في المعاينة
  }
  el.w.addEventListener("input", () => onSizeInput("w"));
  el.h.addEventListener("input", () => onSizeInput("h"));
  el.h.addEventListener("keydown", (e) => { if (e.key === "Enter" && !el.download.disabled) el.download.click(); });
  el.lock.addEventListener("change", () => { if (el.lock.checked) onSizeInput("w"); else updateSizeInfo(); });

  function updateDownloadEnabled() {
    el.download.disabled = !(state.sid && sizeState().ok) || state.finalRunning;
  }

  /* ====================== العرض: تكبير وتحريك متزامن ====================== */
  const views = [
    { view: el.viewBefore, layer: el.layerBefore, id: "before" },
    { view: el.viewAfter, layer: el.layerAfter, id: "after" },
  ];

  function metrics(view) {
    if (!state.cw) return null;
    const vw = view.clientWidth, vh = view.clientHeight;
    const f = Math.min(vw / state.cw, vh / state.ch) * 0.94;
    return { vw, vh, bw: state.cw * f, bh: state.ch * f };
  }

  function clampPan() {
    if (state.k <= 1.001) { state.x = 0; state.y = 0; return; }
    const m = metrics(el.viewBefore);
    if (!m) return;
    const mx = Math.max(0, (m.bw * state.k - m.vw) / 2) + 60;
    const my = Math.max(0, (m.bh * state.k - m.vh) / 2) + 60;
    state.x = Math.max(-mx, Math.min(mx, state.x));
    state.y = Math.max(-my, Math.min(my, state.y));
  }

  function applyView() {
    clampPan();
    for (const { view, layer } of views) {
      const m = metrics(view);
      if (!m) continue;
      layer.style.width = m.bw + "px";
      layer.style.height = m.bh + "px";
      layer.style.left = (m.vw - m.bw) / 2 + "px";
      layer.style.top = (m.vh - m.bh) / 2 + "px";
      layer.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.k})`;
      view.classList.toggle("zoomed", state.k > 1.001);
    }
    el.zoomLabel.textContent = Math.round(state.k * 100) + "%";
  }

  // cx, cy: نقطة التكبير نسبة لمركز اللوحة
  function zoomAt(factor, cx = 0, cy = 0) {
    const k2 = Math.min(MAX_ZOOM, Math.max(1, state.k * factor));
    const r = k2 / state.k;
    state.x = cx - (cx - state.x) * r;
    state.y = cy - (cy - state.y) * r;
    state.k = k2;
    applyView();
  }
  const resetView = () => { state.k = 1; state.x = 0; state.y = 0; applyView(); };

  el.zoomIn.addEventListener("click", () => zoomAt(1.5));
  el.zoomOut.addEventListener("click", () => zoomAt(1 / 1.5));
  el.zoomFit.addEventListener("click", resetView);
  if (window.ResizeObserver) { const ro = new ResizeObserver(applyView); views.forEach((v) => ro.observe(v.view)); }
  window.addEventListener("resize", applyView);

  function attachGestures({ view, id }) {
    const ptrs = new Map();
    let g = null;
    const center = (r) => ({ cx: r.left + r.width / 2, cy: r.top + r.height / 2 });

    view.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = view.getBoundingClientRect(), c = center(r);
      zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)), e.clientX - c.cx, e.clientY - c.cy);
    }, { passive: false });

    view.addEventListener("dblclick", resetView);

    view.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      try { view.setPointerCapture(e.pointerId); } catch (_) {}
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (ptrs.size === 1) g = { sx: e.clientX, sy: e.clientY, moved: false, lastDist: 0, lastMx: 0, lastMy: 0 };
      else if (g) { g.moved = true; g.lastDist = 0; }
    });

    view.addEventListener("pointermove", (e) => {
      const prev = ptrs.get(e.pointerId);
      if (!prev || !g) return;
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (ptrs.size === 1) {
        if (Math.hypot(e.clientX - g.sx, e.clientY - g.sy) > 5) g.moved = true;
        if (g.moved && state.k > 1.001) {
          state.x += e.clientX - prev.x; state.y += e.clientY - prev.y;
          view.classList.add("panning");
          applyView();
        }
      } else if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y) || 1;
        const r = view.getBoundingClientRect(), c = center(r);
        const mx = (a.x + b.x) / 2 - c.cx, my = (a.y + b.y) / 2 - c.cy;
        if (g.lastDist) {
          zoomAt(dist / g.lastDist, mx, my);
          state.x += mx - g.lastMx; state.y += my - g.lastMy;
          applyView();
        }
        g.lastDist = dist; g.lastMx = mx; g.lastMy = my;
      }
    });

    const end = (e) => {
      if (!ptrs.has(e.pointerId)) return;
      ptrs.delete(e.pointerId);
      if (ptrs.size === 0) {
        view.classList.remove("panning");
        if (g && !g.moved && e.type === "pointerup" && id === "before") pickAt(e);
        g = null;
      } else if (g) { g.moved = true; g.lastDist = 0; }
    };
    view.addEventListener("pointerup", end);
    view.addEventListener("pointercancel", end);
  }
  views.forEach(attachGestures);

  /* ====================== لون الخلفية (اختيار بالضغط) ====================== */
  function pickAt(e) {
    if (mode() !== "solid" || state.transparent || !state.cw) return;
    const view = el.viewBefore, m = metrics(view);
    if (!m) return;
    const r = view.getBoundingClientRect();
    const px = e.clientX - r.left - m.vw / 2, py = e.clientY - r.top - m.vh / 2;
    const lx = (px - state.x) / state.k + m.bw / 2, ly = (py - state.y) / state.k + m.bh / 2;
    if (lx < 0 || ly < 0 || lx >= m.bw || ly >= m.bh) return;
    const ix = Math.min(state.cw - 1, Math.floor((lx / m.bw) * state.cw));
    const iy = Math.min(state.ch - 1, Math.floor((ly / m.bh) * state.ch));
    try {
      // متوسط 3×3 لتقليل ضجيج JPG
      const x0 = Math.max(0, ix - 1), y0 = Math.max(0, iy - 1);
      const w = Math.min(state.cw - x0, 3), h = Math.min(state.ch - y0, 3);
      const d = el.beforeCanvas.getContext("2d").getImageData(x0, y0, w, h).data;
      let r_ = 0, g_ = 0, b_ = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i + 3] >= 128) { r_ += d[i]; g_ += d[i + 1]; b_ += d[i + 2]; n++; }
      if (!n) return;
      state.bgColor = hex(r_ / n, g_ / n, b_ / n);
      state.pickedOnce = true;
      syncUI();
      schedulePreview(0);
    } catch (_) { /* لوحة غير قابلة للقراءة */ }
  }

  el.bgAuto.addEventListener("click", () => { state.bgColor = ""; syncUI(); schedulePreview(0); });

  /* ====================== التحكمات ====================== */
  function syncUI() {
    const m = mode(), tr = state.transparent;
    show(el.transNote, tr);
    document.querySelectorAll('input[name="bgMode"]').forEach((i) => { i.disabled = tr; });
    show(el.solidBox, m === "solid");
    el.solidBox.classList.toggle("is-disabled", tr);
    show(el.aiHint, m === "ai" && !tr);
    el.viewBefore.classList.toggle("picking", m === "solid" && !tr);
    show(el.pickHint, m === "solid" && !tr && !state.pickedOnce && state.hasAfter);

    const c = state.bgColor || state.usedBg || state.autoBg;
    el.bgChip.classList.toggle("checker", !c);
    el.bgChip.firstElementChild.style.background = c || "transparent";
    el.bgLabel.textContent = state.bgColor ? `لون الخلفية (يدوي): ${state.bgColor}` : c ? `لون الخلفية (تلقائي): ${c}` : "لون الخلفية: يُكتشف تلقائياً";
    show(el.bgAuto, !!state.bgColor);

    const t = +el.tol.value, e = +el.erode.value, a = +el.alpha.value;
    el.tolOut.textContent = `${t < 15 ? "منخفضة" : t <= 40 ? "متوسطة" : "عالية"} (${t})`;
    el.erodeOut.textContent = e === 0 ? "بدون تصغير" : `${e} بكسل`;
    el.alpha.disabled = el.alphaKeep.checked;
    el.alphaOut.textContent = el.alphaKeep.checked ? "بدون تغيير" : `${a < 35 ? "خفيفة" : a <= 65 ? "معتدلة" : "عالية"} (${a}%)`;
  }

  document.querySelectorAll('input[name="bgMode"]').forEach((i) => i.addEventListener("change", () => { syncUI(); schedulePreview(0); }));
  [el.tol, el.erode, el.alpha, el.alphaKeep].forEach((i) => i.addEventListener("input", () => { syncUI(); schedulePreview(60); }));

  el.reset.addEventListener("click", () => {
    document.querySelector(`input[name="bgMode"][value="${DEFAULTS.mode}"]`).checked = true;
    el.tol.value = DEFAULTS.tol; el.erode.value = DEFAULTS.erode; el.alpha.value = DEFAULTS.alpha; el.alphaKeep.checked = DEFAULTS.alphaKeep;
    state.bgColor = "";
    syncUI(); schedulePreview(0);
  });

  /* ====================== لون خلفية المعاينة ====================== */
  function setStageColor(color) {
    const v = el.viewAfter;
    if (!color) { v.classList.add("checker"); v.classList.remove("solid"); v.style.backgroundColor = ""; }
    else { v.classList.remove("checker"); v.classList.add("solid"); v.style.backgroundColor = color; }
  }
  function activateSwatch(btn) {
    el.swatches.querySelectorAll(".sw").forEach((b) => {
      b.classList.toggle("active", b === btn);
      if (b.getAttribute("role") === "radio") b.setAttribute("aria-checked", String(b === btn));
    });
  }
  el.swatches.addEventListener("click", (e) => {
    const b = e.target.closest("button.sw");
    if (!b) return;
    activateSwatch(b); setStageColor(b.dataset.color);
  });
  el.customColor.addEventListener("input", () => {
    const c = el.customColor.value;
    el.customDot.classList.remove("rainbow"); el.customDot.style.background = c;
    activateSwatch(el.customSw); setStageColor(c);
  });

  /* ====================== الملف والجلسة ====================== */
  function xhrUpload(url, fd, onProgress) {
    return new Promise((resolve) => {
      const x = new XMLHttpRequest();
      x.open("POST", url);
      x.responseType = "json";
      x.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
      x.onload = () => resolve({ status: x.status, json: x.response || {} });
      x.onerror = () => resolve({ status: 0, json: {} });
      x.send(fd);
    });
  }

  function setOverlay(text) { el.overlayText.textContent = text; show(el.afterOverlay, !!text); }
  function setAfterError(msg) { el.afterError.textContent = msg || ""; show(el.afterError, !!msg); }
  function setBusy(on, text) { show(el.liveBadge, on); if (text) el.liveText.textContent = text; }

  function drawBefore(img) {
    const nw = img.naturalWidth, nh = img.naturalHeight;
    const s = Math.min(1, MAX_PREVIEW_SIDE / Math.max(nw, nh));
    state.cw = Math.max(1, Math.round(nw * s)); state.ch = Math.max(1, Math.round(nh * s));
    const cv = el.beforeCanvas;
    cv.width = state.cw; cv.height = state.ch;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, 0, 0, state.cw, state.ch);
  }

  async function setFile(f) {
    show(el.fileError, false);
    const fail = (m) => { if (el.workspace.hidden) { el.fileError.textContent = m; show(el.fileError, true); } else toast(m); };
    if (!f) return;
    const typeOk = ALLOWED_TYPES.includes(f.type) || (!f.type && ALLOWED_EXT.test(f.name));
    if (!typeOk) return fail("صيغة الملف غير مدعومة. الصيغ المسموحة: PNG وJPG وWebP.");
    if (f.size === 0) return fail("الملف فارغ. اختر صورة أخرى.");
    if (f.size > state.cfg.max_upload_mb * 1048576) {
      return fail(`حجم الملف ${fmtBytes(f.size)} يتجاوز الحد الأقصى (${state.cfg.max_upload_mb} ميجابايت). صغّر الصورة وحاول مجدداً.`);
    }
    const url = URL.createObjectURL(f);
    const img = new Image();
    try {
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
    } catch (_) {
      URL.revokeObjectURL(url);
      return fail("تعذّر قراءة الصورة. تأكد أن الملف سليم وليس تالفاً.");
    }

    // صورة جديدة: نظّف كل ما يخص السابقة
    if (state.sid) del(`/api/sessions/${state.sid}`);
    if (state.jobId) { del(`/api/jobs/${state.jobId}`); state.jobId = null; }
    clearTimeout(state.timer); state.ctrl && state.ctrl.abort(); state.seq++;
    state.file = f; state.sid = null;
    state.iw = img.naturalWidth; state.ih = img.naturalHeight;
    state.transparent = false; state.autoBg = ""; state.usedBg = ""; state.bgColor = "";
    state.pickedOnce = false; state.aiReady = false; state.hasAfter = false;
    drawBefore(img);
    URL.revokeObjectURL(url);
    if (state.afterUrl) { URL.revokeObjectURL(state.afterUrl); state.afterUrl = null; }
    el.afterImg.removeAttribute("src");

    show(el.uploadStage, false); show(el.workspace, true); show(el.changeImg, true);
    setAfterError(""); setOverlay("جارٍ رفع الصورة…");
    setDefaultSize();
    resetView();
    syncUI(); updateSizeInfo();
    uploadSession().then((ok) => { if (ok) schedulePreview(0); });
  }

  async function uploadSession() {
    const token = ++state.uploadToken;
    state.sid = null; updateDownloadEnabled();
    setAfterError("");
    if (!state.hasAfter) setOverlay("جارٍ رفع الصورة… 0%");
    const fd = new FormData();
    fd.append("file", state.file, state.file.name);
    const r = await xhrUpload("/api/sessions", fd, (p) => { if (token === state.uploadToken && !state.hasAfter) setOverlay(`جارٍ رفع الصورة… ${Math.round(p * 100)}%`); });
    if (token !== state.uploadToken) { if (r.json.session_id) del(`/api/sessions/${r.json.session_id}`); return false; }
    if (r.status !== 200) {
      setOverlay("");
      setAfterError(r.json.error || (r.status === 0 ? "تعذّر الاتصال بالخادم. تحقق من الإنترنت وحاول مرة أخرى." : "تعذّر رفع الصورة. حاول مرة أخرى."));
      return false;
    }
    const j = r.json;
    state.sid = j.session_id; state.iw = j.width; state.ih = j.height;
    state.transparent = !!j.transparent; state.autoBg = j.auto_bg || ""; state.aiReady = false;
    syncUI(); updateSizeInfo();
    return true;
  }

  el.file.addEventListener("change", () => { const f = el.file.files[0]; el.file.value = ""; setFile(f); });
  el.changeImg.addEventListener("click", () => el.file.click());
  el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); } });
  window.addEventListener("dragenter", (e) => { e.preventDefault(); if (!el.uploadStage.hidden) el.drop.classList.add("over"); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("dragleave", (e) => { if (!e.relatedTarget) el.drop.classList.remove("over"); });
  window.addEventListener("drop", (e) => {
    e.preventDefault(); el.drop.classList.remove("over");
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  /* ====================== المعاينة المباشرة ====================== */
  function buildForm() {
    const fd = new FormData(), m = mode(), s = sizeState();
    fd.append("remove_bg", m !== "none" ? "true" : "false");
    fd.append("bg_mode", m === "ai" ? "ai" : "solid");
    fd.append("bg_tolerance", el.tol.value);
    fd.append("bg_color", m === "solid" ? state.bgColor : "");
    fd.append("erode_px", el.erode.value);
    fd.append("alpha_mode", el.alphaKeep.checked ? "keep" : "threshold");
    fd.append("alpha_threshold", el.alpha.value);
    fd.append("width_cm", s.ok ? String(s.w) : "0");
    fd.append("height_cm", s.ok ? String(s.h) : "0");
    return fd;
  }

  function schedulePreview(delay = 60) {
    clearTimeout(state.timer);
    state.timer = setTimeout(runPreview, delay);
  }

  async function runPreview() {
    if (!state.sid) return;
    const my = ++state.seq;
    if (state.ctrl) state.ctrl.abort();
    const ctrl = (state.ctrl = new AbortController());
    const m = mode();
    const firstAi = m === "ai" && !state.transparent && !state.aiReady;
    if (!state.hasAfter) setOverlay("جارٍ تجهيز المعاينة…");
    else if (firstAi) setOverlay("جارٍ تحليل الصورة بالذكاء الاصطناعي… (مرة واحدة، بضع ثوانٍ)");
    setBusy(true, "جارٍ التحديث…");
    try {
      const r = await fetch(`/api/sessions/${state.sid}/preview`, { method: "POST", body: buildForm(), signal: ctrl.signal, cache: "no-store" });
      if (my !== state.seq) return;
      if (r.status === 404) {  // انتهت الجلسة على الخادم: أعد رفع الصورة تلقائياً
        const ok = await uploadSession();
        if (ok && my === state.seq) schedulePreview(0);
        return;
      }
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        if (my !== state.seq) return;
        setOverlay("");
        setAfterError(j.error || "تعذّرت المعاينة. حاول مرة أخرى.");
        return;
      }
      const blob = await r.blob();
      if (my !== state.seq) return;
      if (m === "ai") state.aiReady = true;
      state.usedBg = r.headers.get("X-Bg-Color") || "";
      const url = URL.createObjectURL(blob);
      await new Promise((res) => { el.afterImg.onload = res; el.afterImg.onerror = res; el.afterImg.src = url; });
      if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
      state.afterUrl = url;
      state.hasAfter = true;
      setOverlay(""); setAfterError("");
      syncUI();
    } catch (e) {
      if (e.name === "AbortError") return;
      if (my !== state.seq) return;
      setOverlay("");
      setAfterError("تعذّر الاتصال بالخادم. تحقق من الإنترنت وحاول مرة أخرى.");
    } finally {
      if (my === state.seq) setBusy(false);
    }
  }

  /* ====================== الملف النهائي ====================== */
  function setProgress(frac, stage) {
    const p = Math.max(0, Math.min(100, Math.round(frac * 100)));
    el.barFill.style.width = p + "%"; el.bar.setAttribute("aria-valuenow", String(p)); el.pct.textContent = p + "%";
    if (stage) el.stage.textContent = stage;
  }
  function modalState(s, msg) {  // running | done | error
    show(el.finalProgress, s === "running"); show(el.finalDone, s === "done");
    show(el.finalError, s === "error"); show(el.closeModal, s !== "running");
    el.finalTitle.textContent = s === "done" ? "الملف النهائي جاهز" : s === "error" ? "تعذّرت المعالجة" : "جارٍ تجهيز الملف النهائي…";
    if (s === "error") el.finalError.textContent = msg || "حدث خطأ غير متوقع. حاول مرة أخرى.";
  }
  function fillNotes(list, warnings, notes) {
    list.replaceChildren();
    (warnings || []).forEach((t) => { const li = document.createElement("li"); li.className = "warn"; li.textContent = t; list.append(li); });
    (notes || []).forEach((t) => { const li = document.createElement("li"); li.textContent = t; list.append(li); });
  }
  function stopPoll() { if (state.poll) { clearTimeout(state.poll); state.poll = null; } }
  function finish(ok, msg) {
    stopPoll(); state.finalRunning = false; updateDownloadEnabled();
    if (!ok) modalState("error", msg);
  }

  function showFinal(job) {
    const i = job.info || {};
    state.jobId = job.id;
    const chips = [
      ["الأبعاد", `${i.output_px[0]} × ${i.output_px[1]} بكسل`], ["المقاس", `${i.output_cm[0]} × ${i.output_cm[1]} سم`],
      ["الدقة", `${i.dpi} DPI`], ["حجم الملف", fmtBytes(i.file_bytes)], ["التكبير", i.ai_upscale ? "Real-ESRGAN" : "بدون تكبير ذكي"],
    ];
    el.chips.replaceChildren(...chips.map(([k, v]) => {
      const li = document.createElement("li"); li.append(k + ": ");
      const b = document.createElement("b"); b.textContent = v; li.append(b); return li;
    }));
    fillNotes(el.finalNotes, job.warnings, job.notes);
    const href = `/api/jobs/${job.id}/result`;
    el.finalLink.href = href;
    modalState("done");
    finish(true);
    const a = document.createElement("a");  // تنزيل تلقائي (والرابط ظاهر إن منعه المتصفح)
    a.href = href; a.download = ""; document.body.append(a); a.click(); a.remove();
  }

  function pollJob(id) {
    let failures = 0;
    const tick = async () => {
      try {
        const r = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) { if (r.status === 404) return finish(false, j.error); throw new Error("http"); }
        failures = 0;
        if (j.status === "error") return finish(false, j.error);
        if (j.status === "done") return showFinal(j);
        const queued = j.status === "queued" && j.queue_position > 0;
        setProgress(j.progress || 0, queued ? `في الانتظار… دورك رقم ${j.queue_position}` : j.stage);
      } catch (_) {
        if (++failures >= 6) return finish(false, "انقطع الاتصال بالخادم. تحقق من الإنترنت وحاول مرة أخرى.");
      }
      state.poll = setTimeout(tick, 700);
    };
    tick();
  }

  async function startFinal(retried = false) {
    const s = sizeState();
    if (!state.sid || !s.ok || (state.finalRunning && !retried)) return;
    state.finalRunning = true; updateDownloadEnabled();
    if (state.jobId) { del(`/api/jobs/${state.jobId}`); state.jobId = null; }
    show(el.modal, true);
    modalState("running"); setProgress(0, "جارٍ البدء…");
    const fd = buildForm();
    fd.set("width_cm", String(s.w)); fd.set("height_cm", String(s.h));
    fd.append("session_id", state.sid);
    try {
      const r = await fetch("/api/process", { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (r.status === 404 && !retried) {
        setProgress(0, "جارٍ إعادة رفع الصورة…");
        if (await uploadSession()) return startFinal(true);
        return finish(false, "تعذّر إعادة رفع الصورة. حاول مرة أخرى.");
      }
      if (!r.ok || !j.job_id) return finish(false, j.error || "تعذّر بدء المعالجة. حاول مرة أخرى.");
      pollJob(j.job_id);
    } catch (_) {
      finish(false, "تعذّر الاتصال بالخادم. تحقق من الإنترنت وحاول مرة أخرى.");
    }
  }

  el.download.addEventListener("click", () => startFinal());
  el.closeModal.addEventListener("click", () => { show(el.modal, false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !el.modal.hidden && !state.finalRunning) show(el.modal, false); });

  window.addEventListener("pagehide", () => {
    if (state.sid) del(`/api/sessions/${state.sid}`);
    if (state.jobId) del(`/api/jobs/${state.jobId}`);
  });

  /* ====================== بدء ====================== */
  fetch("/api/config", { cache: "no-store" }).then((r) => r.json())
    .then((c) => { Object.assign(state.cfg, c); el.maxMb.textContent = state.cfg.max_upload_mb; updateSizeInfo(); })
    .catch(() => {});
  syncUI();
})();
