(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
  const ALLOWED_EXT = /\.(png|jpe?g|webp)$/i;

  const state = {
    file: null,
    jobId: null,
    poll: null,
    busy: false,
    cfg: { max_upload_mb: 20, dpi: 300, min_side_cm: 1, max_side_cm: 120, max_output_mp: 80 },
  };

  const el = {
    drop: $("#drop"), file: $("#file"), dropEmpty: $("#dropEmpty"), dropFile: $("#dropFile"),
    thumb: $("#thumb"), fileName: $("#fileName"), fileDetails: $("#fileDetails"), fileError: $("#fileError"),
    maxMb: $("#maxMb"),
    w: $("#w"), h: $("#h"), sizeHint: $("#sizeHint"), sizeError: $("#sizeError"),
    removeBg: $("#removeBg"), erode: $("#erode"), erodeOut: $("#erodeOut"),
    alpha: $("#alpha"), alphaOut: $("#alphaOut"), alphaKeep: $("#alphaKeep"),
    go: $("#go"),
    progressCard: $("#progressCard"), bar: $("#bar"), barFill: $("#barFill"), stage: $("#stage"),
    pct: $("#pct"), liveNotes: $("#liveNotes"),
    errorCard: $("#errorCard"), errorText: $("#errorText"),
    resultCard: $("#resultCard"), resultNotes: $("#resultNotes"),
    imgBefore: $("#imgBefore"), imgAfter: $("#imgAfter"),
    swatches: $("#swatches"), customColor: $("#customColor"), customDot: $("#customDot"), customSw: $("#customSw"),
    chips: $("#chips"), download: $("#download"), again: $("#again"),
  };

  /* ---------- أدوات ---------- */
  const fmtBytes = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " ميجابايت" : Math.max(1, Math.round(n / 1024)) + " كيلوبايت");

  // يقبل الأرقام العربية/الفارسية والفاصلة العربية: ٦٠٫٥ → 60.5
  function parseNum(raw) {
    let s = String(raw || "").trim();
    if (!s) return NaN;
    s = s
      .replace(/[٠-٩]/g, (d) => d.charCodeAt(0) - 0x0660)
      .replace(/[۰-۹]/g, (d) => d.charCodeAt(0) - 0x06f0)
      .replace(/[٫,،]/g, ".");
    return /^\d*\.?\d+$/.test(s) ? parseFloat(s) : NaN;
  }

  function show(node, on = true) { node.hidden = !on; }
  function setMsg(node, text) { node.textContent = text || ""; show(node, !!text); }

  /* ---------- الملف ---------- */
  function setFile(f) {
    setMsg(el.fileError, "");
    if (!f) return;
    const typeOk = ALLOWED_TYPES.includes(f.type) || (!f.type && ALLOWED_EXT.test(f.name));
    if (!typeOk) {
      setMsg(el.fileError, "صيغة الملف غير مدعومة. الصيغ المسموحة: PNG وJPG وWebP.");
      return;
    }
    const maxBytes = state.cfg.max_upload_mb * 1048576;
    if (f.size > maxBytes) {
      setMsg(el.fileError, `حجم الملف ${fmtBytes(f.size)} يتجاوز الحد الأقصى (${state.cfg.max_upload_mb} ميجابايت). صغّر الصورة وحاول مجدداً.`);
      return;
    }
    if (f.size === 0) {
      setMsg(el.fileError, "الملف فارغ. اختر صورة أخرى.");
      return;
    }
    state.file = f;
    if (el.thumb.dataset.url) URL.revokeObjectURL(el.thumb.dataset.url);
    const url = URL.createObjectURL(f);
    el.thumb.dataset.url = url;
    el.thumb.onload = () => {
      el.fileDetails.textContent = `${el.thumb.naturalWidth}×${el.thumb.naturalHeight} بكسل — ${fmtBytes(f.size)}`;
    };
    el.thumb.onerror = () => {
      state.file = null;
      show(el.dropFile, false); show(el.dropEmpty, true);
      setMsg(el.fileError, "تعذّر قراءة الصورة. تأكد أن الملف سليم وليس تالفاً.");
      refresh();
    };
    el.thumb.src = url;
    el.fileName.textContent = f.name;
    el.fileDetails.textContent = fmtBytes(f.size);
    show(el.dropEmpty, false);
    show(el.dropFile, true);
    refresh();
  }

  el.file.addEventListener("change", () => { setFile(el.file.files[0]); el.file.value = ""; });
  el.drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); }
  });
  ["dragenter", "dragover"].forEach((t) =>
    el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("over"); })
  );
  ["dragleave", "drop"].forEach((t) =>
    el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("over"); })
  );
  el.drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  // منع المتصفح من فتح الصورة لو أُفلتت خارج المنطقة
  ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => { if (!el.drop.contains(e.target)) e.preventDefault(); }));

  /* ---------- المقاس ---------- */
  function readSize() {
    const w = parseNum(el.w.value), h = parseNum(el.h.value);
    const { min_side_cm: mn, max_side_cm: mx, max_output_mp: mp, dpi } = state.cfg;
    const empty = !el.w.value.trim() || !el.h.value.trim();
    if (empty) return { ok: false, msg: "", hint: "أدخل المقاس لعرض أبعاد الملف الناتج بالبكسل." };
    if (!(w > 0) || !(h > 0)) return { ok: false, msg: "المقاس غير صحيح. أدخل العرض والطول بالسنتيمتر كأرقام." };
    if (w < mn || h < mn) return { ok: false, msg: `المقاس صغير جداً. أقل مقاس مسموح ${mn} سم لكل جهة.` };
    if (w > mx || h > mx) return { ok: false, msg: `المقاس كبير جداً. أقصى مقاس مسموح ${mx} سم لكل جهة.` };
    const pw = Math.max(1, Math.round((w / 2.54) * dpi)), ph = Math.max(1, Math.round((h / 2.54) * dpi));
    if (pw * ph > mp * 1e6) {
      const side = Math.round((Math.sqrt(mp * 1e6) / dpi) * 2.54);
      return { ok: false, msg: `مقاس الطباعة كبير على الخادم: الملف الناتج سيكون ${pw}×${ph} بكسل والحد الأقصى ${mp} ميجابكسل (يعادل تقريباً ${side}×${side} سم). صغّر المقاس.` };
    }
    return { ok: true, w, h, hint: `الملف الناتج: <b>${pw} × ${ph}</b> بكسل عند ${dpi} DPI (≈ ${((pw * ph) / 1e6).toFixed(1)} ميجابكسل).` };
  }

  function refresh() {
    const s = readSize();
    el.sizeHint.innerHTML = s.hint || el.sizeHint.innerHTML;
    if (s.hint) show(el.sizeHint, true);
    setMsg(el.sizeError, s.ok ? "" : s.msg);
    if (!s.ok && s.msg) show(el.sizeHint, false);
    el.go.disabled = !(state.file && s.ok) || state.busy;
  }
  [el.w, el.h].forEach((i) => i.addEventListener("input", refresh));
  el.h.addEventListener("keydown", (e) => { if (e.key === "Enter" && !el.go.disabled) el.go.click(); });

  /* ---------- الخيارات ---------- */
  function syncOptions() {
    const e = +el.erode.value;
    el.erodeOut.textContent = e === 0 ? "بدون تصغير" : e === 1 ? "1 بكسل" : `${e} بكسل`;
    const a = +el.alpha.value;
    el.alpha.disabled = el.alphaKeep.checked;
    el.alphaOut.textContent = el.alphaKeep.checked ? "بدون تغيير" : `${a < 35 ? "خفيفة" : a <= 65 ? "معتدلة" : "عالية"} (${a}%)`;
  }
  [el.erode, el.alpha, el.alphaKeep].forEach((i) => i.addEventListener("input", syncOptions));
  syncOptions();

  /* ---------- التقدم والنتيجة ---------- */
  function setProgress(frac, stage) {
    const p = Math.max(0, Math.min(100, Math.round(frac * 100)));
    el.barFill.style.width = p + "%";
    el.bar.setAttribute("aria-valuenow", String(p));
    el.pct.textContent = p + "%";
    if (stage) el.stage.textContent = stage;
  }

  function renderNotes(list, warnings, notes) {
    list.replaceChildren();
    (warnings || []).forEach((t) => { const li = document.createElement("li"); li.className = "warn"; li.textContent = t; list.append(li); });
    (notes || []).forEach((t) => { const li = document.createElement("li"); li.textContent = t; list.append(li); });
  }

  function stopPoll() { if (state.poll) { clearTimeout(state.poll); state.poll = null; } }

  function showError(msg) {
    stopPoll();
    state.busy = false;
    show(el.progressCard, false);
    show(el.resultCard, false);
    el.errorText.textContent = msg || "حدث خطأ غير متوقع. حاول مرة أخرى.";
    show(el.errorCard, true);
    el.errorCard.scrollIntoView({ behavior: "smooth", block: "center" });
    refresh();
  }

  function showResult(job) {
    stopPoll();
    state.busy = false;
    const id = job.id, bust = Date.now();
    el.imgBefore.src = `/api/jobs/${id}/before?t=${bust}`;
    el.imgAfter.src = `/api/jobs/${id}/preview?t=${bust}`;
    el.download.href = `/api/jobs/${id}/result`;
    const i = job.info || {};
    const chips = [
      ["الأبعاد", `${i.output_px[0]} × ${i.output_px[1]} بكسل`],
      ["المقاس", `${i.output_cm[0]} × ${i.output_cm[1]} سم`],
      ["الدقة", `${i.dpi} DPI`],
      ["حجم الملف", fmtBytes(i.file_bytes)],
      ["التكبير", i.ai_upscale ? "Real-ESRGAN" : "بدون تكبير ذكي"],
    ];
    el.chips.replaceChildren(...chips.map(([k, v]) => {
      const li = document.createElement("li");
      li.append(k + ": ");
      const b = document.createElement("b"); b.textContent = v; li.append(b);
      return li;
    }));
    renderNotes(el.resultNotes, job.warnings, job.notes);
    show(el.progressCard, false);
    show(el.errorCard, false);
    show(el.resultCard, true);
    refresh();
    el.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function startPolling(id) {
    state.jobId = id;
    let failures = 0;
    const tick = async () => {
      try {
        const r = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) {
          if (r.status === 404) return showError(j.error || "انتهت صلاحية هذه المعالجة. ارفع الصورة من جديد.");
          throw new Error("http " + r.status);
        }
        failures = 0;
        if (j.status === "error") return showError(j.error);
        renderNotes(el.liveNotes, j.warnings, j.notes);
        if (j.status === "done") return showResult(j);
        const queued = j.status === "queued" && j.queue_position > 0;
        setProgress(j.progress || 0, queued ? `في الانتظار… دورك رقم ${j.queue_position}` : j.stage);
      } catch (_) {
        if (++failures >= 6) return showError("انقطع الاتصال بالخادم. تحقق من الإنترنت وحاول مرة أخرى.");
      }
      state.poll = setTimeout(tick, 700);
    };
    tick();
  }

  el.go.addEventListener("click", () => {
    const s = readSize();
    if (!state.file || !s.ok || state.busy) return;
    state.busy = true;
    el.go.disabled = true;
    show(el.errorCard, false);
    show(el.resultCard, false);
    renderNotes(el.liveNotes, [], []);
    setProgress(0, "جارٍ رفع الصورة…");
    show(el.progressCard, true);
    el.progressCard.scrollIntoView({ behavior: "smooth", block: "center" });

    if (state.jobId) { fetch(`/api/jobs/${state.jobId}`, { method: "DELETE", keepalive: true }).catch(() => {}); state.jobId = null; }

    const fd = new FormData();
    fd.append("file", state.file, state.file.name);
    fd.append("width_cm", String(s.w));
    fd.append("height_cm", String(s.h));
    fd.append("remove_bg", el.removeBg.checked ? "true" : "false");
    fd.append("erode_px", el.erode.value);
    fd.append("alpha_mode", el.alphaKeep.checked ? "keep" : "threshold");
    fd.append("alpha_threshold", el.alpha.value);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/process");
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(e.loaded / e.total, `جارٍ رفع الصورة… ${Math.round((e.loaded / e.total) * 100)}%`);
    };
    xhr.upload.onload = () => setProgress(0, "تم الرفع، جارٍ البدء…");
    xhr.onload = () => {
      const j = xhr.response || {};
      if (xhr.status >= 200 && xhr.status < 300 && j.job_id) {
        setProgress(0, "بدء المعالجة…");
        startPolling(j.job_id);
      } else {
        showError(j.error || (xhr.status === 413 ? `حجم الملف كبير جداً. الحد الأقصى ${state.cfg.max_upload_mb} ميجابايت.` : "تعذّر بدء المعالجة. حاول مرة أخرى."));
      }
    };
    xhr.onerror = () => showError("تعذّر الاتصال بالخادم. تحقق من الإنترنت وحاول مرة أخرى.");
    xhr.send(fd);
  });

  /* ---------- ألوان المعاينة ---------- */
  function applyStageColor(color) {
    document.querySelectorAll("[data-stage]").forEach((st) => {
      if (!color) { st.classList.add("checker"); st.classList.remove("solid"); st.style.backgroundColor = ""; }
      else { st.classList.remove("checker"); st.classList.add("solid"); st.style.backgroundColor = color; }
    });
  }
  function activate(btn) {
    el.swatches.querySelectorAll(".sw").forEach((b) => { b.classList.toggle("active", b === btn); if (b.getAttribute("role") === "radio") b.setAttribute("aria-checked", String(b === btn)); });
  }
  el.swatches.addEventListener("click", (e) => {
    const b = e.target.closest("button.sw");
    if (!b) return;
    activate(b);
    applyStageColor(b.dataset.color);
  });
  el.customColor.addEventListener("input", () => {
    const c = el.customColor.value;
    el.customDot.classList.remove("rainbow");
    el.customDot.style.background = c;
    activate(el.customSw);
    applyStageColor(c);
  });

  /* ---------- جديد ---------- */
  el.again.addEventListener("click", () => {
    if (state.jobId) { fetch(`/api/jobs/${state.jobId}`, { method: "DELETE", keepalive: true }).catch(() => {}); state.jobId = null; }
    state.file = null;
    if (el.thumb.dataset.url) { URL.revokeObjectURL(el.thumb.dataset.url); el.thumb.dataset.url = ""; }
    el.thumb.removeAttribute("src");
    show(el.dropFile, false); show(el.dropEmpty, true);
    show(el.resultCard, false); show(el.errorCard, false);
    el.imgBefore.removeAttribute("src"); el.imgAfter.removeAttribute("src");
    refresh();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  window.addEventListener("pagehide", () => {
    if (state.jobId) fetch(`/api/jobs/${state.jobId}`, { method: "DELETE", keepalive: true }).catch(() => {});
  });

  /* ---------- بدء ---------- */
  fetch("/api/config", { cache: "no-store" })
    .then((r) => r.json())
    .then((c) => { Object.assign(state.cfg, c); el.maxMb.textContent = state.cfg.max_upload_mb; refresh(); })
    .catch(() => {});
  refresh();
})();
