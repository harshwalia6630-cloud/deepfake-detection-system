import { BrowserDetector } from "./detector.js";

const $ = id => document.getElementById(id);
const FRAMES = 32;
let det = null;

// ---------- helpers
const nextFrame = () => new Promise(r => requestAnimationFrame(r));
function seek(video, t) {
  return new Promise((resolve, reject) => {
    const done = () => { clearTimeout(timer); video.removeEventListener("seeked", done); resolve(); };
    const timer = setTimeout(() => { video.removeEventListener("seeked", done); reject(new Error("seek timeout")); }, 8000);
    video.addEventListener("seeked", done);
    video.currentTime = t;
  });
}
const pct = p => `${(p * 100).toFixed(1)}%`;

function drawBox(canvas, box, prob, threshold) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!box) return;
  const fake = prob >= threshold;
  const color = fake ? "#dc2626" : "#16a34a";
  const lw = Math.max(2, canvas.width / 320);
  ctx.lineWidth = lw;
  ctx.strokeStyle = color;
  ctx.strokeRect(box[0], box[1], box[2], box[3]);
  const text = `${fake ? "FAKE" : "REAL"} ${prob.toFixed(2)}`;
  ctx.font = `600 ${Math.max(14, canvas.width / 40)}px system-ui, sans-serif`;
  const tw = ctx.measureText(text).width, th = Math.max(14, canvas.width / 40) * 1.4;
  const y = Math.max(0, box[1] - th);
  ctx.fillStyle = color;
  ctx.fillRect(box[0] - lw / 2, y, tw + 12, th);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, box[0] + 6, y + th * 0.72);
}

function setVerdict(el, labelEl, subEl, label, sub) {
  el.classList.remove("fake", "real");
  if (label === "FAKE") el.classList.add("fake");
  if (label === "REAL") el.classList.add("real");
  labelEl.textContent = label;
  subEl.textContent = sub;
}

// ---------- upload analysis
function renderTimeline(points, threshold) {
  const svg = $("timeline"), W = 300, H = 110, pad = 5;
  const y = p => H - pad - p * (H - 2 * pad);
  const x = i => (points.length < 2 ? W / 2 : pad + (i / (FRAMES - 1)) * (W - 2 * pad));
  const valid = points.map((p, i) => [p, i]).filter(([p]) => p !== null);
  const line = valid.map(([p, i]) => `${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(" ");
  svg.innerHTML = `
    <line x1="0" x2="${W}" y1="${y(threshold)}" y2="${y(threshold)}" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>
    <polyline points="${line}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    ${valid.map(([p, i]) => `<circle cx="${x(i)}" cy="${y(p)}" r="2.6" fill="${p >= threshold ? "var(--fake)" : "var(--real)"}"><title>frame ${i + 1}: ${pct(p)}</title></circle>`).join("")}`;
}

function addThumb(crop, prob, threshold) {
  const div = document.createElement("div");
  const fake = prob >= threshold;
  div.className = `face ${fake ? "fake" : "real"}`;
  const c = document.createElement("canvas");
  c.width = crop.width; c.height = crop.height;
  c.getContext("2d").putImageData(crop, 0, 0);
  const s = document.createElement("span");
  s.textContent = prob.toFixed(2);
  div.append(c, s);
  $("faces").appendChild(div);
}

async function analyzeFile(file) {
  if (!det || !file) return;
  const video = $("video"), overlay = $("overlay");
  const cfg = det.config;
  $("drop").hidden = true;
  $("analysis").hidden = false;
  $("faces-wrap").hidden = false;
  $("faces").innerHTML = "";
  $("timeline").innerHTML = "";
  $("another").disabled = true;
  video.controls = false;
  setVerdict($("verdict"), $("verdict-label"), $("verdict-sub"), "Analysing…", "Loading video");
  $("threshold-note").textContent = `dashed line = decision threshold (${cfg.video_threshold.toFixed(2)})`;

  if (video.src) URL.revokeObjectURL(video.src);
  video.src = URL.createObjectURL(file);
  try {
    await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = () => rej(new Error("unsupported video")); });
  } catch {
    setVerdict($("verdict"), $("verdict-label"), $("verdict-sub"), "Can't read video", "Try an MP4 (H.264) or WebM file.");
    $("another").disabled = false;
    return;
  }
  const w = video.videoWidth, h = video.videoHeight, dur = video.duration;
  overlay.width = w; overlay.height = h;

  const probs = [], points = [];
  let totalMs = 0;
  for (let i = 0; i < FRAMES; i++) {
    const t = dur * (0.02 + 0.96 * (i / (FRAMES - 1)));
    // no requestAnimationFrame here: it is paused in background tabs, and the frame is ready after "seeked"
    try { await seek(video, t); } catch { points.push(null); continue; }
    const r = await det.analyzeFrame(video, w, h);
    totalMs += r.ms;
    points.push(r.prob);
    if (r.prob !== null) {
      probs.push(r.prob);
      addThumb(r.crop, r.prob, cfg.frame_threshold);
    }
    drawBox(overlay, r.box, r.prob ?? 0, cfg.frame_threshold);
    renderTimeline(points, cfg.video_threshold);
    $("analysis-progress").firstElementChild.style.width = `${((i + 1) / FRAMES) * 100}%`;
    $("verdict-sub").textContent = `${i + 1} / ${FRAMES} frames · face found in ${probs.length}`;
  }

  const v = det.verdict(probs);
  const sub = v.score === null
    ? "No face was detected in the sampled frames."
    : `${pct(v.confidence)} confidence · mean fake probability ${v.score.toFixed(3)} · ${probs.length} faces · ${(totalMs / FRAMES).toFixed(0)} ms/frame`;
  setVerdict($("verdict"), $("verdict-label"), $("verdict-sub"), v.label, sub);
  video.controls = true;
  $("another").disabled = false;
}

function resetUpload() {
  $("analysis").hidden = true;
  $("faces-wrap").hidden = true;
  $("drop").hidden = false;
  $("file").value = "";
  $("analysis-progress").firstElementChild.style.width = "0";
  const video = $("video");
  video.pause();
}

// ---------- webcam
let camRunning = false, stream = null;

async function startCam() {
  const cam = $("cam"), overlay = $("cam-overlay");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } });
  } catch (e) {
    $("cam-sub").textContent = "Camera permission was denied or no camera is available.";
    return;
  }
  cam.srcObject = stream;
  await cam.play();
  $("cam-placeholder").hidden = true;
  $("cam-stop").disabled = false;
  overlay.width = cam.videoWidth; overlay.height = cam.videoHeight;
  camRunning = true;
  let smoothed = null, frames = 0, t0 = performance.now();
  const thr = det.config.frame_threshold;
  while (camRunning) {
    const r = await det.analyzeFrame(cam, cam.videoWidth, cam.videoHeight);
    frames++;
    if (r.prob !== null) smoothed = smoothed === null ? r.prob : 0.8 * smoothed + 0.2 * r.prob;
    drawBox(overlay, r.box, smoothed ?? 0, thr);
    if (r.prob === null) setVerdict($("cam-verdict"), $("cam-label"), $("cam-sub"), "No face", "Move into the frame and face the camera");
    else setVerdict($("cam-verdict"), $("cam-label"), $("cam-sub"), det.label(smoothed),
                    `smoothed fake probability ${smoothed.toFixed(3)} (threshold ${thr.toFixed(2)})`);
    const elapsed = (performance.now() - t0) / 1000;
    $("cam-fps").textContent = `${(frames / elapsed).toFixed(1)} FPS`;
    $("cam-ms").textContent = `${r.ms.toFixed(0)} ms`;
    if (elapsed > 3) { frames = 0; t0 = performance.now(); }
    await nextFrame();
  }
}

function stopCam() {
  camRunning = false;
  stream?.getTracks().forEach(t => t.stop());
  $("cam").srcObject = null;
  $("cam-overlay").getContext("2d").clearRect(0, 0, 1e4, 1e4);
  $("cam-placeholder").hidden = false;
  $("cam-stop").disabled = true;
  setVerdict($("cam-verdict"), $("cam-label"), $("cam-sub"), "–", "Start the webcam to begin");
}

// ---------- wiring
document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(t => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
  $("pane-upload").hidden = tab.dataset.tab !== "upload";
  $("pane-webcam").hidden = tab.dataset.tab !== "webcam";
  if (tab.dataset.tab !== "webcam" && camRunning) stopCam();
}));
$("file").addEventListener("change", e => analyzeFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => analyzeFile(e.dataTransfer.files[0]));
$("another").addEventListener("click", resetUpload);
$("cam-start").addEventListener("click", startCam);
$("cam-stop").addEventListener("click", stopCam);
drop.classList.add("disabled");

(async () => {
  const bar = $("load-progress").firstElementChild;
  try {
    det = await BrowserDetector.load("model/", p => {
      if (p === null) { $("status").textContent = "Downloading model (16 MB)…"; return; }
      bar.style.width = `${p * 100}%`;
      $("status").textContent = `Downloading model… ${(p * 100).toFixed(0)}%`;
    });
    // warm-up so the first real frame isn't slow
    const warm = new OffscreenCanvas(320, 240);
    warm.getContext("2d").fillRect(0, 0, 320, 240);
    await det.findFace(warm, 320, 240);
    await det.classify(new ImageData(det.size, det.size));
    bar.style.width = "0";
    $("status").textContent = `Models ready · ONNX Runtime Web (WASM, ${det.threads} thread${det.threads > 1 ? "s" : ""})`;
    $("cam-runtime").textContent = `WASM × ${det.threads}`;
    drop.classList.remove("disabled");
    $("cam-start").disabled = false;
  } catch (err) {
    console.error(err);
    $("status").textContent = "Could not load the models. Please refresh.";
  }
})();
