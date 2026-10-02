// In-browser version of src/deepfake_detector/detector.py:
//   frame -> downscale to 480px -> YuNet -> largest face -> 1.3x square crop -> 224x224 -> EfficientNetB0 (ONNX)

// onnxruntime-web 1.30.0 is self-hosted: threaded WASM spawns workers, which browsers only allow from the same origin
import * as ort from "../vendor/ort/ort.wasm.min.mjs";
import { detectFaces, largestFace } from "./yunet.js";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
// multithreaded WASM needs cross-origin isolation (COOP/COEP headers, set in vercel.json)
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) {            // compressed responses often omit content-length
    onProgress?.(null);
    return new Uint8Array(await res.arrayBuffer());
  }
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, loaded);
    loaded += value.length;
    onProgress?.(loaded / total);
  }
  return buf;
}

export class BrowserDetector {
  static async load(base = "model/", onProgress = () => {}) {
    const config = await fetch(base + "config.json").then(r => r.json());
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const yunetBytes = await fetchWithProgress(base + "face_detection_yunet.onnx");
    const clfBytes = await fetchWithProgress(base + "deepfake_effnetb0.onnx", onProgress);
    const [yunet, classifier] = await Promise.all([
      ort.InferenceSession.create(yunetBytes, opts),
      ort.InferenceSession.create(clfBytes, opts),
    ]);
    return new BrowserDetector(config, yunet, classifier);
  }

  constructor(config, yunet, classifier) {
    this.config = config;
    this.yunet = yunet;
    this.classifier = classifier;
    this.size = config.image_size;
    this.detectCanvas = new OffscreenCanvas(1, 1);
    this.cropCanvas = new OffscreenCanvas(this.size, this.size);
    this.threads = ort.env.wasm.numThreads;
  }

  label(prob, threshold = this.config.frame_threshold) {
    return prob >= threshold ? "FAKE" : "REAL";
  }

  /** Detect the main face in a video/canvas/image source. Returns [x, y, w, h, score] in source pixels, or null. */
  async findFace(source, width, height) {
    const scale = Math.min(1, this.config.detect_width / width);
    const dw = Math.round(width * scale), dh = Math.round(height * scale);
    const c = this.detectCanvas;
    if (c.width !== dw || c.height !== dh) { c.width = dw; c.height = dh; }
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, dw, dh);
    const faces = await detectFaces(ort, this.yunet, ctx.getImageData(0, 0, dw, dh).data, dw, dh,
                                    this.config.face_score_threshold);
    const best = largestFace(faces);
    return best && [best[0] / scale, best[1] / scale, best[2] / scale, best[3] / scale, best[4]];
  }

  /** Square crop around the face with a margin, resized to the model input (FaceDetector.crop in Python). */
  cropFace(source, width, height, box) {
    const [x, y, bw, bh] = box;
    const cx = x + bw / 2, cy = y + bh / 2, side = Math.max(bw, bh) * this.config.face_margin;
    const x0 = Math.trunc(Math.max(0, cx - side / 2)), y0 = Math.trunc(Math.max(0, cy - side / 2));
    const x1 = Math.trunc(Math.min(width, cx + side / 2)), y1 = Math.trunc(Math.min(height, cy + side / 2));
    if (x1 <= x0 || y1 <= y0) return null;
    const ctx = this.cropCanvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, this.size, this.size);
    return ctx.getImageData(0, 0, this.size, this.size);
  }

  async classify(imageData) {
    const n = this.size * this.size, px = imageData.data, input = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {                 // NHWC, RGB, 0-255 (EfficientNet rescales internally)
      input[i * 3] = px[i * 4];
      input[i * 3 + 1] = px[i * 4 + 1];
      input[i * 3 + 2] = px[i * 4 + 2];
    }
    const feeds = { [this.config.onnx_input]: new ort.Tensor("float32", input, [1, this.size, this.size, 3]) };
    const out = await this.classifier.run(feeds);
    return out[this.classifier.outputNames[0]].data[0];
  }

  /** Full pipeline for one frame. */
  async analyzeFrame(source, width, height) {
    const t0 = performance.now();
    const box = await this.findFace(source, width, height);
    if (!box) return { box: null, prob: null, ms: performance.now() - t0 };
    const crop = this.cropFace(source, width, height, box);
    if (!crop) return { box: null, prob: null, ms: performance.now() - t0 };
    const prob = await this.classify(crop);
    return { box, prob, crop, ms: performance.now() - t0 };
  }

  /** Video-level verdict from frame probabilities (mean aggregation + tuned video threshold). */
  verdict(probs) {
    if (!probs.length) return { label: "NO FACE FOUND", score: null, confidence: null };
    const score = probs.reduce((a, b) => a + b, 0) / probs.length;
    const label = this.label(score, this.config.video_threshold);
    return { label, score, confidence: label === "FAKE" ? score : 1 - score };
  }
}
