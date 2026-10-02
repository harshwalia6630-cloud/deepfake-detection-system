// YuNet face detection post-processing, ported from OpenCV's FaceDetectorYN.
// Works with onnxruntime-web (browser) and onnxruntime-node (tests): pass the `ort` module in.
// Verified against cv2.FaceDetectorYN via scripts/yunet_reference.py + tests/web_parity.mjs.

const STRIDES = [8, 16, 32];

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * @param rgba   Uint8ClampedArray / Uint8Array of RGBA pixels (canvas ImageData layout)
 * @returns      faces as [x, y, w, h, score], highest score first
 */
export async function detectFaces(ort, session, rgba, width, height, scoreThreshold = 0.7, nmsThreshold = 0.3) {
  const padH = (Math.floor((height - 1) / 32) + 1) * 32;
  const padW = (Math.floor((width - 1) / 32) + 1) * 32;
  const plane = padH * padW;
  const blob = new Float32Array(3 * plane); // zero padding on the bottom/right, like OpenCV
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4, o = y * padW + x;
      blob[o] = rgba[i + 2];             // OpenCV feeds BGR
      blob[plane + o] = rgba[i + 1];
      blob[2 * plane + o] = rgba[i];
    }
  }
  const out = await session.run({ input: new ort.Tensor("float32", blob, [1, 3, padH, padW]) });

  const faces = [];
  for (const s of STRIDES) {
    const cols = padW / s;
    const cls = out[`cls_${s}`].data, obj = out[`obj_${s}`].data, bbox = out[`bbox_${s}`].data;
    for (let idx = 0; idx < cls.length; idx++) {
      const score = Math.sqrt(Math.min(1, Math.max(0, cls[idx])) * Math.min(1, Math.max(0, obj[idx])));
      if (score < scoreThreshold) continue;
      const r = Math.floor(idx / cols), c = idx % cols;
      const cx = (c + bbox[idx * 4]) * s, cy = (r + bbox[idx * 4 + 1]) * s;
      const w = Math.exp(bbox[idx * 4 + 2]) * s, h = Math.exp(bbox[idx * 4 + 3]) * s;
      faces.push([cx - w / 2, cy - h / 2, w, h, score]);
    }
  }
  // greedy NMS on integer rects, as OpenCV does
  faces.sort((a, b) => b[4] - a[4]);
  const keep = [];
  const toRect = f => f.slice(0, 4).map(Math.trunc);
  for (const f of faces) {
    const rect = toRect(f);
    if (keep.every(k => iou(rect, toRect(k)) <= nmsThreshold)) keep.push(f);
  }
  return keep;
}

/** Largest face by area, mirroring FaceDetector.detect() in src/deepfake_detector/faces.py. */
export function largestFace(faces) {
  return faces.reduce((best, f) => (!best || f[2] * f[3] > best[2] * best[3] ? f : best), null);
}
