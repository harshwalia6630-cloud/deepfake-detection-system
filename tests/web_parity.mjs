// Checks the browser face-detection port against the Python/OpenCV reference.
// Needs local fixtures from `python scripts/yunet_reference.py` (they contain dataset
// frames, so they are not committed).   Run: node tests/web_parity.mjs

import { existsSync, readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { detectFaces } from "../web/js/yunet.js";

const root = new URL("../", import.meta.url);
const fixturePath = new URL("tests/fixtures/yunet_frames.json", root);
if (!existsSync(fixturePath)) {
  console.log("No fixtures; run `python scripts/yunet_reference.py` first.");
  process.exit(1);
}
const frames = JSON.parse(readFileSync(fixturePath));
const session = await ort.InferenceSession.create(new URL("web/model/face_detection_yunet.onnx", root).pathname.replace(/^\/(\w:)/, "$1"));

let maxErr = 0, faces = 0, failures = 0;
for (const f of frames) {
  const [h, w] = f.shape;
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = f.pixels_bgr[i * 3 + 2];
    rgba[i * 4 + 1] = f.pixels_bgr[i * 3 + 1];
    rgba[i * 4 + 2] = f.pixels_bgr[i * 3];
    rgba[i * 4 + 3] = 255;
  }
  const got = await detectFaces(ort, session, rgba, w, h);
  if (got.length !== f.faces.length) { failures++; console.log("FAIL count", got.length, f.faces.length); continue; }
  got.forEach((g, i) => { faces++; g.forEach((v, j) => { maxErr = Math.max(maxErr, Math.abs(v - f.faces[i][j])); }); });
}
console.log(`YuNet: ${faces} faces on ${frames.length} frames, max |js - python| = ${maxErr.toExponential(2)}`);
if (maxErr > 1e-3) failures++;
console.log(failures ? `${failures} FAILURE(S)` : "ALL PARITY CHECKS PASSED");
process.exit(failures ? 1 : 0);
