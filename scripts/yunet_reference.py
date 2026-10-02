"""Reference YuNet post-processing in numpy, checked against cv2.FaceDetectorYN.

The browser has no OpenCV, so web/js/yunet.js re-implements OpenCV's decoding
(anchor-free boxes on strides 8/16/32, score = sqrt(cls * obj), integer-rect
NMS). This script proves the decoding logic matches OpenCV on real video
frames before it is ported, and writes fixtures for the JS port.

It also saves web/model/face_detection_yunet.onnx with dynamic input size,
since the upstream file declares a fixed 640x640 input.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from deepfake_detector.faces import DEFAULT_MODEL  # noqa: E402
from extract_faces import load_split_map  # noqa: E402

STRIDES = (8, 16, 32)
DYNAMIC = ROOT / "web/model/face_detection_yunet.onnx"


def make_dynamic(src=DEFAULT_MODEL, dst=DYNAMIC):
    m = onnx.load(str(src))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    for d, name in ((dims[2], "height"), (dims[3], "width")):
        d.ClearField("dim_value")
        d.dim_param = name
    for out in m.graph.output:
        for i, d in enumerate(out.type.tensor_type.shape.dim):
            d.ClearField("dim_value")
            d.dim_param = f"{out.name}_{i}"
    del m.graph.value_info[:]
    onnx.save(m, str(dst))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def detect(sess, image_bgr, score_threshold=0.7, nms_threshold=0.3):
    h, w = image_bgr.shape[:2]
    pad_h, pad_w = ((h - 1) // 32 + 1) * 32, ((w - 1) // 32 + 1) * 32
    blob = np.zeros((1, 3, pad_h, pad_w), np.float32)
    blob[0, :, :h, :w] = image_bgr.transpose(2, 0, 1)
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input": blob})))
    faces = []
    for s in STRIDES:
        cols, rows = pad_w // s, pad_h // s
        cls, obj = outs[f"cls_{s}"].reshape(-1), outs[f"obj_{s}"].reshape(-1)
        bbox = outs[f"bbox_{s}"].reshape(-1, 4)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        for idx in np.flatnonzero(score >= score_threshold):
            r, c = divmod(idx, cols)
            cx, cy = (c + bbox[idx, 0]) * s, (r + bbox[idx, 1]) * s
            bw, bh = np.exp(bbox[idx, 2]) * s, np.exp(bbox[idx, 3]) * s
            faces.append((float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh), float(score[idx])))
    # OpenCV runs NMS on integer rects, ordered by score
    faces.sort(key=lambda f: -f[4])
    keep = []
    for f in faces:
        rect = tuple(int(v) for v in f[:4])
        if all(iou(rect, tuple(int(v) for v in k[:4])) <= nms_threshold for k in keep):
            keep.append(f)
    return keep


def main(n_videos=12, frames_per_video=6):
    make_dynamic()
    sess = ort.InferenceSession(str(DYNAMIC), providers=["CPUExecutionProvider"])
    split_of = load_split_map(ROOT / "data/splits")
    videos = [v for cls in ("real", "Deepfakes") for v in sorted((ROOT / "data/raw" / cls).glob("*.mp4"))
              if split_of.get(v.stem) == "test"][:: max(1, 280 // n_videos)][:n_videos]

    max_err, n_faces, fixtures = 0.0, 0, []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in np.linspace(0, total - 1, frames_per_video, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            scale = min(1.0, 480 / frame.shape[1])
            small = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
            ref = cv2.FaceDetectorYN.create(str(DEFAULT_MODEL), "", (small.shape[1], small.shape[0]), 0.7, 0.3)
            _, cv_faces = ref.detect(small)
            cv_faces = [] if cv_faces is None else [tuple(f[:4]) + (f[-1],) for f in cv_faces]
            ours = detect(sess, small)
            assert len(ours) == len(cv_faces), (v.name, fi, len(ours), len(cv_faces))
            for a, b in zip(sorted(ours, key=lambda f: -f[4]), sorted(cv_faces, key=lambda f: -f[4])):
                max_err = max(max_err, max(abs(x - y) for x, y in zip(a, b)))
                n_faces += 1
            if len(fixtures) < 8:
                fixtures.append({"shape": list(small.shape), "pixels_bgr": small.flatten().tolist(),
                                 "faces": [list(f) for f in ours]})
    print(f"{n_faces} faces on {len(videos) * frames_per_video} frames: max |ours - OpenCV| = {max_err:.2e}")
    assert max_err < 1e-2
    (ROOT / "tests/fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/fixtures/yunet_frames.json").write_text(json.dumps(fixtures))


if __name__ == "__main__":
    main()
