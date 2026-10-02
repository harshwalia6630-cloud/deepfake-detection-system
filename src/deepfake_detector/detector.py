"""Frame-by-frame deepfake detection on video files or webcam streams."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .faces import FaceDetector
from .model import load_trained
from .scoring import aggregate

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VideoResult:
    verdict: str
    fake_probability: float
    confidence: float
    frames_read: int
    frames_with_face: int
    fps: float
    frame_probs: list = field(repr=False, default_factory=list)


class DeepfakeDetector:
    def __init__(self, tf, model_path=ROOT / "models/deepfake_effnetb0.weights.h5",
                 config_path=ROOT / "models/inference_config.json"):
        cfg = {"frame_threshold": 0.5, "video_threshold": 0.5, "aggregation": "mean", "image_size": 224}
        if Path(config_path).exists():
            cfg.update(json.loads(Path(config_path).read_text()))
        self.frame_threshold = cfg["frame_threshold"]
        self.video_threshold = cfg["video_threshold"]
        self.aggregation = cfg["aggregation"]
        self.image_size = cfg["image_size"]
        self.faces = FaceDetector()

        model = load_trained(tf, model_path, self.image_size)
        # a traced graph avoids Keras' per-call Python overhead, which dominates at batch size 1
        self._predict = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
        self._tf = tf
        self._predict(tf.zeros((1, self.image_size, self.image_size, 3)))  # warm-up / trace

    def predict_faces(self, faces_bgr):
        batch = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces_bgr]).astype(np.float32)
        return self._predict(self._tf.constant(batch)).numpy().ravel()

    def label(self, prob, threshold=None):
        return "FAKE" if prob >= (self.frame_threshold if threshold is None else threshold) else "REAL"

    def analyze(self, source, every=1, max_frames=None, show=False, output=None):
        """Run detection over a video. `source` is a file path or a webcam index."""
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        writer = None
        probs, smoothed = [], None
        read = 0
        start = time.perf_counter()
        try:
            while max_frames is None or read < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                if (read - 1) % every:
                    continue
                face, box = self.faces.extract(frame, size=self.image_size)
                if face is not None:
                    p = float(self.predict_faces([face])[0])
                    probs.append(p)
                    smoothed = p if smoothed is None else 0.8 * smoothed + 0.2 * p
                if show or output:
                    annotated = self._draw(frame, box if face is not None else None, smoothed, read, start)
                    if output:
                        if writer is None:
                            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
                            h, w = annotated.shape[:2]
                            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                        writer.write(annotated)
                    if show:
                        cv2.imshow("Deepfake Detection (q to quit)", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start
        score = aggregate(probs, self.aggregation) if probs else float("nan")
        verdict = "NO FACE FOUND" if not probs else self.label(score, self.video_threshold)
        confidence = float("nan") if not probs else (score if verdict == "FAKE" else 1 - score)
        return VideoResult(verdict, score, confidence, read, len(probs), read / elapsed if elapsed else 0.0, probs)

    def _draw(self, frame, box, prob, n, start):
        out = frame.copy()
        if box is not None and prob is not None:
            lab = self.label(prob)
            color = (0, 0, 255) if lab == "FAKE" else (0, 200, 0)
            x, y, w, h = map(int, box[:4])
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f"{lab} {prob:.2f}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        fps = n / max(time.perf_counter() - start, 1e-6)
        cv2.putText(out, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out
