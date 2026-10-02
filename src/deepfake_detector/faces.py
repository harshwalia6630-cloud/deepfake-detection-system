"""Face detection and cropping with OpenCV's YuNet detector."""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "face" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """Finds the most prominent face in a frame and returns a square crop around it.

    Detection runs on a downscaled copy of the frame (`detect_width`) because
    YuNet's cost grows with input size while accuracy on talking-head video
    barely changes; boxes are mapped back to full resolution before cropping.
    """

    def __init__(self, model_path=DEFAULT_MODEL, score_threshold=0.7, detect_width=480):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. Download it with:\n"
                f"  curl -L -o {model_path} {YUNET_URL}"
            )
        self.detect_width = detect_width
        self._net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold)
        self._input_size = None

    def detect(self, frame):
        """Return the largest face box as (x, y, w, h, score) in full-frame coordinates, or None."""
        h, w = frame.shape[:2]
        scale = min(1.0, self.detect_width / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, faces = self._net.detect(small)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, bw, bh = (best[:4] / scale).tolist()
        return x, y, bw, bh, float(best[-1])

    @staticmethod
    def crop(frame, box, size=224, margin=1.3):
        """Square crop centred on `box`, enlarged by `margin` to keep blending boundaries in view."""
        x, y, bw, bh = box[:4]
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * margin
        h, w = frame.shape[:2]
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 <= x0 or y1 <= y0:
            return None
        face = frame[y0:y1, x0:x1]
        return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)

    def extract(self, frame, size=224, margin=1.3):
        """Detect and crop in one step. Returns (crop_bgr, box) or (None, None)."""
        box = self.detect(frame)
        if box is None:
            return None, None
        return self.crop(frame, box, size, margin), box


def sample_frame_indices(n_frames, n_samples):
    """Evenly spaced frame indices across a video, skipping the very first/last frames."""
    if n_frames <= 0:
        return []
    n = min(n_samples, n_frames)
    return np.linspace(0, n_frames - 1, n + 2, dtype=int)[1:-1].tolist() if n_frames > n + 2 else list(range(n))
