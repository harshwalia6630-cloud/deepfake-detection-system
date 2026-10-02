"""Export the classifier to ONNX for the browser demo and check it matches Keras.

Writes
    web/model/deepfake_effnetb0.onnx     EfficientNetB0 classifier (input: 1x224x224x3 RGB, 0-255)
    web/model/face_detection_yunet.onnx  copy of the YuNet face detector
    web/model/config.json                thresholds and preprocessing settings for the JS app
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detector import env  # noqa: E402

tf = env.setup()
import onnxruntime as ort  # noqa: E402
import tf2onnx  # noqa: E402

from deepfake_detector.data import make_dataset, read_manifest  # noqa: E402
from deepfake_detector.faces import DEFAULT_MODEL as YUNET  # noqa: E402
from deepfake_detector.model import load_trained  # noqa: E402

WEB = ROOT / "web" / "model"


def main(n_check=512):
    WEB.mkdir(parents=True, exist_ok=True)
    # TF32 (on by default for Ampere+ GPUs) would add ~1e-3 noise to the Keras reference
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = load_trained(tf, ROOT / "models/deepfake_effnetb0.weights.h5")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="face"),)
    out = WEB / "deepfake_effnetb0.onnx"
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(out))

    # parity on real test faces
    paths, labels, _ = read_manifest(ROOT / "data/faces", "test")
    idx = np.random.default_rng(0).choice(len(paths), n_check, replace=False)
    ds = make_dataset(tf, [paths[i] for i in idx], [labels[i] for i in idx], 224, 64)
    x = np.concatenate([b.numpy() for b, _ in ds])
    keras_p = model.predict(x, verbose=0).ravel()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_p = np.concatenate([sess.run(None, {"face": x[i:i + 64]})[0].ravel() for i in range(0, len(x), 64)])
    diff = float(np.abs(keras_p - onnx_p).max())
    print(f"ONNX vs Keras on {n_check} test faces: max |diff| = {diff:.2e}, "
          f"decision agreement = {np.mean((keras_p >= 0.5) == (onnx_p >= 0.5)):.4f}")
    assert diff < 1e-3

    shutil.copy(YUNET, WEB / "face_detection_yunet.onnx")
    cfg = json.loads((ROOT / "models/inference_config.json").read_text())
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    bench = json.loads((ROOT / "reports/benchmark.json").read_text())
    (WEB / "config.json").write_text(json.dumps({
        **cfg,
        "face_margin": 1.3,
        "detect_width": 480,
        "face_score_threshold": 0.7,
        "onnx_input": "face",
        "metrics": {
            "frame_accuracy": metrics["test_frame_level@tuned"]["accuracy"],
            "video_accuracy": metrics["test_video_level@tuned"]["accuracy"],
            "video_auc": metrics["test_video_level@tuned"]["auc"],
            "fp_reduction_pct": metrics["frame_false_positive_reduction_pct"],
            "gpu_fps": bench["mean_fps"],
        },
    }, indent=2))
    for p in sorted(WEB.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
