"""Measure end-to-end throughput (decode + face detection + CNN) on held-out test videos.

Every frame is processed, so the reported FPS is the real-time rate the full
pipeline sustains on this machine. Results go to reports/benchmark.json.
"""

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detector import env  # noqa: E402

tf = env.setup()
from deepfake_detector.detector import DeepfakeDetector  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from extract_faces import load_split_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--videos", type=int, default=20, help="test videos to sample (half real, half fake)")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split_of = load_split_map(ROOT / "data/splits")
    rng = random.Random(args.seed)
    picks = []
    for cls in ("real", "Deepfakes"):
        vids = sorted(v for v in (Path(args.raw) / cls).glob("*.mp4") if split_of.get(v.stem) == "test")
        picks += [(v, cls) for v in rng.sample(vids, min(args.videos // 2, len(vids)))]

    det = DeepfakeDetector(tf)
    fps, correct, total_frames = [], 0, 0
    for video, cls in picks:
        r = det.analyze(video, max_frames=args.max_frames)
        truth = "REAL" if cls == "real" else "FAKE"
        correct += r.verdict == truth
        fps.append(r.fps)
        total_frames += r.frames_read
        print(f"{video.name:28s} truth={truth} pred={r.verdict:5s} p_fake={r.fake_probability:.3f} {r.fps:5.1f} FPS")

    gpu = tf.config.list_physical_devices("GPU")
    gpu_name = tf.config.experimental.get_device_details(gpu[0]).get("device_name") if gpu else "none"
    report = {
        "videos": len(picks),
        "frames_processed": total_frames,
        "mean_fps": round(float(np.mean(fps)), 1),
        "min_fps": round(float(np.min(fps)), 1),
        "video_accuracy_on_sample": round(correct / len(picks), 3),
        "hardware": {"gpu": gpu_name, "cpu": platform.processor(), "tensorflow": tf.__version__},
    }
    (ROOT / "reports/benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
