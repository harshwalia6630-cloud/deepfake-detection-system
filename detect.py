"""Detect whether a video is a deepfake.

Examples:
    python detect.py path/to/video.mp4
    python detect.py path/to/video.mp4 --output annotated.mp4
    python detect.py 0 --show            # webcam
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detector import env  # noqa: E402

tf = env.setup()
from deepfake_detector.detector import DeepfakeDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file path or webcam index")
    ap.add_argument("--model", default=str(ROOT / "models/deepfake_effnetb0.weights.h5"))
    ap.add_argument("--every", type=int, default=1, help="analyse every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="display annotated video")
    ap.add_argument("--output", help="write annotated video to this path")
    args = ap.parse_args()

    detector = DeepfakeDetector(tf, args.model)
    r = detector.analyze(args.source, args.every, args.max_frames, args.show, args.output)
    print(f"Verdict:          {r.verdict}")
    if r.frames_with_face:
        print(f"Fake probability: {r.fake_probability:.3f}  (threshold {detector.video_threshold:.3f})")
        print(f"Confidence:       {r.confidence:.1%}")
    print(f"Frames analysed:  {r.frames_with_face}/{r.frames_read} with a face")
    print(f"Throughput:       {r.fps:.1f} FPS")


if __name__ == "__main__":
    main()
