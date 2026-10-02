"""Extract face crops from FaceForensics++ videos into train/val/test folders.

Splits follow the official FaceForensics++ pairs (data/splits/*.json), so every
identity appears in exactly one split and no person leaks from train to test.

Output layout:
    data/faces/<split>/<real|fake>/<video>_<frame>.jpg
    data/faces/manifest.csv
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepfake_detector.faces import FaceDetector, sample_frame_indices  # noqa: E402

_detector = None


def load_split_map(splits_dir):
    """Map every FF++ video stem (e.g. '953' or '953_974') to its split name."""
    mapping = {}
    for split in ("train", "val", "test"):
        for a, b in json.loads((Path(splits_dir) / f"{split}.json").read_text()):
            for stem in (a, b, f"{a}_{b}", f"{b}_{a}"):
                mapping[stem] = split
    return mapping


def _init_worker():
    global _detector
    cv2.setNumThreads(1)
    _detector = FaceDetector()


def process_video(job):
    video, label, split, out_root, n_frames, size = job
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for idx in sample_frame_indices(total, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        face, box = _detector.extract(frame, size=size)
        if face is None:
            continue
        dest = Path(out_root) / split / label / f"{video.stem}_{idx:04d}.jpg"
        cv2.imwrite(str(dest), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append((dest.relative_to(out_root).as_posix(), label, split, video.stem, idx, round(box[4], 3)))
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--frames-per-video", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    split_of = load_split_map(args.splits)
    out = Path(args.out)
    jobs = []
    for class_dir in sorted(Path(args.raw).iterdir()):
        label = "real" if class_dir.name == "real" else "fake"
        for video in sorted(class_dir.glob("*.mp4")):
            split = split_of.get(video.stem)
            if split is None:
                continue
            (out / split / label).mkdir(parents=True, exist_ok=True)
            jobs.append((video, label, split, out, args.frames_per_video, args.size))

    print(f"Extracting faces from {len(jobs)} videos with {args.workers} workers")
    rows = []
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for r in tqdm(pool.imap_unordered(process_video, jobs, chunksize=4), total=len(jobs), unit="video"):
            rows.extend(r)

    rows.sort()
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split", "video", "frame", "face_score"])
        w.writerows(rows)

    counts = {}
    for _, label, split, *_ in rows:
        counts[(split, label)] = counts.get((split, label), 0) + 1
    print(f"Saved {len(rows)} face crops")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split:5s} {label:4s} {n}")


if __name__ == "__main__":
    main()
