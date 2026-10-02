"""Download a subset of FaceForensics++ (c23) videos without pulling the full 18 GB archive.

The archive is read over HTTP range requests, so only the requested folders are
transferred. By default this fetches the 1,000 original YouTube videos (`real`)
and the 1,000 `Deepfakes` manipulations.

FaceForensics++ is released for research use under its own terms of use:
https://github.com/ondyari/FaceForensics. Make sure you comply with them.

Usage:
    python scripts/download_ffpp.py --out data/raw --methods Deepfakes
"""

import argparse
import concurrent.futures as cf
import os
import shutil
import time
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

DEFAULT_URL = (
    "https://huggingface.co/datasets/bitmind/FaceForensicsC23/resolve/main/"
    "FaceForensics++_C23.zip"
)
ROOT = "FaceForensics++_C23"


def wanted_members(names, methods, limit):
    groups = {"real": f"{ROOT}/real/"}
    groups.update({m: f"{ROOT}/fake/{m}/" for m in methods})
    picked = []
    for label, prefix in groups.items():
        videos = sorted(n for n in names if n.startswith(prefix) and n.endswith(".mp4"))
        if limit:
            videos = videos[:limit]
        picked += [(label, n) for n in videos]
    return picked


def fetch(url, member, dest, retries=5):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            # one RemoteZip per worker call keeps HTTP sessions independent across threads
            with RemoteZip(url) as z, z.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(tmp, dest)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--methods", nargs="+", default=["Deepfakes"],
                    help="fake subsets: Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter")
    ap.add_argument("--limit", type=int, default=0, help="max videos per class (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    with RemoteZip(args.url) as z:
        names = z.namelist()
    jobs = wanted_members(names, args.methods, args.limit)
    out = Path(args.out)
    print(f"Fetching {len(jobs)} videos into {out}")

    with cf.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, args.url, m, out / label / Path(m).name) for label, m in jobs]
        for f in tqdm(cf.as_completed(futures), total=len(futures), unit="video"):
            f.result()


if __name__ == "__main__":
    main()
