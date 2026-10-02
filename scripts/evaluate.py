"""Evaluate a trained model on the held-out test split.

Reports frame-level and video-level metrics, tunes the decision threshold on the
validation split, and writes reports/metrics.json plus ROC / confusion-matrix plots.
The tuned threshold and aggregation method are saved to models/inference_config.json
for the video detector.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detector import env  # noqa: E402

tf = env.setup()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from deepfake_detector.data import make_dataset, read_manifest  # noqa: E402
from deepfake_detector.model import load_trained  # noqa: E402
from deepfake_detector.scoring import group_by_video, tune_threshold  # noqa: E402


def predict(model, faces, split, image_size, batch_size):
    paths, labels, videos = read_manifest(faces, split)
    ds = make_dataset(tf, paths, labels, image_size, batch_size)
    probs = model.predict(ds, verbose=1).ravel()
    return np.array(labels), probs, videos


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "n": int(len(y)),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y, scores), 4),
        "false_positive_rate": round(fp / (fp + tn), 4),
        "false_positives": int(fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def plot(y_frame, p_frame, y_video, s_video, cm, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for y, s, name in ((y_frame, p_frame, "frame"), (y_video, s_video, "video")):
        fpr, tpr, _ = roc_curve(y, s)
        ax[0].plot(fpr, tpr, label=f"{name}-level (AUC {roc_auc_score(y, s):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - test split")
    ax[0].legend(loc="lower right")

    ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                       color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "fake"], yticklabels=["real", "fake"],
              xlabel="Predicted", ylabel="Actual", title="Video-level confusion matrix (tuned threshold)")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation.png", dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models/deepfake_effnetb0.weights.h5"))
    ap.add_argument("--faces", default=str(ROOT / "data/faces"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model = load_trained(tf, args.model, args.image_size)
    y_val, p_val, v_val = predict(model, args.faces, "val", args.image_size, args.batch_size)
    y_test, p_test, v_test = predict(model, args.faces, "test", args.image_size, args.batch_size)

    # every choice below is made on validation data; the test split is only scored
    frame_t = tune_threshold(y_val, p_val)

    choices = {}
    for method in ("mean", "confidence"):
        _, yv, sv = group_by_video(v_val, y_val, p_val, method)
        t = tune_threshold(yv, sv)
        choices[method] = (metrics(yv, sv, t)["accuracy"], metrics(yv, sv, t)["auc"], t)
    method = max(choices, key=lambda m: choices[m][:2])
    video_t = choices[method][2]

    _, yt, st = group_by_video(v_test, y_test, p_test, method)
    report = {
        "test_frame_level@0.5": metrics(y_test, p_test, 0.5),
        "test_frame_level@tuned": metrics(y_test, p_test, frame_t),
        "test_video_level@0.5": metrics(yt, st, 0.5),
        "test_video_level@tuned": metrics(yt, st, video_t),
        "video_aggregation": method,
    }
    base_fpr = report["test_frame_level@0.5"]["false_positive_rate"]
    tuned_fpr = report["test_frame_level@tuned"]["false_positive_rate"]
    report["frame_false_positive_reduction_pct"] = round(100 * (base_fpr - tuned_fpr) / base_fpr, 1) if base_fpr else 0.0

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (ROOT / "models/inference_config.json").write_text(json.dumps({
        "frame_threshold": round(frame_t, 4),
        "video_threshold": round(video_t, 4),
        "aggregation": method,
        "image_size": args.image_size,
    }, indent=2))
    plot(y_test, p_test, yt, st, report["test_video_level@tuned"]["confusion_matrix"], out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
