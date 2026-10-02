"""Turn per-frame fake probabilities into video-level decisions."""

import numpy as np


def aggregate(probs, method="confidence"):
    """Combine frame probabilities into one video score.

    "mean"        plain average.
    "confidence"  average weighted by how far each frame is from 0.5, so blurry or
                  occluded frames the model is unsure about count for less.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return float("nan")
    if method == "mean":
        return float(probs.mean())
    weights = np.abs(probs - 0.5) * 2 + 1e-3
    return float(np.average(probs, weights=weights))


def group_by_video(videos, labels, probs, method="confidence"):
    """Return (video_ids, video_labels, video_scores) aggregated from frame-level arrays."""
    order = {}
    for v, y, p in zip(videos, labels, probs):
        entry = order.setdefault(v, [y, []])
        entry[1].append(p)
    ids = list(order)
    y = np.array([order[v][0] for v in ids])
    s = np.array([aggregate(order[v][1], method) for v in ids])
    return ids, y, s


def tune_threshold(y_true, scores):
    """Pick the threshold that maximises balanced accuracy on a validation set.

    Every threshold between two adjacent scores gives identical predictions, so
    the optimum is really an interval. Returning its midpoint (a max-margin
    choice) keeps the threshold away from the edge of the validation data, where
    it would overfit.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    cands = np.unique(scores)
    pos, neg = y_true == 1, y_true == 0
    bacc = np.array([((scores[pos] >= t).mean() + (scores[neg] < t).mean()) / 2 for t in cands])
    best = np.flatnonzero(bacc >= bacc.max() - 1e-9)
    # longest contiguous run of optimal candidates
    runs = np.split(best, np.flatnonzero(np.diff(best) > 1) + 1)
    run = max(runs, key=len)
    lo = cands[run[0] - 1] if run[0] > 0 else 0.0
    hi = cands[run[-1]]
    return float((lo + hi) / 2)
