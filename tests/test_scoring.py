import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepfake_detector.faces import FaceDetector, sample_frame_indices  # noqa: E402
from deepfake_detector.scoring import aggregate, group_by_video, tune_threshold  # noqa: E402


def test_mean_aggregation():
    assert aggregate([0.2, 0.4, 0.6], "mean") == pytest.approx(0.4)


def test_confidence_aggregation_downweights_uncertain_frames():
    # two confident fake frames outweigh three near-0.5 frames
    probs = [0.95, 0.9, 0.45, 0.5, 0.48]
    assert aggregate(probs, "confidence") > aggregate(probs, "mean")


def test_aggregate_empty_is_nan():
    assert np.isnan(aggregate([]))


def test_group_by_video():
    ids, y, s = group_by_video(["a", "a", "b"], [0, 0, 1], [0.1, 0.3, 0.9], "mean")
    assert ids == ["a", "b"]
    assert y.tolist() == [0, 1]
    assert s == pytest.approx([0.2, 0.9])


def test_tune_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.55, 0.6, 0.7, 0.8, 0.9])
    t = tune_threshold(y, s)
    assert ((s >= t) == y.astype(bool)).all()


def test_sample_frame_indices():
    idx = sample_frame_indices(300, 10)
    assert len(idx) == 10
    assert idx == sorted(idx)
    assert 0 < idx[0] and idx[-1] < 299
    assert sample_frame_indices(5, 10) == list(range(5))
    assert sample_frame_indices(0, 10) == []


def test_crop_is_square_and_clipped():
    frame = np.zeros((100, 200, 3), np.uint8)
    face = FaceDetector.crop(frame, (180, 80, 40, 40), size=64)
    assert face.shape == (64, 64, 3)
