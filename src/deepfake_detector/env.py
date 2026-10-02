"""Runtime setup that must happen before TensorFlow is imported."""

import os
import sys
from pathlib import Path


def _register_pip_cuda_dlls():
    """On Windows, expose CUDA/cuDNN DLLs installed via `nvidia-*-cu11` pip wheels."""
    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def setup():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _register_pip_cuda_dlls()
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf
