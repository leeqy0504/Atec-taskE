"""Fit the fixed-camera Task E pixel-to-table homography offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from perception import _components


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value):
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value) if isinstance(value, str) else value


def _separated_components(rgb: np.ndarray) -> dict[str, dict] | None:
    """Return only unambiguous first-frame components.

    Sugar and mustard can touch in the camera image.  A merged component is
    useful for tracking, but it must not be used as a calibration point: its
    centroid is not the center of either physical object.
    """
    hsv = cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([15, 70, 70], np.uint8), np.array([45, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    parts = _components(mask, min_area=180.0)
    sugar = [c for c in parts if 350.0 <= c["area"] <= 1800.0 and c["aspect"] <= 1.50]
    mustard = [c for c in parts if c["area"] >= 1700.0 and c["aspect"] >= 2.10]
    used = {id(c) for c in sugar + mustard}
    banana = [c for c in parts if id(c) not in used and 800.0 <= c["area"] <= 1800.0]
    if not sugar or not mustard or not banana:
        return None
    return {
        "object_1": max(sugar, key=lambda c: c["area"]),
        "object_2": max(mustard, key=lambda c: c["area"]),
        "object_3": max(banana, key=lambda c: c["area"]),
    }


def collect_correspondences(dataset: Path, max_trajectories: int | None) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    pixels: list[list[float]] = []
    worlds: list[list[float]] = []
    labels: list[str] = []
    rejected = 0
    with h5py.File(dataset, "r") as h5:
        keys = sorted(h5.keys(), key=lambda key: int(key.split("_")[1]))
        if max_trajectories is not None:
            keys = keys[:max_trajectories]
        for key in keys:
            group = h5[key]
            initial = decode(group.attrs["initial_object_positions_local_m"])
            components = _separated_components(group["images/rgb"][0])
            if components is None:
                rejected += 1
                continue
            for object_name in ("object_1", "object_2", "object_3"):
                component = components.get(object_name)
                if component is None:
                    continue
                pixels.append(component["xy"])
                worlds.append(initial[object_name][:2])
                labels.append(object_name)
    if len(pixels) < 4:
        raise RuntimeError(f"Only {len(pixels)} RGB/GT correspondences found; need at least 4")
    return np.asarray(pixels, dtype=np.float32), np.asarray(worlds, dtype=np.float32), labels, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/root/gpufree-data/atec_task_e_compact_20/trajectory.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("/root/gpufree-data/rgb_runs/task_e_calibration.json"))
    parser.add_argument("--max-trajectories", type=int, default=None)
    args = parser.parse_args()

    pixel, world, labels, rejected = collect_correspondences(args.dataset, args.max_trajectories)
    # The visible table region is narrow and nearly fronto-parallel.  Fitting
    # a full projective model from only a dozen object centroids is ill
    # conditioned (the denominator can approach zero at valid pixels), while
    # an affine homography is stable and is sufficient over this ROI.
    affine, mask = cv2.estimateAffine2D(
        pixel, world, method=cv2.RANSAC, ransacReprojThreshold=0.05,
        maxIters=5000, confidence=0.999,
    )
    if affine is None:
        raise RuntimeError("Could not fit pixel-to-world affine mapping")
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = affine
    predicted = cv2.perspectiveTransform(pixel.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - world, axis=1)
    object_offsets = {}
    for object_name in ("object_1", "object_2", "object_3"):
        indices = [i for i, label in enumerate(labels) if label == object_name]
        residual = world[indices] - predicted[indices] if indices else np.zeros((0, 2), dtype=np.float32)
        object_offsets[object_name] = residual.mean(axis=0).tolist() if len(residual) else [0.0, 0.0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "correspondences": int(len(pixel)),
        "inliers": int(mask.sum()) if mask is not None else 0,
        "rejected_merged_frames": int(rejected),
        "pixel_to_world_xy": matrix.tolist(),
        "object_world_offsets_m": object_offsets,
        "fit_error_m": {
            "mean": float(errors.mean()),
            "p95": float(np.percentile(errors, 95)),
            "max": float(errors.max()),
        },
        "quality": {
            "usable_for_runtime": bool(len(pixel) >= 8 and mask is not None and int(mask.sum()) >= 8),
            "meets_world_error_target": bool(float(np.percentile(errors, 95)) <= 0.015),
            "target_world_error_m": 0.015,
        },
        "detector": {
            "yellow_hsv": [15, 70, 70, 45, 255, 255],
            "pink_hsv": [135, 35, 45, 179, 255, 255],
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
