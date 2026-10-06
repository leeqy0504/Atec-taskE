"""Fit the fixed-camera Task E pixel-to-table homography offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from perception import _components, _sugar_full_component


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
        "object_1": _sugar_full_component(rgb) or max(sugar, key=lambda c: c["area"]),
        "object_2": max(mustard, key=lambda c: c["area"]),
        "object_3": max(banana, key=lambda c: c["area"]),
    }


def collect_correspondences(datasets: list[Path], max_trajectories: int | None) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    pixels: list[list[float]] = []
    worlds: list[list[float]] = []
    labels: list[str] = []
    rejected = 0
    for dataset in datasets:
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
    parser.add_argument(
        "--dataset", type=Path, action="append", dest="datasets",
        help="Demonstration HDF5 (repeat to combine independent calibration sets)",
    )
    parser.add_argument("--output", type=Path, default=Path("/root/gpufree-data/rgb_runs/task_e_calibration.json"))
    parser.add_argument("--max-trajectories", type=int, default=None)
    args = parser.parse_args()

    datasets = args.datasets or [Path("/root/gpufree-data/atec_task_e_compact_20/trajectory.hdf5")]
    pixel, world, labels, rejected = collect_correspondences(datasets, args.max_trajectories)
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
    object_matrices = {}
    object_fit_errors = {}
    for object_name in ("object_1", "object_2", "object_3"):
        indices = [i for i, label in enumerate(labels) if label == object_name]
        residual = world[indices] - predicted[indices] if indices else np.zeros((0, 2), dtype=np.float32)
        object_offsets[object_name] = residual.mean(axis=0).tolist() if len(residual) else [0.0, 0.0]
        if len(indices) >= 3:
            object_affine, object_mask = cv2.estimateAffine2D(
                pixel[indices], world[indices], method=cv2.RANSAC,
                ransacReprojThreshold=0.03, maxIters=5000, confidence=0.999,
            )
            if object_affine is None:
                object_affine = cv2.getAffineTransform(
                    pixel[indices[:3]].astype(np.float32),
                    world[indices[:3]].astype(np.float32),
                )
            object_matrix = np.eye(3, dtype=np.float64)
            object_matrix[:2, :] = object_affine
            object_matrices[object_name] = object_matrix.tolist()
            object_predicted = cv2.perspectiveTransform(
                pixel[indices].reshape(-1, 1, 2), object_matrix
            ).reshape(-1, 2)
            object_fit_errors[object_name] = {
                "samples": len(indices),
                "inliers": int(object_mask.sum()) if object_mask is not None else None,
                "mean_m": float(np.linalg.norm(object_predicted - world[indices], axis=1).mean()),
                "p95_m": float(np.percentile(np.linalg.norm(object_predicted - world[indices], axis=1), 95)),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "datasets": [str(dataset.resolve()) for dataset in datasets],
        "dataset_sha256": {str(dataset.resolve()): sha256(dataset) for dataset in datasets},
        "correspondences": int(len(pixel)),
        "inliers": int(mask.sum()) if mask is not None else 0,
        "rejected_merged_frames": int(rejected),
        "pixel_to_world_xy": matrix.tolist(),
        "object_pixel_to_world_xy": object_matrices,
        "object_world_offsets_m": object_offsets,
        "object_fit_error_m": object_fit_errors,
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
