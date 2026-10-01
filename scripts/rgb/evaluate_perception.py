"""Evaluate RGB detections against offline demonstration metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from perception import RGBPerception


def decode(value):
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value) if isinstance(value, str) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/root/gpufree-data/atec_task_e_compact_20/trajectory.hdf5"))
    parser.add_argument("--calibration", type=Path, default=Path("/root/gpufree-data/rgb_runs/task_e_calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("/root/gpufree-data/rgb_runs/perception_eval.json"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-trajectories", type=int, default=None)
    args = parser.parse_args()
    detector = RGBPerception(args.calibration)
    totals = defaultdict(int)
    errors: dict[str, list[float]] = defaultdict(list)
    jitter: dict[str, list[float]] = defaultdict(list)
    previous: dict[str, np.ndarray] = {}

    with h5py.File(args.dataset, "r") as h5:
        keys = sorted(h5.keys(), key=lambda key: int(key.split("_")[1]))
        if args.max_trajectories is not None:
            keys = keys[:args.max_trajectories]
        for key in keys:
            group = h5[key]
            detector.reset()
            trace = decode(group.attrs["step_trace"])
            for step in range(0, len(trace), max(1, args.stride)):
                # Static/pre-grasp frames are the valid object-center benchmark;
                # carried objects intentionally leave their original image location.
                state = trace[step].get("state", "")
                if state not in {"INIT", "PRE_GRASP", "REACH", "CLOSE"}:
                    continue
                detections = detector.detect(group["images/rgb"][step])
                current_name = f"object_{trace[step].get('object', -1)}"
                if current_name not in {"object_1", "object_2", "object_3"}:
                    continue
                for object_name in ("object_1", "object_2", "object_3"):
                    if object_name != current_name:
                        continue
                    totals[object_name] += 1
                    detection = detections[object_name]
                    if not detection.visible or detection.world_xy is None:
                        continue
                    reference = np.asarray(trace[step]["object_local_m"], dtype=np.float64)
                    actual = np.asarray(detection.world_xy, dtype=np.float64)
                    errors[object_name].append(float(np.linalg.norm(actual - reference[:2])))
                    if object_name in previous:
                        jitter[object_name].append(float(np.linalg.norm(actual - previous[object_name])))
                    previous[object_name] = actual

    summary = {"dataset": str(args.dataset.resolve()), "calibration": str(args.calibration.resolve()), "objects": {}}
    for name in ("object_1", "object_2", "object_3"):
        values = np.asarray(errors[name], dtype=np.float64)
        jitter_values = np.asarray(jitter[name], dtype=np.float64)
        summary["objects"][name] = {
            "evaluated": int(totals[name]),
            "detections": int(len(values)),
            "detection_rate": float(len(values) / max(totals[name], 1)),
            "world_error_mean_m": float(values.mean()) if len(values) else None,
            "world_error_p95_m": float(np.percentile(values, 95)) if len(values) else None,
            "jitter_p95_m": float(np.percentile(jitter_values, 95)) if len(jitter_values) else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
