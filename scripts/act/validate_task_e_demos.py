"""Validate complete Task E demonstration trajectories.

The validator is intentionally independent of Isaac Sim.  It checks the
contract consumed by ACT training: successful full-task episodes, aligned
temporal datasets, absolute 8-joint observations, environment-scaled actions,
and RGB frame alignment.

Usage:
    python scripts/act/validate_task_e_demos.py \
        --input datasets/atec_task_e_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _json_attr(group: h5py.Group, name: str, default):
    value = group.attrs.get(name, default)
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _bool_attr(group: h5py.Group, name: str, default=False) -> bool:
    value = group.attrs.get(name, default)
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _display(value) -> str:
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    return str(value)


def validate(input_path: Path, require_rgb: bool = True) -> int:
    h5_path = input_path / "trajectory.hdf5" if input_path.is_dir() else input_path
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    total = 0
    successes = 0
    failed = []
    lengths: list[int] = []
    initial_positions: dict[str, list[list[float]]] = {}
    object_completion_counts = {f"object_{i}": 0 for i in (1, 2, 3)}

    with h5py.File(h5_path, "r") as data:
        traj_keys = sorted(data.keys(), key=lambda key: int(key.split("_")[1]))
        for key in traj_keys:
            total += 1
            group = data[key]
            reasons: list[str] = []
            required = ("obs", "actions", "qvel", "ee_pos", "ee_quat")
            missing = [name for name in required if name not in group]
            if missing:
                reasons.append(f"missing datasets: {','.join(missing)}")
                failed.append((key, reasons))
                continue

            shape_lengths = {name: int(group[name].shape[0]) for name in required}
            if len(set(shape_lengths.values())) != 1:
                reasons.append(f"length mismatch: {shape_lengths}")
            steps = shape_lengths["obs"]
            lengths.append(steps)
            if group["obs"].ndim != 2 or group["obs"].shape[1] != 8:
                reasons.append(f"obs shape is {group['obs'].shape}, expected (T,8)")
            if group.attrs.get("observation_type", "") != "absolute_joint_position":
                reasons.append("observation_type is not absolute_joint_position")
            if group["actions"].ndim != 2 or group["actions"].shape[1] != 8:
                reasons.append(f"actions shape is {group['actions'].shape}, expected (T,8)")

            rgb = group.get("images/rgb")
            if require_rgb and rgb is None:
                reasons.append("missing images/rgb")
            elif rgb is not None and (rgb.ndim != 4 or rgb.shape[0] != steps or rgb.shape[-1] != 3):
                reasons.append(f"RGB shape is {rgb.shape}, expected (T,H,W,3)")

            success = _bool_attr(group, "success")
            objects = _json_attr(group, "objects_in_basket", {})
            order = _json_attr(group, "object_order", [])
            if not success:
                reasons.append("success is false")
            if int(group.attrs.get("completed_object_count", -1)) != 3:
                reasons.append("completed_object_count != 3")
            if order != [1, 2, 3]:
                reasons.append(f"object_order is {order}, expected [1,2,3]")
            if objects != {"object_1": True, "object_2": True, "object_3": True}:
                reasons.append(f"objects_in_basket is {objects}")
            for object_name in object_completion_counts:
                object_completion_counts[object_name] += int(bool(objects.get(object_name, False)))

            terminal_reset = _bool_attr(group, "terminal_reset_encountered")
            termination = _json_attr(group, "termination_reason", [])
            if terminal_reset and "basket_success" not in termination:
                reasons.append(f"unexpected terminal reset: {termination}")
            if _bool_attr(group, "cross_reset"):
                reasons.append("cross_reset is true")

            if group["obs"].dtype.kind not in "fc" or not np.isfinite(group["obs"][:]).all():
                reasons.append("obs is not finite numeric data")
            if not np.isfinite(group["actions"][:]).all():
                reasons.append("actions contains non-finite values")

            default_jpos = np.asarray(_json_attr(group, "default_joint_pos", []), dtype=np.float32)
            scale = float(group.attrs.get("action_scale", 0.5))
            trace = _json_attr(group, "step_trace", [])
            if default_jpos.shape != (8,):
                reasons.append("default_joint_pos attribute is missing or not length 8")
            elif len(trace) != steps:
                reasons.append(f"step_trace length {len(trace)} != {steps}")
            else:
                targets = np.asarray([row.get("joint_target", []) for row in trace], dtype=np.float32)
                if targets.shape != (steps, 8):
                    reasons.append(f"joint_target trace shape is {targets.shape}, expected ({steps},8)")
                else:
                    expected_actions = (targets - default_jpos[None, :]) / scale
                    if not np.allclose(group["actions"][:], expected_actions, atol=2e-5, rtol=2e-5):
                        reasons.append("actions do not match (joint_target-default_joint_pos)/action_scale")

            place_diag = _json_attr(group, "place_diagnostics", {})
            for object_name in ("object_1", "object_2", "object_3"):
                diag = place_diag.get(object_name, {})
                if not bool(diag.get("target_z_monotonic", False)):
                    reasons.append(f"{object_name} PLACE Z target is not monotonic")
                if float(diag.get("target_lateral_max_m", 1.0)) > 1e-6:
                    reasons.append(f"{object_name} PLACE target moved laterally")

            positions = _json_attr(group, "initial_object_positions_local_m", {})
            for object_name, position in positions.items():
                initial_positions.setdefault(object_name, []).append(position)

            if reasons:
                failed.append((key, reasons))
            else:
                successes += 1

    print(f"total_trajectories: {total}")
    print(f"successful_trajectories: {successes}")
    print(f"failed_trajectories: {len(failed)}")
    print("trajectory_lengths:", lengths)
    if lengths:
        print(f"length_range: {min(lengths)}..{max(lengths)}")
    print("per_object_completion:", object_completion_counts)
    print("initial_position_ranges:")
    for object_name in sorted(initial_positions):
        values = np.asarray(initial_positions[object_name], dtype=np.float64)
        print(f"  {object_name}: min={_display(values.min(axis=0))} max={_display(values.max(axis=0))}")
    if failed:
        print("failure_details:")
        for key, reasons in failed:
            print(f"  {key}: {'; '.join(reasons)}")
    print("rgb_action_state_alignment: checked")
    print("cross_reset_or_failure_present:", bool(failed))
    return 0 if total > 0 and not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task E ACT demonstration HDF5 data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--allow_missing_rgb", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate(args.input, require_rgb=not args.allow_missing_rgb))


if __name__ == "__main__":
    main()
