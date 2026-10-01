"""Compare ACT predictions with expert actions at recorded state-machine phases.

This is an offline diagnostic.  It reads the existing HDF5 trajectories and
checkpoint, but never writes to either input.  Images are loaded in small
batches so the full RGB dataset is not copied into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source" / "atec_rl_lab"))

from demo.solution_act import Agent, Args  # noqa: E402


TARGET_PHASES = ("PRE_GRASP", "REACH", "CLOSE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return json.loads(value)
    return value


def cosine_similarity(pred: np.ndarray, expert: np.ndarray) -> float:
    denom = float(np.linalg.norm(pred) * np.linalg.norm(expert))
    return float(np.dot(pred, expert) / denom) if denom > 1e-8 else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else math.nan


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0}
    pred = np.asarray([row["pred_action"] for row in rows], dtype=np.float32)
    expert = np.asarray([row["expert_action"] for row in rows], dtype=np.float32)
    l1 = np.mean(np.abs(pred - expert), axis=1)
    max_abs = np.max(np.abs(pred - expert), axis=1)
    cos = np.asarray([cosine_similarity(p, e) for p, e in zip(pred, expert)])
    return {
        "samples": len(rows),
        "l1_mean": float(np.mean(l1)),
        "l1_median": float(np.median(l1)),
        "l1_p90": percentile(l1.tolist(), 90),
        "max_abs_mean": float(np.mean(max_abs)),
        "cosine_mean": float(np.mean(cos)),
        "pred_action_std": np.std(pred, axis=0).tolist(),
        "expert_action_std": np.std(expert, axis=0).tolist(),
        "pred_action_mean": np.mean(pred, axis=0).tolist(),
        "expert_action_mean": np.mean(expert, axis=0).tolist(),
        "prediction_error_max_abs_mean": float(np.mean([row["prediction_error_max_abs"] for row in rows])),
        "expert_temporal_delta_mean": float(np.mean([row["expert_temporal_delta"] for row in rows])),
        "pred_near_previous_expert_fraction": float(np.mean([row["pred_near_previous_expert"] for row in rows])),
        "expert_near_static_fraction": float(np.mean([row["expert_near_static"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/root/gpufree-data/atec_task_e_full_50/trajectory.hdf5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/gpufree-data/act_runs/phase2_sampling_motion_v1/checkpoints/best_loss.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/gpufree-data/eval/phase2_sampling_motion_v1/offline_diagnostics"))
    parser.add_argument("--samples-per-phase", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional trajectory cap for quick diagnostics on compressed RGB HDF5 files.",
    )
    args = parser.parse_args()
    if args.samples_per_phase < 1 or args.batch_size < 1:
        parser.error("--samples-per-phase and --batch-size must be positive")
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error("--max-trajectories must be positive when provided")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("Offline ACT diagnostic requires CUDA for the current Agent implementation")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    norm_stats = checkpoint["norm_stats"]
    state_dim = int(norm_stats["state_mean"].shape[-1])
    action_dim = int(norm_stats["action_mean"].shape[-1])
    checkpoint_meta = checkpoint.get("metadata", {})
    context_mode = checkpoint_meta.get(
        "context_mode", "legacy_qpos" if state_dim == 8 else "temporal_v1"
    )
    progress_horizon = int(checkpoint_meta.get("progress_horizon", 1))
    weight_key = "ema_agent" if "ema_agent" in checkpoint else "agent"
    model_args = Args()
    model_args.num_queries = int(checkpoint_meta.get("num_queries", model_args.num_queries))
    model_args.include_rgb = any("backbone" in key for key in checkpoint[weight_key].keys())
    agent = Agent(state_dim, action_dim, model_args).to(device)
    agent.load_state_dict(checkpoint[weight_key])
    agent.eval()
    state_mean = norm_stats["state_mean"].to(device)
    state_std = norm_stats["state_std"].to(device)
    action_mean = norm_stats["action_mean"].to(device)
    action_std = norm_stats["action_std"].to(device)

    rows: list[dict] = []
    with h5py.File(args.dataset, "r") as h5:
        traj_keys = sorted(h5.keys(), key=lambda key: int(key.split("_")[1]))
        if args.max_trajectories is not None:
            traj_keys = traj_keys[:args.max_trajectories]
        for traj_key in traj_keys:
            group = h5[traj_key]
            phases = decode_attr(group.attrs.get("phase_intervals", "[]")) or []
            for interval in phases:
                phase = str(interval.get("state", ""))
                if phase not in TARGET_PHASES:
                    continue
                start = max(0, int(interval["start_step"]))
                end = min(int(interval["end_step"]), int(group["actions"].shape[0]))
                if end <= start:
                    continue
                count = min(args.samples_per_phase, end - start)
                indices = np.unique(np.linspace(start, end - 1, count, dtype=np.int64)).tolist()
                for batch_start in range(0, len(indices), args.batch_size):
                    batch_indices = indices[batch_start:batch_start + args.batch_size]
                    obs_np = group["obs"][batch_indices].astype(np.float32)
                    expert_np = group["actions"][batch_indices].astype(np.float32)
                    rgb_np = group["images/rgb"][batch_indices]
                    if context_mode == "temporal_v1":
                        if "qvel" not in group:
                            raise ValueError(f"{traj_key} is missing qvel for temporal checkpoint")
                        qvel_np = group["qvel"][batch_indices].astype(np.float32)
                        previous_indices = [max(0, int(step) - 1) for step in batch_indices]
                        previous_action_np = group["actions"][previous_indices].astype(np.float32)
                        progress = np.asarray(
                            [
                                min(
                                    float(step) / float(max(progress_horizon - 1, 1)),
                                    1.0,
                                )
                                for step in batch_indices
                            ],
                            dtype=np.float32,
                        )[:, None]
                        # A trajectory's first action has no previous action context.
                        for row_idx, step in enumerate(batch_indices):
                            if int(step) == 0:
                                previous_action_np[row_idx] = 0.0
                        raw_state_np = np.concatenate(
                            (obs_np, qvel_np, previous_action_np, progress), axis=1
                        )
                    else:
                        raw_state_np = obs_np
                    if raw_state_np.shape[1] != state_dim:
                        raise ValueError(
                            f"Constructed state has dim {raw_state_np.shape[1]}, expected {state_dim}"
                        )
                    state = (torch.from_numpy(raw_state_np).to(device) - state_mean) / state_std
                    rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).to(device)
                    rgb = TF.resize(rgb, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                    model_obs = {"state": state, "rgb": rgb.unsqueeze(1)}
                    with torch.inference_mode():
                        pred_seq = agent.get_action(model_obs)
                        pred_np = (pred_seq[:, 0, :] * action_std + action_mean).detach().cpu().numpy()
                    for offset, step in enumerate(batch_indices):
                        expert_action = expert_np[offset]
                        pred_action = pred_np[offset]
                        previous_step = max(start, step - 1)
                        previous_expert = group["actions"][previous_step].astype(np.float32)
                        previous_state = group["obs"][previous_step].astype(np.float32)
                        prediction_error_max_abs = float(np.max(np.abs(pred_action - expert_action)))
                        expert_temporal_delta = float(np.max(np.abs(expert_action - previous_expert)))
                        rows.append({
                            "trajectory": traj_key,
                            "step": int(step),
                            "object": int(interval.get("object", -1)),
                            "phase": phase,
                            "pred_action": pred_action.astype(float).tolist(),
                            "expert_action": expert_action.astype(float).tolist(),
                            "prediction_error_max_abs": prediction_error_max_abs,
                            "expert_temporal_delta": expert_temporal_delta,
                            "pred_near_previous_expert": bool(np.max(np.abs(pred_action - previous_expert)) <= 0.002),
                            "expert_near_static": bool(np.max(np.abs(expert_action - previous_expert)) <= 0.002),
                            "state_delta_from_previous": float(np.max(np.abs(obs_np[offset] - previous_state))),
                        })

    by_phase: dict[str, list[dict]] = defaultdict(list)
    by_traj: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)
        by_traj[row["trajectory"]].append(row)
    summary = {
        "dataset": {"path": str(args.dataset.resolve()), "sha256": sha256(args.dataset)},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint), "weight_key": weight_key},
        "device": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "model": {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "include_rgb": model_args.include_rgb,
            "num_queries": model_args.num_queries,
            "context_mode": context_mode,
            "progress_horizon": progress_horizon,
        },
        "sampling": {
            "target_phases": list(TARGET_PHASES),
            "samples_per_phase": args.samples_per_phase,
            "batch_size": args.batch_size,
            "max_trajectories": args.max_trajectories,
        },
        "overall": summarize(rows),
        "by_phase": {phase: summarize(phase_rows) for phase, phase_rows in sorted(by_phase.items())},
        "by_trajectory": {traj: summarize(traj_rows) for traj, traj_rows in sorted(by_traj.items())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["trajectory", "step", "object", "phase", "prediction_error_max_abs", "expert_temporal_delta", "pred_near_previous_expert", "expert_near_static", "state_delta_from_previous"]
        for joint in range(action_dim):
            fieldnames.extend([f"pred_action_{joint + 1}", f"expert_action_{joint + 1}"])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fieldnames if key in row}
            for joint in range(action_dim):
                output[f"pred_action_{joint + 1}"] = row["pred_action"][joint]
                output[f"expert_action_{joint + 1}"] = row["expert_action"][joint]
            writer.writerow(output)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "overall": summary["overall"], "by_phase": summary["by_phase"]}, indent=2))


if __name__ == "__main__":
    main()
