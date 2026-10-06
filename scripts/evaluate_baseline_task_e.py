"""Run one reproducible Task E baseline episode and save its evidence."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max-steps", type=int, default=1500)
parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "baseline_act" / "seed_1000")
parser.add_argument("--video-stride", type=int, default=5)
parser.add_argument(
    "--temporal-agg",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override ACT temporal aggregation for this evaluation only.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_steps < 1 or args.video_stride < 1:
    parser.error("--max-steps and --video-stride must be positive")
args.enable_cameras = True
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import atec_rl_lab.tasks  # noqa: E402,F401
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_rl_lab.tasks.task_e.env_cfg import (  # noqa: E402
    BASKET_SUCCESS_CENTER,
    BASKET_SUCCESS_HALF_X,
    BASKET_SUCCESS_HALF_Y,
    TABLE_TOP_Z,
    TaskEEnvPiperCfg,
)
from demo.solution import AlgSolution  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def local_object_positions(env) -> dict[str, list[float]]:
    origin = env.unwrapped.scene.env_origins[0]
    return {
        name: (env.unwrapped.scene[name].data.root_pos_w[0, :3] - origin).detach().cpu().tolist()
        for name in ("object_1", "object_2", "object_3")
    }


def robot_diagnostics(env) -> dict:
    """Return Isaac's measured end-effector and joint-limit state.

    The RGB controller has its own FK model, so this records the simulator
    frame separately instead of treating the controller's FK as ground truth.
    """
    robot = env.unwrapped.scene["robot"]
    body_names = list(robot.body_names)
    ee_idx = body_names.index("gripper_base")
    origin = env.unwrapped.scene.env_origins[0]
    ee_pos = robot.data.body_pos_w[0, ee_idx, :3] - origin
    ee_quat = robot.data.body_quat_w[0, ee_idx]
    finger_indices = [
        idx for name, idx in zip(body_names, range(len(body_names)))
        if name in {"link7", "link8"}
    ]
    finger_positions = robot.data.body_pos_w[0, finger_indices, :3] - origin if finger_indices else None
    q = robot.data.joint_pos[0]
    qvel = robot.data.joint_vel[0]
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is not None:
        limits = limits[0] if limits.ndim == 3 else limits
        lower_values = limits[:, 0]
        upper_values = limits[:, 1]
    else:
        lower_values = upper_values = None
    return {
        "body_names": body_names,
        "ee_body_name": "gripper_base",
        "ee_position_local_m": ee_pos.detach().cpu().tolist(),
        "ee_quaternion_wxyz": ee_quat.detach().cpu().tolist(),
        "finger_body_names": [body_names[idx] for idx in finger_indices],
        "finger_positions_local_m": None if finger_positions is None else finger_positions.detach().cpu().tolist(),
        "finger_center_local_m": None if finger_positions is None else finger_positions.mean(dim=0).detach().cpu().tolist(),
        "joint_position_absolute": q.detach().cpu().tolist(),
        "joint_velocity": qvel.detach().cpu().tolist(),
        "joint_position_lower": None if lower_values is None else lower_values.detach().cpu().tolist(),
        "joint_position_upper": None if upper_values is None else upper_values.detach().cpu().tolist(),
    }


def inside_basket(position: list[float]) -> bool:
    return (
        abs(position[0] - BASKET_SUCCESS_CENTER[0]) <= BASKET_SUCCESS_HALF_X
        and abs(position[1] - BASKET_SUCCESS_CENTER[1]) <= BASKET_SUCCESS_HALF_Y
        and TABLE_TOP_Z <= position[2] <= TABLE_TOP_Z + 0.15
    )


def frame_from_obs(obs) -> np.ndarray:
    frame = obs["image"]["video_rgb"][0].detach().cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame


def run() -> None:
    output_dir = args.output_dir.resolve()
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_logs").mkdir(exist_ok=True)

    policy_path = Path(os.environ.get("ATEC_TASK_E_POLICY") or ROOT / "atec_robot_model/baseline/act/policy.pt").resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

    cfg = TaskEEnvPiperCfg(seed=args.seed)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    solution = AlgSolution()
    policy_mode = os.environ.get("ATEC_TASK_E_POLICY_MODE", "act").strip().lower()
    if args.temporal_agg is not None:
        solution.temporal_agg = bool(args.temporal_agg)
        solution.reset()
    cfg = apply_safe_action_spec(cfg, solution.get_action_spec())
    step_dt = cfg.sim.dt * cfg.decimation
    manifest = {
        "task": "ATEC-TaskE-Piper",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "policy": {
            "path": str(policy_path),
            "sha256": sha256(policy_path),
            "kind": "original_baseline" if policy_path == (ROOT / "atec_robot_model/baseline/act/policy.pt").resolve() else "trained_checkpoint",
        },
        "evaluation_code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "solution_code": {
            "path": str(ROOT / ("demo/solution_rgb.py" if policy_mode == "rgb" else "demo/solution_act.py")),
            "sha256": sha256(ROOT / ("demo/solution_rgb.py" if policy_mode == "rgb" else "demo/solution_act.py")),
            "mode": policy_mode,
        },
        "environment_code": {
            "path": str(ROOT / "source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py"),
            "sha256": sha256(ROOT / "source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py"),
        },
        "seed": args.seed,
        "num_envs": 1,
        "sim_dt_s": cfg.sim.dt,
        "decimation": cfg.decimation,
        "control_dt_s": step_dt,
        "environment_timeout_s": cfg.episode_length_s,
        "evaluation_max_steps": args.max_steps,
        "basket_center": BASKET_SUCCESS_CENTER,
        "basket_half_x_m": BASKET_SUCCESS_HALF_X,
        "basket_half_y_m": BASKET_SUCCESS_HALF_Y,
        "table_top_z_m": TABLE_TOP_Z,
        "video_stride": args.video_stride,
        "temporal_agg": solution.temporal_agg,
        "video_fps": round(1 / (step_dt * args.video_stride)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    if policy_mode == "rgb":
        calibration_path = Path(os.environ.get(
            "ATEC_TASK_E_RGB_CALIBRATION",
            "/root/gpufree-data/rgb_runs/task_e_calibration.json",
        )).resolve()
        templates_path = Path(os.environ.get(
            "ATEC_TASK_E_RGB_TEMPLATES",
            "/root/gpufree-data/rgb_runs/expert_templates.json",
        )).resolve()
        manifest["rgb_assets"] = {
            "calibration": {
                "path": str(calibration_path),
                "sha256": sha256(calibration_path) if calibration_path.is_file() else None,
            },
            "expert_templates": {
                "path": str(templates_path),
                "sha256": sha256(templates_path) if templates_path.is_file() else None,
            },
        }
    save_json(output_dir / "manifest.json", manifest)

    env = None
    writer = None
    start_wall = time.monotonic()
    steps = 0
    score = 0.0
    end_reason = "evaluation_cap"
    termination_terms = {}
    final_positions = {}
    max_in_basket = 0
    first_in_basket_step = None
    first_stall_step = None
    policy_start_step = None
    stationary_steps = 0
    previous_qpos = None
    previous_in_basket = None
    trace = None
    try:
        env = gym.make("ATEC-TaskE-Piper", cfg=cfg)
        obs, _ = env.reset(seed=args.seed)
        solution.reset()
        initial_positions = local_object_positions(env)
        manifest["initial_object_positions_local_m"] = initial_positions
        save_json(output_dir / "manifest.json", manifest)
        video_path = output_dir / "videos" / f"seed_{args.seed}.mp4"
        writer = imageio.get_writer(str(video_path), fps=manifest["video_fps"], codec="libx264", quality=6)
        writer.append_data(frame_from_obs(obs))
        print(f"[BASELINE] seed={args.seed} initial_positions={initial_positions}", flush=True)
        trace = (output_dir / "steps.jsonl").open("w", encoding="utf-8")

        while steps < args.max_steps and app.is_running():
            with torch.inference_mode():
                response = solution.predicts(obs, score)
                if response["giveup"]:
                    end_reason = "giveup"
                    break
                action = torch.as_tensor(response["action"], dtype=torch.float32, device=env.unwrapped.device)
                if action.shape != (1, 8) or not bool(torch.isfinite(action).all()):
                    end_reason = "invalid_action"
                    trace.write(json.dumps({"step": steps, "reason": end_reason, "action": response["action"]}) + "\n")
                    break
                control_phase = "home" if not solution._home_done or solution._ts == 0 else "act"
                if control_phase == "act" and policy_start_step is None:
                    policy_start_step = steps + 1
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                score += float(reward[0].item()) / step_dt
                termination_terms = {
                    name: bool(env.unwrapped.termination_manager.get_term(name)[0].item())
                    for name in env.unwrapped.termination_manager.active_terms
                }
                done = bool(terminated[0].item() or truncated[0].item())
                if done:
                    end_reason = "|".join(name for name, active in termination_terms.items() if active) or "unknown_termination"
                else:
                    final_positions = local_object_positions(env)
                    in_basket = {name: inside_basket(pos) for name, pos in final_positions.items()}
                    basket_count = sum(in_basket.values())
                    max_in_basket = max(max_in_basket, basket_count)
                    if basket_count and first_in_basket_step is None:
                        first_in_basket_step = steps
                    qpos = env.unwrapped.scene["robot"].data.joint_pos[0].detach().cpu().tolist()
                    qvel = env.unwrapped.scene["robot"].data.joint_vel[0].detach().cpu().tolist()
                    if control_phase == "act" and previous_qpos is not None:
                        moved = max(abs(a - b) for a, b in zip(qpos, previous_qpos)) > 0.002
                        stationary_steps = 0 if moved or in_basket != previous_in_basket else stationary_steps + 1
                        if stationary_steps >= 150 and first_stall_step is None:
                            first_stall_step = steps - 149
                    else:
                        stationary_steps = 0
                    previous_qpos = qpos
                    previous_in_basket = in_basket
                trace.write(json.dumps({
                    "step": steps,
                    "simulation_time_s": round(steps * step_dt, 4),
                    "control_phase": control_phase,
                    "action": action[0].detach().cpu().tolist(),
                    "joint_position": None if done else qpos,
                    "joint_velocity": None if done else qvel,
                    "robot_diagnostics": None if done else robot_diagnostics(env),
                    "object_positions_local_m": None if done else final_positions,
                    "objects_in_basket": None if done else in_basket,
                    "basket_count": None if done else basket_count,
                    "stationary_steps": stationary_steps,
                    "termination_reason": end_reason if done else None,
                    "controller_diagnostics": getattr(solution, "last_diagnostics", None),
                }, separators=(",", ":")) + "\n")
                if not done and steps % args.video_stride == 0:
                    writer.append_data(frame_from_obs(obs))
                if steps % 100 == 0 or done:
                    print(f"[BASELINE] step={steps} sim_s={steps * step_dt:.2f} reason={end_reason if done else 'running'}", flush=True)
                if done:
                    break
        if not app.is_running() and end_reason == "evaluation_cap" and steps < args.max_steps:
            end_reason = "app_stopped"
        result = {
            "seed": args.seed,
            "success": bool(termination_terms.get("basket_success", False)),
            "completed": end_reason not in ("evaluation_cap", "app_stopped"),
            "termination_reason": end_reason,
            "termination_terms": termination_terms,
            "steps": steps,
            "simulation_time_s": round(steps * step_dt, 4),
            "wall_time_s": round(time.monotonic() - start_wall, 2),
            "score": round(score, 4),
            "max_simultaneous_objects_in_basket": max_in_basket,
            "policy_start_step": policy_start_step,
            "first_in_basket_step": first_in_basket_step,
            "first_stall_step": first_stall_step,
            "stall_definition": "150 consecutive ACT steps with max joint delta <= 0.002 rad and unchanged basket state",
            "last_observed_object_positions_local_m": final_positions,
            "video": str(video_path),
            "step_trace": str(output_dir / "steps.jsonl"),
        }
        with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        save_json(output_dir / "summary.json", {
            "episodes": 1,
            "completed_episodes": int(result["completed"]),
            "successful_episodes": int(result["success"]),
            "success_rate_all_runs": float(result["success"]),
            "success_rate_completed_episodes": float(result["success"]) if result["completed"] else None,
            "average_steps": steps,
            "average_simulation_time_s": result["simulation_time_s"],
            "termination_reasons": {end_reason: 1},
        })
        print(f"[BASELINE] result={json.dumps(result, ensure_ascii=False)}", flush=True)
    finally:
        if trace is not None:
            trace.close()
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    run()
