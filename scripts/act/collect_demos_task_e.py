"""Scripted oracle data collection for Task E (pick-and-place).

Usage
-----
# Single-object smoke test:
python scripts/act/collect_demos_task_e.py --pick_objects 3 --num_demos 5 --seed 17 --headless
# Full Task E demonstrations:
python scripts/act/collect_demos_task_e.py --pick_objects 1 2 3 --num_demos 50 --seed 123 --headless --save_images --only_success

"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

# sys.path.insert(0, os.path.dirname(__file__))

# Isaac Lab's process does not always resolve PyTorch's split CUDA linalg
# library when DifferentialIK first calls torch.inverse(). Load it explicitly
# before Isaac Sim starts, while the absolute library path is still available.
_torch_linalg = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "torch"
    / "lib"
    / "libtorch_cuda_linalg.so"
)
if _torch_linalg.is_file():
    ctypes.CDLL(str(_torch_linalg), mode=ctypes.RTLD_GLOBAL)

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Collect Task E demonstrations for ACT.")
add_collect_demo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# TaskEEnvPiperCfg always declares the external and end-effector camera
# sensors. Isaac Sim must enable camera extensions before constructing that
# environment, even when this run does not save RGB frames.
args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
from atec_rl_lab.utils import CartesianController

from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT,
    CAM_H, CAM_W, CAM_POS, CAM_ROT,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS, GRASP_Z_OFFSETS, GRASP_LONG_AXIS,
    GRASP_FINGER_CENTER_OBJECTS,
    LIFT_MAX_TARGET_XY_STEP, LIFT_MAX_LATERAL_DISPLACEMENT,
    CARRY_Z, PLACE_HEIGHT, STEPS, STEP_PROFILES, STATE_ORDER,
)
from task_e.collector import collect_one_demo


def build_env(pick_objects: list[int], need_camera: bool, seed: int) -> ManagerBasedRLEnv:
    cfg = TaskEEnvPiperCfg()
    cfg.seed              = int(seed)
    cfg.scene.num_envs    = 1
    cfg.episode_length_s  = 40.0 * len(pick_objects) + 10.0
    cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=ACT_EFFORT_LIMIT,
        velocity_limit=ACT_VEL_LIMIT,
        stiffness=ACT_STIFFNESS,
        damping=ACT_DAMPING,
    )
    # if need_camera:
    #     cfg.scene.video_cam = CameraCfg(
    #         prim_path="{ENV_REGEX_NS}/video_cam",
    #         update_period=0.0,
    #         height=CAM_H, width=CAM_W,
    #         data_types=["rgb"],
    #         spawn=sim_utils.PinholeCameraCfg(
    #             focal_length=24.0, focus_distance=400.0,
    #             horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
    #         ),
    #         offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
    #     )
    return ManagerBasedRLEnv(cfg)


def init_output(output_dir: str, metadata: dict) -> tuple[str, str]:
    """Create output directory, wipe any existing trajectory.hdf5, write JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    with h5py.File(traj_path, "w"):   # truncate / create fresh
        pass
    with open(json_path, "w") as fh:
        json.dump({
            "env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}},
            "collection": metadata,
        }, fh, indent=2)
    return traj_path, json_path


def save_traj(traj_path: str, traj_idx: int, data: dict,
              save_images: bool) -> None:
    """Append one trajectory group to the consolidated HDF5."""
    temporal_keys = ("qpos", "qvel", "ee_pos", "ee_quat", "action")
    lengths = {key: len(data[key]) for key in temporal_keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Temporal trajectory lengths disagree: {lengths}")
    if save_images and "frames" in data and len(data["frames"]) != lengths["qpos"]:
        raise ValueError(
            f"RGB frame count {len(data['frames'])} does not match trajectory length {lengths['qpos']}"
        )
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",     data=data["qpos"],    compression="gzip")
        grp.create_dataset("actions", data=data["action"],  compression="gzip")
        grp.create_dataset("qvel",    data=data["qvel"],    compression="gzip")
        grp.create_dataset("ee_pos",  data=data["ee_pos"],  compression="gzip")
        grp.create_dataset("ee_quat", data=data["ee_quat"], compression="gzip")
        if save_images and "frames" in data:
            grp.create_group("images").create_dataset(
                "rgb", data=data["frames"], compression="gzip"
            )
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            if value is None:
                # HDF5 attributes have no native None/object dtype.  Keep the
                # distinction between an unset optional diagnostic and an
                # empty string in a JSON-compatible representation.
                value = "null"
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, separators=(",", ":"))
            grp.attrs[key] = value


def main() -> None:
    pick_objects = sorted(set(args_cli.pick_objects))
    active_steps = dict(STEP_PROFILES[args_cli.expert_profile])
    need_camera = args_cli.save_video or args_cli.save_images

    if any(obj not in (1, 2, 3) for obj in pick_objects):
        raise ValueError("--pick_objects values must be drawn from {1, 2, 3}")
    if not pick_objects:
        raise ValueError("--pick_objects must contain at least one object")

    np.random.seed(args_cli.seed)
    env    = build_env(pick_objects, need_camera, args_cli.seed)
    dev    = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids,     _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    collection_metadata = {
        "seed": int(args_cli.seed),
        "object_order": pick_objects,
        "num_requested": int(args_cli.num_demos),
        "save_images": bool(args_cli.save_images),
        "success_filter": bool(args_cli.only_success),
        "expert_parameters": {
            "gripper_open_pos": GRIPPER_OPEN_POS,
            "gripper_close_pos_configured": GRIPPER_CLOSE_POS,
            "grasp_z_offsets_finger_center_m": GRASP_Z_OFFSETS,
            "grasp_long_axis": GRASP_LONG_AXIS,
            "grasp_finger_center_objects": sorted(GRASP_FINGER_CENTER_OBJECTS),
            "lift_max_target_xy_step_m": LIFT_MAX_TARGET_XY_STEP,
            "lift_max_lateral_displacement_m": LIFT_MAX_LATERAL_DISPLACEMENT,
            "carry_z": CARRY_Z,
            "place_height": PLACE_HEIGHT,
            "place_mode": "fixed_basket_xy_monotonic_z_descent",
            "state_order": STATE_ORDER,
            "state_steps": active_steps,
            "expert_profile": args_cli.expert_profile,
            "ik_lambda": 0.05,
            "ik_max_joint_delta": 0.2,
        },
    }
    traj_path, _ = init_output(args_cli.output_dir, collection_metadata)
    rng = np.random.default_rng(args_cli.seed)

    n_ok = 0
    attempt = 0
    max_attempts = args_cli.max_attempts or max(args_cli.num_demos * 20, args_cli.num_demos)
    while n_ok < args_cli.num_demos and attempt < max_attempts:
        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")

        data = collect_one_demo(
            env, robot, ik_ctrl,
            arm_ids, gripper_ids,
            pick_objects, dev,
            default_jpos=default_jpos,
            rng=rng,
            camera=camera,
            steps=active_steps,
            expert_profile=args_cli.expert_profile,
        )
        if data is None:
            print("[WARN] Collector returned no data — skipping.")
            continue

        success = bool(data.get("metadata", {}).get("success", False))
        if args_cli.only_success and not success:
            print("[WARN] Demo did not satisfy the Task E success condition — skipping (--only_success).")
            continue

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T     = len(data["qpos"])
        notes = [f"{T} steps"]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video → {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}")
        n_ok += 1

    if n_ok < args_cli.num_demos:
        raise RuntimeError(
            f"Collected {n_ok}/{args_cli.num_demos} demos after {attempt} attempts "
            f"(max_attempts={max_attempts})."
        )
    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
