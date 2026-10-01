"""Single-episode demo collection and success checking for Task E."""

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    TABLE_TOP_Z,
    BASKET_SUCCESS_CENTER, BASKET_SUCCESS_HALF_X, BASKET_SUCCESS_HALF_Y,
)

from .config import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN,
    GRASP_FINGER_CENTER_OBJECTS,
    LIFT_MAX_TARGET_XY_STEP, LIFT_MAX_LATERAL_DISPLACEMENT,
    STEPS,
    WARMUP_STEPS, SETTLE_STEPS,
)
from .state_machine import PickPlaceStateMachine


def _rerandomize_objects(env: ManagerBasedRLEnv, rng: np.random.Generator) -> dict[str, list[float]]:
    """Place each object randomly with AABB-based overlap rejection."""
    placed: dict[int, tuple[float, float]] = {}  # obj_idx -> (x, y)

    for obj_idx in [1, 2, 3]:
        obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        y_min, y_max = OBJ_SPAWN_Y_BANDS[obj_idx]
        hx, hy = OBJ_HALF_EXTENTS[obj_idx]

        x = y = None
        for _ in range(200):
            cx = float(rng.uniform(OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX))
            cy = float(rng.uniform(y_min, y_max))
            # AABB overlap check against all already-placed objects
            ok = all(
                abs(cx - px) >= hx + OBJ_HALF_EXTENTS[pi][0] + OBJ_BBOX_MARGIN or
                abs(cy - py) >= hy + OBJ_HALF_EXTENTS[pi][1] + OBJ_BBOX_MARGIN
                for pi, (px, py) in placed.items()
            )
            if ok:
                x, y = cx, cy
                break

        if x is None:  # fallback: band centre
            x = (OBJ_SPAWN_X_MIN + OBJ_SPAWN_X_MAX) / 2.0
            y = (y_min + y_max) / 2.0

        placed[obj_idx] = (x, y)
        state = obj.data.default_root_state[0:1].clone()
        state[0, 0] = x
        state[0, 1] = y
        state[0, 2] = OBJ_SPAWN_Z
        state[0, 7:] = 0.0   # zero velocities
        obj.write_root_state_to_sim(state)

    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    origin = env.unwrapped.scene.env_origins[0].detach().cpu()
    return {
        f"object_{obj_idx}": [float(x - y) for x, y in zip((x, y, OBJ_SPAWN_Z), origin.tolist())]
        for obj_idx, (x, y) in placed.items()
    }

_BASKET_MIN_Z = TABLE_TOP_Z
_BASKET_MAX_Z = TABLE_TOP_Z + 0.15
_LIFTED_MIN_Z = TABLE_TOP_Z + 0.01

def objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> dict[str, bool]:
    """Return per-object flags using the exact Task E termination bounds."""
    origin = env.unwrapped.scene.env_origins[0]
    result = {}
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0, :3] - origin
        result[f"object_{obj_idx}"] = bool(
            abs(pos[0].item() - BASKET_SUCCESS_CENTER[0]) <= BASKET_SUCCESS_HALF_X
            and abs(pos[1].item() - BASKET_SUCCESS_CENTER[1]) <= BASKET_SUCCESS_HALF_Y
            and _BASKET_MIN_Z <= pos[2].item() <= _BASKET_MAX_Z
        )
    return result


def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """Return True only if every picked object satisfies Task E's success bounds."""
    return all(objects_in_basket(env, pick_objects).values())


def _termination_terms(env: ManagerBasedRLEnv) -> dict[str, bool]:
    manager = env.unwrapped.termination_manager
    return {
        name: bool(manager.get_term(name)[0].item())
        for name in manager.active_terms
    }


def _quaternion_angle(q_a: torch.Tensor, q_b: torch.Tensor) -> float:
    """Return the shortest angular distance between two unit quaternions."""
    dot = torch.abs(torch.sum(q_a * q_b)).clamp(0.0, 1.0)
    return float((2.0 * torch.acos(dot)).item())


def collect_one_demo(
    env:         ManagerBasedRLEnv,
    robot,
    ik_ctrl:     CartesianController,
    arm_ids:     list[int],
    gripper_ids: list[int],
    pick_objects: list[int],
    device:      str,
    default_jpos: torch.Tensor,
    rng:         np.random.Generator,
    camera=None,
    steps: dict[str, int] | None = None,
    expert_profile: str = "stable",
) -> dict:
    """Run one full episode and return recorded data plus terminal metadata.

    Returns a dict with keys:
      qpos    (T, 8)        absolute joint positions
      qvel    (T, 8)        joint velocities
      ee_pos  (T, 3)        end-effector position (world frame)
      ee_quat (T, 4)        end-effector quaternion (w,x,y,z)
      action  (T, 8)        env action = (joint_target - default_jpos) / ACTION_SCALE
      frames  (T, H, W, 3)  RGB uint8 — only present when camera is given
      metadata  terminal status, object positions, and per-object completion flags
    """
    env.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )

    initial_positions = _rerandomize_objects(env, rng)  # write positions and forward sim
    default_jpos = robot.data.default_joint_pos.clone()
    origin = env.unwrapped.scene.env_origins[0]

    def _local_position(obj_idx: int) -> list[float]:
        pos = (
            env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0, :3]
            - origin
        )
        return [float(value) for value in pos.detach().cpu().tolist()]

    ee_home = torch.tensor([[RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z]],
                            dtype=torch.float32, device=device)
    eq_home = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=device)
    g_open  = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=device)

    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()

    finger_ids, finger_names = robot.find_bodies(["link7", "link8"], preserve_order=True)
    if finger_names != ["link7", "link8"]:
        raise RuntimeError(
            f"Expected Piper finger links ['link7', 'link8'], found {finger_names}"
        )
    gripper_limits = robot.data.joint_pos_limits[0, gripper_ids].detach().cpu()
    close_target = torch.tensor(
        [gripper_limits[0, 0].item(), gripper_limits[1, 1].item()],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(close_target < robot.data.joint_pos_limits[0, gripper_ids, 0]) or torch.any(
        close_target > robot.data.joint_pos_limits[0, gripper_ids, 1]
    ):
        raise RuntimeError(f"Derived gripper close target {close_target.tolist()} is outside hard limits")

    # Warm-up: drive arm to HOME position (not recorded)
    for _ in range(WARMUP_STEPS):
        if _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                    ee_home, eq_home, g_open, default_jpos):
            print("[WARN] Episode ended during warm-up; discarding attempt.")
            return None

    # Create the state machine before settling.  We retain the reset pose and
    # conditionally refresh it below if settling materially rolls an object.
    active_steps = dict(steps or STEPS)
    sm = PickPlaceStateMachine(pick_objects, device, active_steps)
    pre_settle_poses = {
        obj_idx: env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        .data.root_state_w[0].clone()
        for obj_idx in pick_objects
    }
    for obj_idx, state in pre_settle_poses.items():
        sm.set_grasp_quat(obj_idx, state[3:7])

    # Settle
    for _ in range(SETTLE_STEPS):
        if _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                    ee_home, eq_home, g_open, default_jpos):
            print("[WARN] Episode ended during settle; discarding attempt.")
            return None

    ik_ctrl.reset()

    # Use the post-settle pose only when the object actually rolled or moved
    # materially.  Stable assets keep the reset-pose grasp orientation that was
    # calibrated in the single-object regressions.
    for obj_idx in pick_objects:
        post_state = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                     .data.root_state_w[0].clone()
        pre_state = pre_settle_poses[obj_idx]
        position_delta = torch.linalg.norm(post_state[:3] - pre_state[:3]).item()
        orientation_delta = _quaternion_angle(post_state[3:7], pre_state[3:7])
        if position_delta > 0.02 or orientation_delta > 0.10:
            sm.set_grasp_quat(obj_idx, post_state[3:7])

    settled_positions = {
        f"object_{obj_idx}": _local_position(obj_idx) for obj_idx in pick_objects
    }

    ee_quat_home_w = ik_ctrl.ee_quat_w.clone()
    ee_pos_home_w = ik_ctrl.ee_pos_w.clone()
    finger_positions_home_w = robot.data.body_link_pos_w[:, finger_ids, :3]
    finger_center_offset_b = quat_apply_inverse(
        ee_quat_home_w,
        finger_positions_home_w.mean(dim=1) - ee_pos_home_w,
    )

    # ---- Recording loop ---- #
    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    frames_buf = [] if camera is not None else None

    terminal_terms: dict[str, bool] = {}
    terminal_reset_encountered = False
    last_positions = settled_positions
    ever_in_basket = {f"object_{obj_idx}": False for obj_idx in pick_objects}
    # A lift is relative to the object's settled spawn height.  Comparing to
    # table height would incorrectly mark tall objects as lifted at t=0.
    initial_z = {
        key: float(position[2]) for key, position in settled_positions.items()
    }
    ever_lifted = {f"object_{obj_idx}": False for obj_idx in pick_objects}
    lift_consecutive = {f"object_{obj_idx}": 0 for obj_idx in pick_objects}
    lift_loss_consecutive = {f"object_{obj_idx}": 0 for obj_idx in pick_objects}
    lift_lost_state: dict[str, str | None] = {f"object_{obj_idx}": None for obj_idx in pick_objects}
    close_push: dict[str, bool] = {f"object_{obj_idx}": False for obj_idx in pick_objects}
    close_start_pos: dict[str, list[float] | None] = {f"object_{obj_idx}": None for obj_idx in pick_objects}
    lift_anchor_xy_w: dict[str, torch.Tensor | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_target_xy_w: dict[str, torch.Tensor | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_start_ee_local: dict[str, list[float] | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_start_object_local: dict[str, list[float] | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_start_z_w: dict[str, float | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_hold_quat: dict[str, torch.Tensor | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_step_count: dict[str, int] = {f"object_{obj_idx}": 0 for obj_idx in pick_objects}
    lift_prev_target_w: dict[str, torch.Tensor | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_prev_actual_z: dict[str, float | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    lift_target_lateral_max: dict[str, float] = {f"object_{obj_idx}": 0.0 for obj_idx in pick_objects}
    lift_actual_lateral_max: dict[str, float] = {f"object_{obj_idx}": 0.0 for obj_idx in pick_objects}
    lift_object_lateral_max: dict[str, float] = {f"object_{obj_idx}": 0.0 for obj_idx in pick_objects}
    lift_z_monotonic: dict[str, bool] = {f"object_{obj_idx}": True for obj_idx in pick_objects}
    lift_actual_z_monotonic: dict[str, bool] = {f"object_{obj_idx}": True for obj_idx in pick_objects}
    lift_orientation_error_max: dict[str, float] = {f"object_{obj_idx}": 0.0 for obj_idx in pick_objects}
    place_prev_target_w: dict[str, torch.Tensor | None] = {
        f"object_{obj_idx}": None for obj_idx in pick_objects
    }
    place_target_lateral_max: dict[str, float] = {f"object_{obj_idx}": 0.0 for obj_idx in pick_objects}
    place_target_z_monotonic: dict[str, bool] = {f"object_{obj_idx}": True for obj_idx in pick_objects}
    step_trace: list[dict] = []
    phase_intervals: list[dict] = []
    active_phase: dict | None = None
    while not sm.done:
        step_idx = len(qpos_buf)
        state_name = sm.state
        object_idx = sm._obj_indices[sm._ptr]
        object_key = f"object_{object_idx}"
        # Keep explicit phase intervals so a failed episode can be located
        # without reconstructing the state machine from the raw actions.
        if active_phase is None:
            active_phase = {
                "object": object_idx,
                "state": state_name,
                "start_step": step_idx,
            }
        elif active_phase["object"] != object_idx or active_phase["state"] != state_name:
            active_phase["end_step"] = step_idx
            phase_intervals.append(active_phase)
            active_phase = {
                "object": object_idx,
                "state": state_name,
                "start_step": step_idx,
            }

        obj_pos_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_pos_w[0].clone()
        obj_quat_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_state_w[0, 3:7].clone()
        current_flags = objects_in_basket(env, pick_objects)
        for key, value in current_flags.items():
            ever_in_basket[key] |= value
        origin = env.unwrapped.scene.env_origins[0]
        last_positions = {
            f"object_{obj_idx}": (
                env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0, :3] - origin
            ).detach().cpu().tolist()
            for obj_idx in pick_objects
        }
        ee_pos_des, ee_quat_des, gripper_cmd = sm.tick(obj_pos_w)

        object_local = _local_position(object_idx)
        ee_local = (
            ik_ctrl.ee_pos_w[0].detach().cpu() - origin.detach().cpu()
        ).tolist()
        lifted = object_local[2] >= initial_z[object_key] + 0.05
        finger_positions_w = robot.data.body_link_pos_w[0, finger_ids, :3]
        finger_positions_local = (finger_positions_w - origin).detach().cpu().tolist()
        finger_center_local = [
            float(value) for value in finger_positions_w.mean(dim=0)
            .sub(origin).detach().cpu().tolist()
        ]
        finger_center_w = finger_positions_w.mean(dim=0)
        finger_gap = float(torch.linalg.norm(
            robot.data.body_link_pos_w[0, finger_ids[0], :3]
            - robot.data.body_link_pos_w[0, finger_ids[1], :3]
        ).item())
        joint_actual = robot.data.joint_pos[0, gripper_ids].detach().cpu().tolist()
        if state_name == "CLOSE" and close_start_pos[object_key] is None:
            close_start_pos[object_key] = object_local
        step_trace.append({
            "step": step_idx,
            "object": object_idx,
            "state": state_name,
            "object_local_m": object_local,
            "object_quat_w": [float(value) for value in obj_quat_w.detach().cpu().tolist()],
            "ee_local_m": [float(value) for value in ee_local],
            "ee_target_local_m": [
                float(value) for value in (ee_pos_des.detach().cpu() - origin.detach().cpu()).tolist()
            ],
            "ee_base_target_local_m": None,
            "finger_positions_local_m": finger_positions_local,
            "finger_center_local_m": finger_center_local,
            "finger_center_target_local_m": None,
            "finger_gap_m": finger_gap,
            "lift_target_local_m": None,
            "lift_target_delta_xy_m": None,
            "lift_target_delta_z_m": None,
            "lift_lateral_displacement_m": None,
            "lift_z_monotonic": None,
            "lift_orientation_error": None,
            "gripper": gripper_cmd,
            "gripper_joint_actual": [float(value) for value in joint_actual],
            "lifted": bool(lifted),
            "lift_consecutive_steps": lift_consecutive[object_key],
            "objects_in_basket": dict(current_flags),
        })

        # REACH/CLOSE targets are specified at the fingertip midpoint.  The
        # other state-machine waypoints retain their historical gripper-base
        # semantics (carry/place heights were calibrated in that frame).
        use_finger_center_target = object_idx in GRASP_FINGER_CENTER_OBJECTS
        if use_finger_center_target and state_name in ("REACH", "CLOSE"):
            grasp_feedback_gain = 1.5
            finger_target = ee_pos_des.clone()
            if state_name == "PRE_GRASP":
                # PRE_GRASP's waypoint is a safe-height approach point.  With
                # the object-aligned orientation already active, interpret it
                # at the fingertip center as well, otherwise rotating the wrist
                # changes the base-to-finger XY offset and causes a sideways
                # approach before the actual descent.
                finger_target[2] = CARRY_Z
            ee_pos_base_des = ik_ctrl.ee_pos_w.clone() + grasp_feedback_gain * (
                finger_target.unsqueeze(0) - finger_center_w.unsqueeze(0)
            )
            step_trace[-1]["finger_center_target_local_m"] = [
                float(value) for value in (finger_target.detach().cpu() - origin.detach().cpu()).tolist()
            ]
            step_trace[-1]["grasp_feedback_gain"] = grasp_feedback_gain
        elif state_name == "LIFT":
            # Lock the grasp pose at the start of LIFT.  The Z target is
            # interpolated over the whole phase; XY may correct only toward
            # the locked anchor and is limited to a small per-step movement.
            if lift_anchor_xy_w[object_key] is None:
                lift_anchor_xy_w[object_key] = ik_ctrl.ee_pos_w[:, :2].clone()
                lift_target_xy_w[object_key] = lift_anchor_xy_w[object_key].clone()
                lift_start_ee_local[object_key] = list(ee_local)
                lift_start_object_local[object_key] = list(object_local)
                lift_start_z_w[object_key] = float(ik_ctrl.ee_pos_w[0, 2].item())
                lift_hold_quat[object_key] = ee_quat_des.clone()
                lift_prev_target_w[object_key] = ik_ctrl.ee_pos_w[0].clone()

            anchor_xy = lift_anchor_xy_w[object_key]
            target_xy = lift_target_xy_w[object_key]
            assert anchor_xy is not None and target_xy is not None
            xy_error = anchor_xy[0] - ik_ctrl.ee_pos_w[:, :2]
            xy_norm = torch.linalg.norm(xy_error, dim=1, keepdim=True).clamp(min=1e-9)
            xy_step = xy_error * torch.clamp(
                LIFT_MAX_TARGET_XY_STEP / xy_norm, max=1.0
            )
            target_xy = target_xy + xy_step
            lift_target_xy_w[object_key] = target_xy
            lift_step_count[object_key] += 1
            alpha = min(lift_step_count[object_key] / max(active_steps["LIFT"], 1), 1.0)
            start_z = lift_start_z_w[object_key]
            assert start_z is not None
            lift_target_z = start_z + alpha * (CARRY_Z - start_z)

            ee_pos_base_des = ik_ctrl.ee_pos_w.clone()
            ee_pos_base_des[:, :2] = target_xy
            ee_pos_base_des[:, 2] = lift_target_z
            if lift_hold_quat[object_key] is not None:
                ee_quat_des = lift_hold_quat[object_key]
            step_trace[-1]["lift_axis"] = "world_z_primary_limited_xy"

            target_w = ee_pos_base_des[0].clone()
            previous_target = lift_prev_target_w[object_key]
            if previous_target is None:
                target_delta = torch.zeros(3, device=device)
            else:
                target_delta = target_w - previous_target
            lift_prev_target_w[object_key] = target_w
            step_trace[-1]["lift_target_local_m"] = [
                float(value) for value in (target_w.detach().cpu() - origin.detach().cpu()).tolist()
            ]
            step_trace[-1]["lift_target_delta_xy_m"] = float(
                torch.linalg.norm(target_delta[:2]).item()
            )
            step_trace[-1]["lift_target_delta_z_m"] = float(target_delta[2].item())
            lift_target_lateral_max[object_key] = max(
                lift_target_lateral_max[object_key],
                float(torch.linalg.norm(target_delta[:2]).item()),
            )
            step_trace[-1]["lift_z_monotonic"] = bool(target_delta[2].item() >= -1e-6)
            lift_z_monotonic[object_key] &= bool(target_delta[2].item() >= -1e-6)
            step_trace[-1]["lift_orientation_error"] = _quaternion_angle(
                ik_ctrl.ee_quat_w[0], ee_quat_des
            )
            lift_orientation_error_max[object_key] = max(
                lift_orientation_error_max[object_key], step_trace[-1]["lift_orientation_error"]
            )
        elif state_name == "PLACE":
            # PLACE is deliberately a one-dimensional descent.  The state
            # machine supplies a fixed basket XY and a monotonic Z waypoint;
            # keep that target explicit in the trace for dataset acceptance.
            ee_pos_base_des = ee_pos_des.unsqueeze(0)
            place_target = ee_pos_base_des[0].clone()
            previous_place_target = place_prev_target_w[object_key]
            if previous_place_target is not None:
                place_delta = place_target - previous_place_target
                place_target_lateral_max[object_key] = max(
                    place_target_lateral_max[object_key],
                    float(torch.linalg.norm(place_delta[:2]).item()),
                )
                place_target_z_monotonic[object_key] &= bool(place_delta[2].item() <= 1e-6)
                step_trace[-1]["place_target_delta_xy_m"] = float(
                    torch.linalg.norm(place_delta[:2]).item()
                )
                step_trace[-1]["place_target_delta_z_m"] = float(place_delta[2].item())
                step_trace[-1]["place_z_monotonic"] = bool(place_delta[2].item() <= 1e-6)
            else:
                step_trace[-1]["place_target_delta_xy_m"] = 0.0
                step_trace[-1]["place_target_delta_z_m"] = 0.0
                step_trace[-1]["place_z_monotonic"] = True
            place_prev_target_w[object_key] = place_target
        else:
            ee_pos_base_des = ee_pos_des.unsqueeze(0)
        step_trace[-1]["ee_base_target_local_m"] = [
            float(value) for value in (ee_pos_base_des[0].detach().cpu() - origin.detach().cpu()).tolist()
        ]
        ee_error = ee_pos_base_des[0] - ik_ctrl.ee_pos_w[0]
        finger_error = ee_pos_des - finger_center_w
        step_trace[-1]["ee_target_error_m"] = [float(value) for value in ee_error.detach().cpu().tolist()]
        step_trace[-1]["ee_target_error_norm_m"] = float(torch.linalg.norm(ee_error).item())
        step_trace[-1]["finger_center_error_m"] = [
            float(value) for value in finger_error.detach().cpu().tolist()
        ]
        step_trace[-1]["finger_center_error_norm_m"] = float(torch.linalg.norm(finger_error).item())
        arm_jpos_des   = ik_ctrl.compute(ee_pos_base_des, ee_quat_des.unsqueeze(0))
        gripper_vals   = GRIPPER_OPEN_POS if gripper_cmd == "open" else close_target.tolist()
        gripper_target = torch.tensor([gripper_vals], dtype=torch.float32, device=device)

        full_target = robot.data.joint_pos.clone()
        full_target[:, arm_ids]     = arm_jpos_des
        full_target[:, gripper_ids] = gripper_target
        step_trace[-1]["gripper_target"] = [float(value) for value in gripper_vals]
        step_trace[-1]["joint_target"] = full_target[0].detach().cpu().tolist()
        env_action = (full_target - default_jpos) / ACTION_SCALE

        # Record BEFORE stepping (obs at time t, action at time t)
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        ee_pos_buf.append(ik_ctrl.ee_pos_w[0].cpu().numpy())
        ee_quat_buf.append(ik_ctrl.ee_quat_w[0].cpu().numpy())
        action_buf.append(env_action[0].cpu().numpy())
        if frames_buf is not None:
            rgba = camera.data.output["rgb"][0].cpu().numpy()
            frames_buf.append(rgba[:, :, :3])

        _, _, terminated, truncated, _ = env.step(env_action)

        object_local_post = _local_position(object_idx)
        object_quat_post = env.unwrapped.scene.rigid_objects[object_key] \
                              .data.root_state_w[0, 3:7].detach().cpu().tolist()
        ee_local_post = (
            ik_ctrl.ee_pos_w[0].detach().cpu() - origin.detach().cpu()
        ).tolist()
        finger_positions_post = (
            robot.data.body_link_pos_w[0, finger_ids, :3] - origin
        ).detach().cpu().tolist()
        finger_center_post = [
            float(value) for value in robot.data.body_link_pos_w[0, finger_ids, :3]
            .mean(dim=0).sub(origin).detach().cpu().tolist()
        ]
        joint_actual_post = robot.data.joint_pos[0, gripper_ids].detach().cpu().tolist()
        lift_now = object_local_post[2] >= initial_z[object_key] + 0.05
        if state_name == "LIFT":
            start_ee = lift_start_ee_local[object_key]
            start_obj = lift_start_object_local[object_key]
            if start_ee is not None:
                lateral_displacement = float(np.linalg.norm(
                    np.asarray(ee_local_post[:2]) - np.asarray(start_ee[:2])
                ))
                lift_actual_lateral_max[object_key] = max(
                    lift_actual_lateral_max[object_key], lateral_displacement
                )
                step_trace[-1]["lift_lateral_displacement_m"] = lateral_displacement
            if start_obj is not None:
                object_lateral_displacement = float(np.linalg.norm(
                    np.asarray(object_local_post[:2]) - np.asarray(start_obj[:2])
                ))
                lift_object_lateral_max[object_key] = max(
                    lift_object_lateral_max[object_key], object_lateral_displacement
                )
            previous_actual_z = lift_prev_actual_z[object_key]
            actual_z_monotonic = previous_actual_z is None or object_local_post[2] >= previous_actual_z - 1e-4
            lift_actual_z_monotonic[object_key] &= actual_z_monotonic
            lift_prev_actual_z[object_key] = object_local_post[2]
            step_trace[-1]["lift_actual_z_monotonic"] = bool(actual_z_monotonic)
        if lift_now:
            lift_consecutive[object_key] += 1
            lift_loss_consecutive[object_key] = 0
        else:
            lift_consecutive[object_key] = 0
            if ever_lifted[object_key]:
                lift_loss_consecutive[object_key] += 1
                if lift_loss_consecutive[object_key] >= 5 and lift_lost_state[object_key] is None:
                    lift_lost_state[object_key] = state_name
        if lift_consecutive[object_key] >= 5:
            ever_lifted[object_key] = True
        object_displacement = float(np.linalg.norm(
            np.asarray(object_local_post) - np.asarray(object_local)
        ))
        if state_name == "CLOSE" and close_start_pos[object_key] is not None:
            close_displacement = float(np.linalg.norm(
                np.asarray(object_local_post) - np.asarray(close_start_pos[object_key])
            ))
            close_push[object_key] |= close_displacement >= 0.02
        else:
            close_displacement = 0.0
        step_trace[-1].update({
            "ee_local_post_m": [float(value) for value in ee_local_post],
            "finger_positions_post_local_m": finger_positions_post,
            "finger_center_post_local_m": finger_center_post,
            "finger_gap_post_m": float(torch.linalg.norm(
                robot.data.body_link_pos_w[0, finger_ids[0], :3]
                - robot.data.body_link_pos_w[0, finger_ids[1], :3]
            ).item()),
            "gripper_joint_actual_post": [float(value) for value in joint_actual_post],
            "gripper_joint_error_post": [
                float(target - actual)
                for target, actual in zip(gripper_vals, joint_actual_post)
            ],
            "object_local_post_m": object_local_post,
            "object_quat_post_w": [float(value) for value in object_quat_post],
            "object_displacement_m": object_displacement,
            "object_delta_xy_m": float(np.linalg.norm(
                np.asarray(object_local_post[:2]) - np.asarray(object_local[:2])
            )),
            "object_delta_z_m": float(object_local_post[2] - object_local[2]),
            "close_displacement_from_start_m": close_displacement,
            "lift_threshold_met": bool(lift_now),
            "lift_consecutive_steps_post": lift_consecutive[object_key],
            "ever_lifted_post": ever_lifted[object_key],
        })
        last_positions = {
            f"object_{idx}": _local_position(idx) for idx in pick_objects
        }

        if terminated.any() or truncated.any():
            terminal_terms = _termination_terms(env)
            terminal_reset_encountered = True
            if terminal_terms.get("basket_success", False):
                ever_in_basket = {f"object_{obj_idx}": True for obj_idx in pick_objects}
            break

        current_flags = objects_in_basket(env, pick_objects)
        for key, value in current_flags.items():
            ever_in_basket[key] |= value

    final_flags = objects_in_basket(env, pick_objects) if not terminal_reset_encountered else {
        key: bool(terminal_terms.get("basket_success", False)) for key in ever_in_basket
    }
    success = bool(terminal_terms.get("basket_success", False)) or (
        sm.done and all(final_flags.values())
    )
    if terminal_reset_encountered:
        termination_reason = [name for name, active in terminal_terms.items() if active]
    else:
        termination_reason = ["state_machine_complete"] if sm.done else ["collector_stopped"]
    if active_phase is not None:
        active_phase["end_step"] = len(step_trace)
        phase_intervals.append(active_phase)

    failure_stage = None
    failure_mode = None
    failure_object = None
    if not success:
        failure_object = next(
            (obj_idx for obj_idx in pick_objects
             if not final_flags.get(f"object_{obj_idx}", False)),
            None,
        )
        if failure_object is not None:
            object_key = f"object_{failure_object}"
            if not ever_lifted[object_key]:
                if close_push[object_key]:
                    failure_stage = "CLOSE"
                    failure_mode = "object_displaced_during_close"
                elif lift_object_lateral_max[object_key] > LIFT_MAX_LATERAL_DISPLACEMENT:
                    failure_stage = "LIFT"
                    failure_mode = "object_displaced_during_lift"
                else:
                    failure_stage = "LIFT"
                    failure_mode = "lift_not_confirmed"
            elif not ever_in_basket[object_key]:
                failure_stage = lift_lost_state[object_key] or "PLACE"
                if lift_lost_state[object_key] == "TRANSPORT":
                    failure_mode = "object_lost_during_transport"
                elif lift_lost_state[object_key] == "LIFT":
                    failure_mode = "object_lost_during_lift"
                else:
                    failure_mode = "released_not_in_basket"
            else:
                failure_stage = phase_intervals[-1]["state"] if phase_intervals else None
                failure_mode = "object_left_basket_region_after_entry"

    metadata = {
        "seed": int(getattr(env.unwrapped.cfg, "seed", -1)),
        "object_order": list(pick_objects),
        "initial_object_positions_local_m": initial_positions,
        "settled_object_positions_local_m": settled_positions,
        "final_object_positions_local_m": last_positions,
        "objects_in_basket": final_flags,
        "ever_in_basket": ever_in_basket,
        "ever_lifted": ever_lifted,
        "completed_object_count": int(sum(final_flags.values())),
        "ever_completed_object_count": int(sum(ever_in_basket.values())),
        "observation_type": "absolute_joint_position",
        "success": success,
        "termination_reason": termination_reason,
        "termination_step": len(step_trace),
        "last_phase": phase_intervals[-1]["state"] if phase_intervals else None,
        "failure_stage": failure_stage,
        "failure_mode": failure_mode,
        "failure_object": failure_object,
        "runtime_gripper_joint_limits": gripper_limits.tolist(),
        "runtime_gripper_close_target": close_target.detach().cpu().tolist(),
        "default_joint_pos": default_jpos[0].detach().cpu().tolist(),
        "action_scale": ACTION_SCALE,
        "finger_body_names": finger_names,
        "finger_center_offset_base_m": finger_center_offset_b[0].detach().cpu().tolist(),
        "lift_confirmation_steps": 5,
        "lift_diagnostics": {
            key: {
                "target_lateral_max_m": lift_target_lateral_max[key],
                "actual_lateral_max_m": lift_actual_lateral_max[key],
                "object_lateral_max_m": lift_object_lateral_max[key],
                "target_z_monotonic": lift_z_monotonic[key],
                "actual_z_monotonic": lift_actual_z_monotonic[key],
                "orientation_error_max_rad": lift_orientation_error_max[key],
            }
            for key in ever_lifted
        },
        "place_diagnostics": {
            key: {
                "target_lateral_max_m": place_target_lateral_max[key],
                "target_z_monotonic": place_target_z_monotonic[key],
            }
            for key in ever_lifted
        },
        "phase_intervals": phase_intervals,
        "step_trace": step_trace,
        "expert_profile": expert_profile,
        "state_steps": active_steps,
        "terminal_reset_encountered": terminal_reset_encountered,
        "cross_reset": False,
    }

    result = {
        "qpos":    np.stack(qpos_buf),
        "qvel":    np.stack(qvel_buf),
        "ee_pos":  np.stack(ee_pos_buf),
        "ee_quat": np.stack(ee_quat_buf),
        "action":  np.stack(action_buf),
        "metadata": metadata,
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)
    return result


# ------------------------------------------------------------------ #
# Internal helper
# ------------------------------------------------------------------ #

def _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
             ee_pos, ee_quat, gripper_target, default_jpos):
    """Single IK step toward a target pose (utility used during warm-up/settle)."""
    arm_des = ik_ctrl.compute(ee_pos, ee_quat)
    tgt = robot.data.joint_pos.clone()
    tgt[:, arm_ids]     = arm_des
    tgt[:, gripper_ids] = gripper_target
    _, _, terminated, truncated, _ = env.step((tgt - default_jpos) / ACTION_SCALE)
    robot.update(dt=env.unwrapped.physics_dt)
    return bool(terminated.any() or truncated.any())
