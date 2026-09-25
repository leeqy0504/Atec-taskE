"""Single-episode demo collection and success checking for Task E."""

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    TABLE_TOP_Z,
    BASKET_SUCCESS_CENTER, BASKET_SUCCESS_HALF_X, BASKET_SUCCESS_HALF_Y,
)

from .config import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN,
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

    ee_home = torch.tensor([[RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z]],
                            dtype=torch.float32, device=device)
    eq_home = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=device)
    g_open  = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=device)

    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()

    # Warm-up: drive arm to HOME position (not recorded)
    for _ in range(WARMUP_STEPS):
        if _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                    ee_home, eq_home, g_open, default_jpos):
            print("[WARN] Episode ended during warm-up; discarding attempt.")
            return None

    # Pre-compute grasp quaternions from actual object orientations after reset
    sm = PickPlaceStateMachine(pick_objects, device)
    for obj_idx in pick_objects:
        obj_quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                       .data.root_state_w[0, 3:7]
        sm.set_grasp_quat(obj_idx, obj_quat)

    # Settle
    for _ in range(SETTLE_STEPS):
        if _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
                    ee_home, eq_home, g_open, default_jpos):
            print("[WARN] Episode ended during settle; discarding attempt.")
            return None

    ik_ctrl.reset()

    # ---- Recording loop ---- #
    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    frames_buf = [] if camera is not None else None

    terminal_terms: dict[str, bool] = {}
    terminal_reset_encountered = False
    last_positions = initial_positions
    ever_in_basket = {f"object_{obj_idx}": False for obj_idx in pick_objects}
    # A lift is relative to the object's settled spawn height.  Comparing to
    # table height would incorrectly mark tall objects as lifted at t=0.
    initial_z = {
        key: float(position[2]) for key, position in initial_positions.items()
    }
    ever_lifted = {f"object_{obj_idx}": False for obj_idx in pick_objects}
    step_trace: list[dict] = []
    phase_intervals: list[dict] = []
    active_phase: dict | None = None

    def _local_position(obj_idx: int) -> list[float]:
        origin = env.unwrapped.scene.env_origins[0]
        pos = (
            env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0, :3]
            - origin
        )
        return [float(value) for value in pos.detach().cpu().tolist()]

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
        ever_lifted[object_key] |= lifted
        step_trace.append({
            "step": step_idx,
            "object": object_idx,
            "state": state_name,
            "object_local_m": object_local,
            "ee_local_m": [float(value) for value in ee_local],
            "ee_target_local_m": [
                float(value) for value in (ee_pos_des.detach().cpu() - origin.detach().cpu()).tolist()
            ],
            "gripper": gripper_cmd,
            "lifted": bool(lifted),
            "objects_in_basket": dict(current_flags),
        })

        arm_jpos_des   = ik_ctrl.compute(ee_pos_des.unsqueeze(0), ee_quat_des.unsqueeze(0))
        gripper_vals   = GRIPPER_OPEN_POS if gripper_cmd == "open" else GRIPPER_CLOSE_POS
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
                failure_stage = "LIFT"
            elif not ever_in_basket[object_key]:
                failure_stage = "PLACE"
            else:
                failure_stage = phase_intervals[-1]["state"] if phase_intervals else None

    metadata = {
        "seed": int(getattr(env.unwrapped.cfg, "seed", -1)),
        "object_order": list(pick_objects),
        "initial_object_positions_local_m": initial_positions,
        "final_object_positions_local_m": last_positions,
        "objects_in_basket": final_flags,
        "ever_in_basket": ever_in_basket,
        "ever_lifted": ever_lifted,
        "completed_object_count": int(sum(final_flags.values())),
        "ever_completed_object_count": int(sum(ever_in_basket.values())),
        "success": success,
        "termination_reason": termination_reason,
        "termination_step": len(step_trace),
        "last_phase": phase_intervals[-1]["state"] if phase_intervals else None,
        "failure_stage": failure_stage,
        "failure_object": failure_object,
        "phase_intervals": phase_intervals,
        "step_trace": step_trace,
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
