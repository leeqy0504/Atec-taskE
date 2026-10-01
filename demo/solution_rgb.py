"""Traditional RGB Task-E controller.

Only the public observation image and proprioception are consumed here.  The
object state is estimated by :class:`RGBPerception`; the deterministic
pick/place sequencing and bounded Piper IK replace the learned ACT policy.
"""

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import torch

try:
    from scripts.rgb.perception import RGBPerception, Detection
    from scripts.rgb.piper_kinematics import PiperKinematics
except ImportError:  # allows running with scripts/rgb directly on PYTHONPATH
    from rgb.perception import RGBPerception, Detection
    from rgb.piper_kinematics import PiperKinematics


DEFAULT_JOINT_POS = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=np.float64)
HOME_JOINT_POS = np.array([-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035], dtype=np.float64)
OPEN_GRIPPER = np.array([0.035, -0.035], dtype=np.float64)
CLOSE_GRIPPER = np.array([0.0, 0.0], dtype=np.float64)

# These values are fixed task geometry, not observations of object state.
TABLE_TOP_Z = 0.8266426476
OBJECT_Z = TABLE_TOP_Z + 0.030000
BASE_WORLD = np.array([1.0 + 0.6468062441 * 1.25 * 0.5, 0.0, TABLE_TOP_Z], dtype=np.float64)
BASKET_WORLD_XY = np.array([1.08, -0.30], dtype=np.float64)
CARRY_Z = TABLE_TOP_Z + 0.40
PLACE_Z = TABLE_TOP_Z + 0.15
GRASP_OFFSETS = {"object_1": 0.0, "object_2": 0.0, "object_3": 0.09}
OBJECT_Y_BANDS = {"object_1": (0.25, 0.29), "object_2": (0.14, 0.20), "object_3": (0.03, 0.09)}
STEPS = {
    "INIT": 60, "PRE_GRASP": 160, "REACH": 100, "CLOSE": 40,
    "LIFT": 160, "TRANSPORT": 200, "PLACE": 80, "OPEN": 60,
    "LIFT_RETRACT": 80, "RETRACT": 80,
}
STATES = tuple(STEPS)
ROOT_R = np.diag([-1.0, -1.0, 1.0])  # Piper base is rotated pi around world Z
GRIP_R_WORLD = np.diag([1.0, -1.0, -1.0])


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class AlgSolution:
    """Drop-in ``predicts`` adapter with the same 8-D environment action."""

    def __init__(self):
        calibration = Path(os.environ.get(
            "ATEC_TASK_E_RGB_CALIBRATION",
            "/root/gpufree-data/rgb_runs/task_e_calibration.json",
        ))
        self.perception = RGBPerception(calibration, hold_frames=10, smoothing=0.35)
        # The environment's 20 ms position actuator lags the target.  Allow
        # a bounded but larger per-step IK increment so REACH can close the
        # 0.35 m vertical gap within its fixed 100-step budget.
        self.kinematics = PiperKinematics(max_delta=0.12)
        self.default_joint_pos = DEFAULT_JOINT_POS.copy()
        self.teleop_home_joint_pos = HOME_JOINT_POS.copy()
        self.temporal_agg = False
        self.reset()

    def get_action_spec(self):
        return {}

    def reset(self):
        self._ts = 0
        self._home_done = False
        self._home_stable_steps = 0
        self._state = "INIT"
        self._state_step = 0
        self._object_ptr = 0
        self._object_order = ("object_1", "object_2", "object_3")
        self._target_xy = None
        self._grasp_xy = None
        self._last_qpos = None
        self._joint_goal_key = None
        self._joint_goal = None
        self.last_diagnostics = {
            "state": "HOME",
            "object": None,
            "detection": {},
            "ik_target_world": None,
            "lost_detection": False,
        }

    def _proprio(self, obs) -> np.ndarray:
        if not isinstance(obs, dict) or "proprio" not in obs:
            raise ValueError("RGB controller expects obs['proprio']")
        p = _as_numpy(obs["proprio"]).astype(np.float64, copy=False)
        if p.ndim == 1:
            p = p[None, :]
        if p.shape[1] < 8:
            raise ValueError(f"Expected at least 8 proprio values, got {p.shape}")
        return p

    @staticmethod
    def _image(obs) -> np.ndarray:
        try:
            image = _as_numpy(obs["image"]["video_rgb"])
        except (KeyError, TypeError) as exc:
            raise ValueError("RGB controller expects obs['image']['video_rgb']") from exc
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected HxWxC RGB frame, got {image.shape}")
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        return image

    def _home_action(self, p: np.ndarray) -> tuple[np.ndarray, bool]:
        q = p[0, :8] + self.default_joint_pos
        qvel = p[0, 8:16] if p.shape[1] >= 16 else np.zeros(8)
        settled = np.all(np.abs(q - self.teleop_home_joint_pos) <= 0.02) and np.all(np.abs(qvel) <= 0.10)
        self._home_stable_steps = self._home_stable_steps + 1 if settled else 0
        done = self._home_stable_steps >= 5
        action = (self.teleop_home_joint_pos - self.default_joint_pos) / 0.5
        return action, done

    def _world_to_base(self, position_world: np.ndarray) -> np.ndarray:
        return ROOT_R.T @ (np.asarray(position_world, dtype=np.float64) - BASE_WORLD)

    def _target_rotation_base(self) -> np.ndarray:
        return ROOT_R.T @ GRIP_R_WORLD

    def _hold_action(self, qpos: np.ndarray) -> np.ndarray:
        return (qpos - self.default_joint_pos) / 0.5

    def _advance(self, allow: bool = True):
        if not allow:
            return
        self._state_step += 1
        if self._state_step < STEPS[self._state]:
            return
        self._state_step = 0
        if self._state == "RETRACT":
            self._object_ptr += 1
            self._target_xy = None
            self._grasp_xy = None
            if self._object_ptr >= len(self._object_order):
                self._state = "DONE"
            else:
                self._state = "PRE_GRASP"
        else:
            self._state = STATES[STATES.index(self._state) + 1]

    def _detect(self, obs) -> dict[str, Detection]:
        detections = self.perception.detect(self._image(obs))
        self.last_diagnostics["detection"] = {k: v.to_dict() for k, v in detections.items()}
        return detections

    @staticmethod
    def _calibrated_xy(object_name: str, detection: Detection) -> np.ndarray:
        """Apply the task's known spawn-band prior to reject bad projections.

        The prior is not object state or GT: it is the public randomisation
        range used by the evaluator.  It prevents a partially merged yellow
        mask from placing a target outside the legal band.
        """
        xy = np.asarray(detection.world_xy, dtype=np.float64).copy()
        lo, hi = OBJECT_Y_BANDS[object_name]
        xy[1] = np.clip(xy[1], lo, hi)
        return xy

    def predicts(self, obs, current_score):
        del current_score
        p = self._proprio(obs)
        qpos = p[0, :8] + self.default_joint_pos
        if not self._home_done:
            home_action, reached = self._home_action(p)
            self.last_diagnostics.update({"state": "HOME", "object": None, "lost_detection": False})
            if reached:
                self._home_done = True
                self._state = "INIT"
                self._state_step = 0
            self._ts += 1
            return {"action": [home_action.tolist()], "giveup": False}

        if self._state == "DONE":
            self._ts += 1
            return {"action": [self._hold_action(qpos).tolist()], "giveup": False}

        detections = self._detect(obs)
        object_name = self._object_order[self._object_ptr]
        detection = detections.get(object_name)
        visible = detection is not None and detection.visible and detection.world_xy is not None and detection.confidence >= 0.20

        if self._state in ("PRE_GRASP", "REACH", "CLOSE"):
            if visible and self._target_xy is None:
                self._target_xy = self._calibrated_xy(object_name, detection)
            # Do not advance toward a guessed location when the current object
            # is lost.  The perception module may hold it for a few frames.
            if self._target_xy is None:
                self.last_diagnostics.update({"state": self._state, "object": object_name, "lost_detection": True})
                self._ts += 1
                return {"action": [self._hold_action(qpos).tolist()], "giveup": False}
            if self._state == "CLOSE" and self._grasp_xy is None:
                self._grasp_xy = self._target_xy.copy()
            # Once PRE_GRASP has acquired a target, freeze its XY through
            # REACH/CLOSE/LIFT.  Per-frame centroid jitter must not invalidate
            # the cached IK goal or drag the gripper sideways.

        if self._state == "LIFT" and self._grasp_xy is None:
            self._grasp_xy = self._target_xy.copy() if self._target_xy is not None else None

        if self._state in ("LIFT", "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT") and self._grasp_xy is None:
            self.last_diagnostics.update({"state": self._state, "object": object_name, "lost_detection": True})
            self._ts += 1
            return {"action": [self._hold_action(qpos).tolist()], "giveup": False}

        xy = self._target_xy if self._state in ("PRE_GRASP", "REACH", "CLOSE") else self._grasp_xy
        if self._state in ("TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"):
            xy = BASKET_WORLD_XY
        if self._state == "INIT":
            target_world = np.array([BASE_WORLD[0] - 0.05, BASE_WORLD[1], CARRY_Z])
            grip = OPEN_GRIPPER
        elif self._state == "PRE_GRASP":
            target_world = np.array([xy[0], xy[1], CARRY_Z])
            grip = OPEN_GRIPPER
        elif self._state in ("REACH", "CLOSE"):
            target_world = np.array([xy[0], xy[1], OBJECT_Z + GRASP_OFFSETS[object_name]])
            grip = OPEN_GRIPPER if self._state == "REACH" else CLOSE_GRIPPER
        elif self._state == "LIFT":
            target_world = np.array([xy[0], xy[1], CARRY_Z])
            grip = CLOSE_GRIPPER
        elif self._state == "TRANSPORT":
            target_world = np.array([BASKET_WORLD_XY[0], BASKET_WORLD_XY[1], CARRY_Z])
            grip = CLOSE_GRIPPER
        elif self._state == "PLACE":
            alpha = min((self._state_step + 1) / STEPS["PLACE"], 1.0)
            target_world = np.array([BASKET_WORLD_XY[0], BASKET_WORLD_XY[1], CARRY_Z + alpha * (PLACE_Z - CARRY_Z)])
            grip = CLOSE_GRIPPER
        elif self._state == "OPEN":
            target_world = np.array([BASKET_WORLD_XY[0], BASKET_WORLD_XY[1], PLACE_Z])
            grip = OPEN_GRIPPER
        elif self._state == "LIFT_RETRACT":
            target_world = np.array([BASKET_WORLD_XY[0], BASKET_WORLD_XY[1], CARRY_Z])
            grip = OPEN_GRIPPER
        else:  # RETRACT
            target_world = np.array([BASE_WORLD[0] - 0.05, BASE_WORLD[1], CARRY_Z])
            grip = OPEN_GRIPPER

        # Position is the hard safety constraint for this fixed top-down task.
        # The URDF flange orientation and the simulator gripper-base frame
        # differ by a fixed tool transform; enforcing all six orientation
        # errors here would leave a 0.5 m vertical error within the short
        # REACH window.  The bounded position IK keeps the current wrist
        # configuration smooth while the state machine preserves the grasp
        # orientation target across CLOSE/LIFT.
        ik_target_base = self._world_to_base(target_world)
        # Hold one absolute IK solution for each state target.  Re-solving
        # from the lagging measured qpos every frame makes the target move
        # faster than the actuator and prevents the end effector from ever
        # reaching the object height.
        goal_key = (self._state, *(np.asarray(target_world, dtype=np.float64).round(4).tolist()))
        if self._joint_goal_key != goal_key or self._joint_goal is None:
            self._joint_goal = self.kinematics.solve(
                qpos[:6], ik_target_base, None, iterations=80, tolerance=1e-4
            )
            self._joint_goal_key = goal_key
        q_arm = self._joint_goal.copy()
        joint_target = np.concatenate((q_arm, grip))
        if not np.isfinite(joint_target).all():
            joint_target = qpos
        # Do not apply a symmetric action clip: joint2 legitimately needs an
        # environment action above 2.0 to reach its legal 3.14-rad position
        # limit.  The IK already clips absolute targets to per-joint limits.
        action = self._hold_action(joint_target)
        self.last_diagnostics.update({
            "state": self._state,
            "object": object_name,
            "ik_target_world": target_world.tolist(),
            "lost_detection": False,
            "confidence": float(detection.confidence) if detection is not None else 0.0,
        })
        self._advance(allow=True)
        self._ts += 1
        return {"action": [action.tolist()], "giveup": False}


__all__ = ["AlgSolution"]
