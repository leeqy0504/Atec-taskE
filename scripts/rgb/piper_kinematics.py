"""Small, dependency-light Piper arm kinematics for RGB deployment.

The deployment path must not depend on Isaac Lab controllers.  This module
uses the joint origins from the Piper URDF and exposes position/orientation
FK, a geometric Jacobian, and a bounded damped-least-squares IK step.  The
gripper joints are passed through by the controller; this class solves the
six arm joints only.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rpy(rpy)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return out


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s, v = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
    ], dtype=np.float64)


@dataclass(frozen=True)
class PiperModel:
    origins: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.123),
        (0.0, 0.0, 0.0),
        (0.28503, 0.0, 0.0),
        (-0.021984, -0.25075, 0.0),
        (0.0, 0.0, 0.0),
        (0.000088259, -0.091, 0.0),
    )
    rpy: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (1.5708, -0.1359, -3.1416),
        (0.0, 0.0, -1.7939),
        (1.5708, 0.0, 0.0),
        (-1.5708, 0.0, 0.0),
        (1.5708, 0.0, 0.0),
    )
    axes: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    )
    lower: tuple[float, ...] = (-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944)
    upper: tuple[float, ...] = (2.618, 3.14, 0.0, 1.745, 1.22, 2.0944)


class PiperKinematics:
    """FK/Jacobian/IK in the Piper base frame."""

    def __init__(self, model: PiperModel | None = None, max_delta: float = 0.08):
        self.model = model or PiperModel()
        self.max_delta = float(max_delta)
        self._fixed = tuple(_transform(x, r) for x, r in zip(self.model.origins, self.model.rpy))

    def _chain(self, q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        T = np.eye(4, dtype=np.float64)
        joint_pos: list[np.ndarray] = []
        joint_axis: list[np.ndarray] = []
        for i in range(6):
            T = T @ self._fixed[i]
            joint_pos.append(T[:3, 3].copy())
            joint_axis.append(T[:3, :3] @ np.asarray(self.model.axes[i], dtype=np.float64))
            R = np.eye(4, dtype=np.float64)
            R[:3, :3] = _axis_angle(self.model.axes[i], q[i])
            T = T @ R
        return T, joint_pos, joint_axis

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 flange pose in the Piper base frame."""
        return self._chain(q)[0]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        T, positions, axes = self._chain(q)
        p = T[:3, 3]
        J = np.zeros((6, 6), dtype=np.float64)
        for i, (origin, axis) in enumerate(zip(positions, axes)):
            J[:3, i] = np.cross(axis, p - origin)
            J[3:, i] = axis
        return J

    @staticmethod
    def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Rotation vector for target * current^-1.  The small-angle branch is
        # sufficient for the incremental controller and avoids acos noise.
        R = target @ current.T
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    def ik_step(
        self,
        q: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray | None = None,
        damping: float = 0.04,
        gain: float = 0.65,
    ) -> np.ndarray:
        """Take one bounded IK step and return a legal absolute joint vector."""
        q = np.clip(np.asarray(q, dtype=np.float64).reshape(6), self.model.lower, self.model.upper)
        T = self.fk(q)
        error = np.asarray(target_position, dtype=np.float64).reshape(3) - T[:3, 3]
        if target_rotation is not None:
            task_error = np.concatenate((error, self._orientation_error(T[:3, :3], np.asarray(target_rotation))))
            J = self.jacobian(q)
        else:
            task_error = error
            J = self.jacobian(q)[:3]
        JJ = J @ J.T
        dq = gain * J.T @ np.linalg.solve(JJ + (damping ** 2) * np.eye(JJ.shape[0]), task_error)
        dq = np.clip(dq, -self.max_delta, self.max_delta)
        return np.clip(q + dq, self.model.lower, self.model.upper)

    def solve(
        self,
        q0: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray | None = None,
        iterations: int = 80,
        tolerance: float = 1e-3,
    ) -> np.ndarray:
        q = np.asarray(q0, dtype=np.float64).reshape(6).copy()
        for _ in range(max(1, int(iterations))):
            q_next = self.ik_step(q, target_position, target_rotation)
            if np.linalg.norm(q_next - q) < 1e-7:
                break
            q = q_next
            if np.linalg.norm(np.asarray(target_position) - self.fk(q)[:3, 3]) <= tolerance:
                break
        return q


__all__ = ["PiperKinematics", "PiperModel"]
