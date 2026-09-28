"""Task E demo-collection constants.
"""

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X, BASKET_CENTER_Y,
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, TABLE_HALF_X,
    BASKET_EXCL_HALF_X, BASKET_EXCL_HALF_Y,
)

__all__ = [
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "ACTION_SCALE",
    # state machine
    "STEPS", "STATE_ORDER",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "GRASP_Z_OFFSETS", "GRASP_LONG_AXIS",
    "GRASP_FINGER_CENTER_OBJECTS",
    "LIFT_MAX_TARGET_XY_STEP", "LIFT_MAX_LATERAL_DISPLACEMENT",
    "CARRY_Z", "PLACE_HEIGHT",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    # success check
    "BASKET_IN_X", "BASKET_IN_Y",
    # spawn regions
    "OBJ_SPAWN_X_MIN", "OBJ_SPAWN_X_MAX", "OBJ_SPAWN_Z", "OBJ_SPAWN_Y_BANDS",
    "OBJ_HALF_EXTENTS", "OBJ_BBOX_MARGIN",
    # camera
    "CAM_POS", "CAM_ROT", "CAM_H", "CAM_W",
    # actuator
    "ACT_STIFFNESS", "ACT_DAMPING", "ACT_EFFORT_LIMIT", "ACT_VEL_LIMIT",
    # warm-up
    "WARMUP_STEPS", "SETTLE_STEPS",
]

# ------------------------------------------------------------------ #
# Robot
# ------------------------------------------------------------------ #
EE_BODY_NAME        = "gripper_base"
ARM_JOINT_NAMES     = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS    = [0.035, -0.035]   # joint7, joint8
GRIPPER_CLOSE_POS   = [0.0, 0.0]

# Must match ActionsCfg: scale=0.5, use_default_offset=True
#   env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# ------------------------------------------------------------------ #
# State-machine
# ------------------------------------------------------------------ #
STEPS: dict[str, int] = {
    "INIT":         100,
    "PRE_GRASP":    200,
    "REACH":        100,
    "CLOSE":         40,
    "LIFT":         160,
    "TRANSPORT":    200,
    "PLACE":         80,
    "OPEN":          60,
    "LIFT_RETRACT":  80,
    "RETRACT":       80,
}
STATE_ORDER = ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
PRE_GRASP_CLEARANCE = 0.12   # metres above object before descent
GRASP_Z_OFFSET      = 0.09   # metres: gripper approach height above object centre
# Vertical offset of the finger-link midpoint target from the object root.
# Box and bottle start centered on their root; banana keeps the base-target
# height from the previously successful expert converted through the measured
# gripper-base-to-finger-center transform.
GRASP_Z_OFFSETS = {1: 0.000, 2: 0.000, 3: GRASP_Z_OFFSET}
# Asset-local long axes used to orient the jaw opening.  The first two Task-E
# meshes are laid out along local/world Y; banana keeps the orientation that
# passed the initial regression.
GRASP_LONG_AXIS = {1: 1, 2: 1, 3: 0}
# These two assets use the fingertip-center calibration. Banana retains the
# legacy gripper-base target that has a known successful trajectory.
GRASP_FINGER_CENTER_OBJECTS = {1, 2}

# LIFT follows the case video: the target XY may correct slowly toward the
# grasp anchor, but never jumps laterally in a single 20 ms control step.
LIFT_MAX_TARGET_XY_STEP = 0.002
LIFT_MAX_LATERAL_DISPLACEMENT = 0.08

CARRY_Z      = TABLE_TOP_Z + 0.40   # safe carry height
PLACE_HEIGHT = TABLE_TOP_Z + 0.15   # height at which to release into basket

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]   # top-down orientation (w,x,y,z)

# ------------------------------------------------------------------ #
# Object spawn regions
#
# These ranges mirror the Task E evaluation randomisation in env_cfg.py.
#   object_1  Y ∈ [0.25, 0.29]   (top band)
#   object_2  Y ∈ [0.14, 0.20]   (middle band)
#   object_3  Y ∈ [0.03, 0.09]   (bottom band, closest to basket)
# ------------------------------------------------------------------ #
OBJ_SPAWN_X_MIN = TABLE_CENTER_X - 0.10
OBJ_SPAWN_X_MAX = TABLE_CENTER_X + 0.10
OBJ_SPAWN_Z     = TABLE_TOP_Z + 0.03

# Per-object Y-bands: {object_idx: (y_min, y_max)}
OBJ_SPAWN_Y_BANDS = {
    1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
    2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
    3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09),
}

# Per-object 2-D bounding-box half-extents (metres, world XY plane, scale=1).
# Used for AABB overlap rejection during randomisation.
OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: (0.050, 0.044),   # Sugar box   (half_x, half_y)
    2: (0.050, 0.030),   # Mustard bottle
    3: (0.100, 0.040),   # Banana
}
OBJ_BBOX_MARGIN = 0.015  # extra clearance between object bounding boxes

# ------------------------------------------------------------------ #
# Success / basket bounds
# ------------------------------------------------------------------ #
BASKET_IN_X = 0.20   # ± metres in X around BASKET_CENTER_X
BASKET_IN_Y = 0.11   # ± metres in Y around BASKET_CENTER_Y

# ------------------------------------------------------------------ #
# Camera (for --save_video / --save_images)
# ------------------------------------------------------------------ #
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)   # ~34° around +Y → looks forward-down

# ------------------------------------------------------------------ #
# Actuator overrides for data collection (high stiffness for fast tracking)
# ------------------------------------------------------------------ #
ACT_STIFFNESS    = 800.0
ACT_DAMPING      = 80.0
ACT_EFFORT_LIMIT = 100.0
ACT_VEL_LIMIT    = 100.0

# ------------------------------------------------------------------ #
# Warm-up / settle steps before recording starts
# ------------------------------------------------------------------ #
WARMUP_STEPS = 150
SETTLE_STEPS = 30
