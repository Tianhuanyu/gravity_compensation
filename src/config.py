"""
Shared configuration values for the gravity compensation project.

This module resolves commonly used paths up front so that other scripts can
simply import and use them without repeating `get_package_share_directory`
and `os.path.join` calls.
"""

import os
from ament_index_python import get_package_share_directory

# Package share directories
GRAVITY_COMPENSATION_SHARE = get_package_share_directory("gravity_compensation")
MED7_DOCK_DESCRIPTION_SHARE = get_package_share_directory("med7_dock_description")
FRANKA_DESCRIPTION_SHARE = get_package_share_directory("franka_panda_description")
XARM_DESCRIPTION_SHARE = get_package_share_directory("xarm_description")
PACKAGE_SRC_DIR = os.path.dirname(__file__)
DEFAULT_TARGET_TRAJECTORY_PATH = "/tmp/target_joint_states.csv"

ROBOT_MODEL = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "urdf",
    "med",
    "med7dock.urdf",
)
EE_LINK_NAME = "link_ee"

# ROBOT_MODEL = os.path.join(
#     GRAVITY_COMPENSATION_SHARE,
#     "urdf",
#     "xarm",
#     "xarm7base.urdf",
# )

# ROBOT_MODEL = os.path.join(
#     GRAVITY_COMPENSATION_SHARE,
#     "urdf",
#     "panda",
#     "panda_arm.urdf",
# )
# EE_LINK_NAME = "panda_link8"


TRAJGEN_MESH_PATH = os.path.join(
    MED7_DOCK_DESCRIPTION_SHARE,
    "meshes",
    "EndEffector.STL",
)


# ---------------------------------------------------------------------------
# Trajectory generation hyperparameters
# ---------------------------------------------------------------------------
# Bounds and nominal biases for the optimisation depend on the robot model.
# Keeping them here makes it easy to tune without touching the solver code.
_DEFAULT_TRAJGEN_HYPERPARAMETERS = {
    "bias": [0.0] * 7,
    "q_min": [-2.0] * 7,
    "q_max": [2.0] * 7,
    "q_vmin": [-8.0] * 7,
    "q_vmax": [8.0] * 7,
    "fourier_rank": 5,
}

TRAJGEN_HYPERPARAMETER_SETS = {
    # Franka Emika Panda arm
    "franka_panda": {
        "bias": [0.0, 0.0, 0.0, -1.5708, 0.0, 1.0, 0.0],
        "q_min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        "q_max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        "q_vmin": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.5, -2.5],
        "q_vmax": [2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5],
        "fourier_rank": 8,
    },
    # KUKA LBR / Med7 style arm
    "kuka_med7": {
        "bias": [0.0, 0.0, 0.0, 1.9, 0.0, 1.0, 0.0],
        "q_min": [-6.2, -2.0, -6.2, -0.19, -6.2, -1.69, -6.2],
        "q_max": [6.2, 2.0, 6.2, 3.94, 6.2, 3.14, 6.2],
        "q_vmin": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        "q_vmax": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "fourier_rank": 5,
    },
    # UFactory xArm 7 DoF
    "xarm7": {
        "bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "q_min": [
            -6.283185307179586,
            -2.059,
            -6.283185307179586,
            -0.19198,
            -6.283185307179586,
            -1.69297,
            -6.283185307179586,
        ],
        "q_max": [
            6.283185307179586,
            2.0944,
            6.283185307179586,
            3.927,
            6.283185307179586,
            3.141592653589793,
            6.283185307179586,
        ],
        "q_vmin": [-3.14] * 7,
        "q_vmax": [3.14] * 7,
        "fourier_rank": 6,
    },
}


def _infer_trajgen_hyperparam_key(model_path: str) -> str:
    """Return the config key best matching the provided model path."""
    basename = os.path.basename(model_path)
    if "panda" in basename:
        return "franka_panda"
    if "med7" in basename or "iiwa" in basename or "kuka" in basename:
        return "kuka_med7"
    if "xarm" in basename:
        return "xarm7"
    return ""


def trajgen_hyperparameters(model_path: str) -> dict:
    """Return copies of the trajectory generation hyperparameters for a model."""
    key = _infer_trajgen_hyperparam_key(model_path)
    params = TRAJGEN_HYPERPARAMETER_SETS.get(key, _DEFAULT_TRAJGEN_HYPERPARAMETERS)
    result = {}
    for k, v in params.items():
        result[k] = list(v) if isinstance(v, list) else v
    return result

# Enable/disable self-collision inequality constraints during trajectory optimisation.
TRAJGEN_ENABLE_SELF_COLLISION = False

# Trajectory optimisation objective configuration.
# type: either 'log_cond_fro' (default) or future variants
TRAJGEN_OBJECTIVE = {
    "type": "log_cond_fro",
    "epsilon": 1e-6,
}


# ---------------------------------------------------------------------------
# Regression (parameter estimation) resources
# ---------------------------------------------------------------------------
REGRESSION_MODEL_PATH = ROBOT_MODEL

REGRESSION_MEASUREMENT_PATH = os.path.join(
    PACKAGE_SRC_DIR,
    "data",
    "robot_data.csv",
)

REGRESSION_OUTPUT_PARAMETERS_PATH = os.path.join(
    PACKAGE_SRC_DIR,
    "data",
    "DynamicParameters.csv",
)


# ---------------------------------------------------------------------------
# Trajectory generation resources
# ---------------------------------------------------------------------------

TRAJGEN_MODEL_PATH =  ROBOT_MODEL


# TRAJGEN_MESH_PATH = os.path.join(
#     MED7_DOCK_DESCRIPTION_SHARE,
#     "meshes",
#     "EndEffector.STL",
# )


# ---------------------------------------------------------------------------
# Trajectory simulation resources
# ---------------------------------------------------------------------------
TRAJSIM_MED_URDF_PATH = ROBOT_MODEL
TRAJSIM_OUTPUT_CSV_PATH = REGRESSION_MEASUREMENT_PATH

TRAJSIM_RESOURCE_PATHS = [
    MED7_DOCK_DESCRIPTION_SHARE,
    FRANKA_DESCRIPTION_SHARE,
    os.path.join(GRAVITY_COMPENSATION_SHARE, "urdf", "med"),
]
DEFAULT_TRAJGEN_MODEL = "med7dock"
def trajgen_model_path(model_name: str = DEFAULT_TRAJGEN_MODEL) -> str:
    """Return the absolute path to the requested model Xacro used by TrajGeneration."""
    return os.path.join(
        GRAVITY_COMPENSATION_SHARE,
        "urdf",
        "med",
        f"{model_name}.urdf",
    )
