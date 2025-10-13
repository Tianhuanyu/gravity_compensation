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
LBR_DESCRIPTION_SHARE = get_package_share_directory("lbr_description")
FRANKA_DESCRIPTION_SHARE = get_package_share_directory("franka_panda_description")
PACKAGE_SRC_DIR = os.path.dirname(__file__)
DEFAULT_TARGET_TRAJECTORY_PATH = "/tmp/target_joint_states.csv"

# ROBOT_MODEL = os.path.join(
#     GRAVITY_COMPENSATION_SHARE,
#     "urdf",
#     "med",
#     "med7dock.urdf",
# )
# EE_LINK_NAME = "ee_link"

# ROBOT_MODEL = os.path.join(
#     GRAVITY_COMPENSATION_SHARE,
#     "urdf",
#     "xarm",
#     "xarm7base.urdf",
# )

ROBOT_MODEL = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "urdf",
    "panda",
    "panda_arm.urdf",
)
EE_LINK_NAME = "panda_link8"


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
DEFAULT_TRAJGEN_MODEL = "med7dock"


def trajgen_model_path(model_name: str = DEFAULT_TRAJGEN_MODEL) -> str:
    """Return the absolute path to the requested model Xacro used by TrajGeneration."""
    return os.path.join(
        GRAVITY_COMPENSATION_SHARE,
        "urdf",
        "med",
        f"{model_name}.urdf",
    )

TRAJGEN_MODEL_PATH =  ROBOT_MODEL


TRAJGEN_MEASUREMENT_WITH_EXT_TAU_PATH = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "test",
    "measurements_with_ext_tau.csv",
)

TRAJGEN_MESH_PATH = os.path.join(
    MED7_DOCK_DESCRIPTION_SHARE,
    "meshes",
    "EndEffector.STL",
)


# ---------------------------------------------------------------------------
# Trajectory simulation resources
# ---------------------------------------------------------------------------
TRAJSIM_MED_URDF_PATH = ROBOT_MODEL

TRAJSIM_RESOURCE_PATHS = [
    MED7_DOCK_DESCRIPTION_SHARE,
    LBR_DESCRIPTION_SHARE,
    FRANKA_DESCRIPTION_SHARE,
    os.path.join(GRAVITY_COMPENSATION_SHARE, "urdf", "med"),
]

TRAJSIM_OUTPUT_CSV_PATH = REGRESSION_MEASUREMENT_PATH
