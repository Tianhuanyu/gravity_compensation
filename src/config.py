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

# ---------------------------------------------------------------------------
# Regression (parameter estimation) resources
# ---------------------------------------------------------------------------
REGRESSION_MODEL_PATH = os.path.join(
    MED7_DOCK_DESCRIPTION_SHARE,
    "urdf",
    "med7dock.urdf.xacro",
)

REGRESSION_PRIMARY_MEASUREMENT_PATH = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "test",
    "robot_data copy 2.csv",
)

REGRESSION_SECONDARY_MEASUREMENT_PATH = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "test",
    "measurements_0dgr.csv",
)

REGRESSION_OUTPUT_PARAMETERS_PATH = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "test",
    "DynamicParameters.csv",
)

# ---------------------------------------------------------------------------
# Trajectory generation resources
# ---------------------------------------------------------------------------
DEFAULT_TRAJGEN_MODEL = "med7"


def trajgen_model_path(model_name: str = DEFAULT_TRAJGEN_MODEL) -> str:
    """Return the absolute path to the requested model Xacro used by TrajGeneration."""
    return os.path.join(
        LBR_DESCRIPTION_SHARE,
        "urdf",
        model_name,
        f"{model_name}.urdf.xacro",
    )


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

DEFAULT_TARGET_TRAJECTORY_PATH = "/tmp/target_joint_states.csv"

XARM_URDF_PATH = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "urdf",
    "xarm",
    "xarm7base.urdf.xacro",
)

# ---------------------------------------------------------------------------
# Trajectory simulation resources
# ---------------------------------------------------------------------------
TRAJSIM_MED_URDF_PATH = os.path.join(
    GRAVITY_COMPENSATION_SHARE,
    "urdf",
    "med",
    "med7dock.urdf",
)

TRAJSIM_RESOURCE_PATHS = [
    MED7_DOCK_DESCRIPTION_SHARE,
    LBR_DESCRIPTION_SHARE,
    os.path.join(GRAVITY_COMPENSATION_SHARE, "urdf", "med"),
]
