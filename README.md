# Gravity Compensation

Tools and scripts for exciting, simulating, and identifying the rigid–body dynamics of fixed‑base manipulators.  The workflow targets 7‑DoF arms such as the KUKA LBR Med, Franka Emika Panda, and UFactory xArm, but can be adapted to any revolute robot by editing the configuration.

---

## Project Layout

- `src/config.py` – centralised configuration: robot model selection, trajectory hyper‑parameters, file locations, optimisation objective, and safety toggles.
- `src/TrajGeneration.py` – builds a CasADi/OptaS optimisation problem that produces minimum‑condition-number excitation trajectories. [step 1]
- `src/trajsimulation.py` – replays joint trajectories in PyBullet to synthesise joint efforts and export measurement CSV files. [step 2]
- `src/regression.py` – runs weighted least squares on recorded data to recover the minimal set of dynamic parameters. [step 3]
- `src/ground_truth_gen_sim.py` – optional end-to-end pipeline combining excitation, simulation, and identification.

Follow the step 1-2-3 to generate target trajectory csv, force recording, and dynamics identification results.
---

## Requirements

### System

- Ubuntu 22.04 LTS.
- ROS 2 Humble (provides `rclpy`, `xacro`, `ament_index_python`, `urdf_parser_py`, etc.).
- Access to the robot description packages referenced in `config.py`:
  - `gravity_compensation`
  - `med7_dock_description` [med7_dock_description](https://github.com/Tianhuanyu/med7_dock_description)
  - `xarm_description` [xarm_description](https://github.com/xArm-Developer/xarm_ros2)
  - `franka_panda_description` [franka_panda_description](https://github.com/Tianhuanyu/franka_panda_description)

> Make sure your ROS workspace sources these packages before running any of the scripts.

### Python packages

Install the Python dependencies into the same environment that sources ROS:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.1 scipy matplotlib pybullet open3d==0.17.0 \
    casadi urdf-parser-py rospkg
```

`optas` must be installed from source (the official instructions are at [cmower/optas](https://github.com/cmower/optas)).  After cloning OptaS, add it to your `PYTHONPATH` or install it with `python3 -m pip install .`.

---

## Setup

1. Clone this repository into your ROS workspace (`~/ros2_ws/src` in the examples below).
2. Build and source your workspace if required:
   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```
3. Verify that `ros2 pkg list | grep franka_panda_description` (and the other description packages) succeeds.
4. Install the Python dependencies listed above.

---

## Configuration (`src/config.py`)

All runtime options are declared in `src/config.py`.  Key sections include:

- **Package share directories** – resolved with `get_package_share_directory` so that meshes and URDFs can be found at runtime.  Adjust these if your packages live under different names.
- **File locations** – `DEFAULT_TARGET_TRAJECTORY_PATH`, `REGRESSION_MEASUREMENT_PATH`, and similar constants control where CSVs are written/read.
- **Robot selection** – `ROBOT_MODEL` and `EE_LINK_NAME` define the default URDF used by the scripts.
- **Trajectory hyper‑parameters** – `TRAJGEN_HYPERPARAMETER_SETS` stores per-robot bias vectors, joint and velocity limits, and the Fourier series rank used in optimisation.  Example:
  - `franka_panda`: bias centred on joint 4’s operational range, asymmetric limits, rank 8.
  - `kuka_med7`: symmetric bounds with rank 5.
  - `xarm7`: wide ±360° bounds with rank 6.
  Add a new robot by inserting another entry with the same keys (`bias`, `q_min`, `q_max`, `q_vmin`, `q_vmax`, `fourier_rank`).
- **Self-collision toggle** – `TRAJGEN_ENABLE_SELF_COLLISION` enables or disables the conservative convex-hull constraints during trajectory optimisation.
- **Objective definition** – `TRAJGEN_OBJECTIVE` selects the metric for the optimiser.  The default (`type: "log_cond_fro"`, `epsilon: 1e-6`) minimises `log(cond(YᵀY + εI))`, giving a numerically robust excitation quality measure.

Every script reads these settings on start-up; no command-line flags are required for the standard workflow.

---

## Typical Workflow

> Always source your ROS environment (`source ~/ros2_ws/install/setup.bash`) before running the Python scripts.

### 1. Generate an excitation trajectory

```bash
python3 src/TrajGeneration.py
```

The script:

- Loads the robot specified in `ROBOT_MODEL`.
- Applies the corresponding hyper-parameters (rank, bias, joint limits).
- Solves the CasADi/Ipopt problem with the configured objective.
- Writes the excitation sequence to `DEFAULT_TARGET_TRAJECTORY_PATH` (CSV containing joint positions and velocities).

Tune the excitation by editing the `mainO` function inside `src/TrajGeneration.py` (e.g., change `Ff`, `sampling_rate`, or the optimisation gravity vector) or by updating the hyper-parameter entry in `config.py`.

### 2. Simulate the trajectory in PyBullet (optional but useful for testing)

```bash
python3 src/trajsimulation.py
```

This script:

- Loads the same URDF in PyBullet, resolving mesh paths via ROS package shares.
- Replays the CSV generated in step 1.
- Records joint torques/forces and saves them to `TRAJSIM_OUTPUT_CSV_PATH` (defaults to `src/data/robot_data.csv`).

Set `use_gui=False` in `TrajectoryConductionSim` if you want to run the simulation headless.  You can also provide custom trajectories via `TrajectoryConductionSim.import_traj_frompath`.

### 3. Estimate dynamic parameters from measurements

```bash
python3 src/regression.py
```

The regression node:

- Loads the measurement CSV pointed to by `REGRESSION_MEASUREMENT_PATH`.
- Filters velocities/efforts, builds the regressor using `optas`, and runs weighted least squares.
- Saves the identified parameters to `REGRESSION_OUTPUT_PARAMETERS_PATH` and plots measured vs. reconstructed torques.

Ensure the measurement file includes positions and torques in the same format produced by the simulation step (or by your real robot logger).

### Optional: End-to-end dataset generation

`src/ground_truth_gen_sim.py` combines trajectory generation, PyBullet replay, and regression into a single pipeline that appends results to a CSV via `utility_math.csv_save`.  Review that script if you need automated batch generation.

---

## Additional Notes

- Trajectory optimisation relies on Ipopt via CasADi.  If the solver fails to converge, relax joint/velocity bounds or increase the Fourier rank for your robot.
- When switching robots, update `ROBOT_MODEL`, `EE_LINK_NAME`, and the relevant entry in `TRAJGEN_HYPERPARAMETER_SETS`.  The scripts automatically pick up the new configuration.
- Set `TRAJGEN_ENABLE_SELF_COLLISION = True` to restore the conservative convex-hull-based collision constraints once you are satisfied with the excitation quality.
- The examples under `src/examples/` show how to programmatically call the classes for more customised workflows (e.g., running multiple optimisation instances or integrating with ROS services).

---

## Troubleshooting

- **`ImportError: librcl_action.so`** – ensure you have sourced the ROS 2 Humble environment before running the scripts (`source /opt/ros/humble/setup.bash` and `source ~/ros2_ws/install/setup.bash`).
- **Missing meshes/URDFs** – the scripts depend on ROS packages for robot descriptions.  Install the packages listed under *Requirements* and keep your workspace sourced.
- **Ipopt reports `lb > ub`** – check the joint limit arrays in `TRAJGEN_HYPERPARAMETER_SETS`; asymmetric ranges must match the bias used for that robot.
- **Large condition numbers** – try increasing `fourier_rank`, relaxing velocity limits, or re‑enabling self-collision constraints to shape the search space.

---

Happy identifying!  Contributions and issue reports are welcome.  Feel free to open a ticket if you integrate another robot or add new optimisation metrics.
