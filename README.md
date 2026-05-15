# EECS 106A Final Project: Vision-Guided Robotic Game Player

This project is a ROS 2 robotics system that uses 3D perception, game logic,
inverse kinematics, and motion planning to let a UR7e robot arm play a
Blokus-like board game with physical pieces.

Built by Connor, Varin, and Joya for UC Berkeley EECS 106A, the project
connects a RealSense depth camera, colored block detection, board calibration,
MoveIt planning, gripper control, and a turn-based strategy engine into one
hardware-facing autonomy stack.

## Why This Project Matters

This is not a simulation-only demo. The code coordinates perception, planning,
robot control, and human interaction under real-world constraints: noisy point
clouds, camera-to-robot transforms, stale sensor data, calibration offsets,
gripper timing, and safe motion gating.

For recruiters and engineering reviewers, this project demonstrates experience
with:

- Building an end-to-end robotics pipeline from sensor input to physical robot
  action
- Working with ROS 2 nodes, topics, services, launch files, parameters, and TF
- Integrating perception outputs with motion planning and execution
- Designing safety checks around stale data and explicit motion start commands
- Translating game-state decisions into real board coordinates
- Debugging hardware-facing software where calibration, timing, and transforms
  matter

## Technical Highlights

- **3D perception:** Processes RealSense `PointCloud2` data to identify colored
  clusters, detect blue robot pickup blocks, and use red corner blocks as board
  calibration markers.
- **Board localization:** Converts between game-board cells and world-frame
  robot targets using measured board geometry and configurable calibration
  offsets.
- **Motion planning:** Uses MoveIt IK and planning services to generate UR7e
  joint-space trajectories for pick, lift, place, release, reset, and tuck
  actions.
- **Robot execution:** Sends planned trajectories to the scaled joint
  trajectory controller and coordinates gripper toggles through a ROS service.
- **Game autonomy:** Maintains a Blokus-like game state, validates human moves,
  selects robot moves, and maps selected pieces to physical placement targets.
- **Operational safety:** Requires an explicit `/start_robot_move` service call
  before motion, and rejects stale cube or board poses.

## Tech Stack

- **Languages:** Python
- **Robotics:** ROS 2, MoveIt 2, TF2, UR robot tooling
- **Sensors:** Intel RealSense depth camera
- **Core libraries:** NumPy, SciPy
- **ROS interfaces:** `PointCloud2`, `PoseStamped`, `MarkerArray`,
  `FollowJointTrajectory`, `Trigger`, parameter services
- **Developer tools:** colcon, RViz, ROS CLI tools

## System Architecture

```text
RealSense Point Cloud
        |
        v
perception/process_pointcloud
        |                 |
        |                 +--> /board_test_pose
        |                 +--> /board_divot_markers
        v
  /cube_pose_blue
        |
        v
planning/main  <--- /start_robot_move --- planning/game_manager
        |                                      |
        |                                      v
        |                             logic/game_logic.py
        v
MoveIt IK + Motion Planning
        |
        v
UR7e Trajectory Controller + Gripper Service
```

## Main Packages

```text
project/src/
├── logic/
│   └── game_logic.py
├── perception/
│   └── perception/process_pointcloud.py
└── planning/
    ├── launch/project_bringup.launch.py
    └── planning/
        ├── game_manager.py
        ├── ik.py
        ├── main.py
        └── static_tf_transform.py
```

### `perception`

The perception package subscribes to the RealSense point cloud, filters and
clusters colored points, estimates block poses, publishes robot pickup poses,
and localizes board placement targets.

Key outputs:

- `/cube_pose_blue`: detected blue pickup block pose
- `/board_test_pose`: target board placement pose
- `/detected_blocks`: readable detection summary
- `/board_divot_markers`: RViz board visualization

### `planning`

The planning package consumes perception targets, computes IK, plans robot
motion with MoveIt, sends trajectories to the UR controller, and coordinates the
gripper.

Key responsibilities:

- Pick up the staged blue robot block
- Move above the calibrated board target
- Place and release the piece
- Reset and tuck the arm after the action
- Reject stale pose data before moving

### `logic`

The logic module implements the board game engine used by the game manager. It
tracks board state, legal moves, player inventories, rotations, placements, and
robot move selection.

## Runtime Flow

1. Red corner blocks define the physical board frame.
2. A blue block is staged in the robot pickup area.
3. The human enters or clicks a game move.
4. The game manager validates the human move and selects a robot response.
5. The selected robot move is sent to perception as target board cells.
6. Perception publishes a board target pose and a pickup pose.
7. The operator confirms readiness.
8. The robot performs the pick-place sequence.

## Build and Run

The active ROS workspace is `project/`. The older `lab5/` directory is retained
as reference material.

```bash
cd project
colcon build --packages-up-to planning perception
source install/setup.bash
```

Launch the camera, perception node, static transform publisher, MoveIt, and
pick-place planner:

```bash
ros2 launch planning project_bringup.launch.py
```

Start the game manager in another terminal:

```bash
cd project
source install/setup.bash
ros2 run planning game_manager
```

Human moves use:

```text
piece row col [rotation_index]
```

Example:

```text
1x2-a 0 0 0
```

## Useful Commands

Run perception only:

```bash
ros2 run perception process_pointcloud
```

Run the pick-place planner only:

```bash
ros2 run planning main
```

Run the static camera transform publisher:

```bash
ros2 run planning tf
```

Manually start a robot move:

```bash
ros2 service call /start_robot_move std_srvs/srv/Trigger {}
```

Inspect important topics:

```bash
ros2 topic echo /cube_pose_blue
ros2 topic echo /board_test_pose
ros2 topic echo /detected_blocks
ros2 topic echo /board_divot_markers
```

## Important ROS Interfaces

| Name | Type | Purpose |
| --- | --- | --- |
| `/camera/camera/depth/color/points` | `sensor_msgs/PointCloud2` | RealSense point cloud input |
| `/cube_pose_blue` | `geometry_msgs/PoseStamped` | Detected blue pickup block pose |
| `/board_test_pose` | `geometry_msgs/PoseStamped` | Target board placement pose |
| `/detected_blocks` | `std_msgs/String` | Human-readable block detection info |
| `/board_divot_markers` | `visualization_msgs/MarkerArray` | RViz board/grid visualization |
| `/place_target_marker` | `visualization_msgs/Marker` | Planned placement marker |
| `/start_robot_move` | `std_srvs/Trigger` | Starts the current pick-place sequence |
| `/toggle_gripper` | `std_srvs/Trigger` | Opens/closes the gripper |

## Calibration and Debugging

The perception node assumes a physical `10 x 12` board with a playable `8 x 10`
area offset by one cell from the outside border. Red `1 x 1` corner blocks are
used as calibration markers. Blue blocks are treated as robot pickup pieces.

Common tuning parameters:

- `block_pick_z_offset`
- `pick_x_offset`
- `pick_y_offset`
- `place_x_offset`
- `place_y_offset`
- `place_along_piece_offset`
- `place_across_piece_offset`
- `place_z_offset`
- `approach_offset`
- `grasp_offset`
- `place_down_adjustment`

Example parameter updates:

```bash
ros2 param set /process_pointcloud place_z_offset 0.005
ros2 param set /cube_grasp grasp_offset 0.145
```

Run package tests:

```bash
cd project
colcon test --packages-select planning perception
colcon test-result --verbose
```

Helpful checks:

- Confirm the RealSense point cloud is active:
  `ros2 topic hz /camera/camera/depth/color/points`
- Confirm TF connects the camera frame to `base_link`.
- Watch `/detected_blocks` to verify red and blue cluster detection.
- Use RViz to display `/board_divot_markers`, `/place_target_marker`, and the
  point cloud.

## Current Notes

`TODO.md` contains working notes around perception and rotation calibration.
The project is hardware-facing, so final offsets and board setup may vary with
the physical robot, camera mount, and game board placement.
