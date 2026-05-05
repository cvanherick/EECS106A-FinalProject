# EECS106A Final Project TODO - Rotation & Perception Fixes

## 1. Environment Setup (Critical - ros2 not available)
```
cd /Users/revivedbonsai63/Desktop/'Personal Projects'/EECS106A-FinalProject/project
source install/setup.bash
```
Verify: `ros2 topic list` (should work). If not, `colcon build --packages-up-to planning perception`.

## 2. Diagnose Perception (process_pointcloud.py working yesterday?)
Run:
```
ros2 run perception process_pointcloud
```
- Check logs: clusters detected? pts count? centroids?
- `ros2 topic echo /cube_pose_red`
- RViz: /camera/color/image_raw, /camera/camera/depth/color/points, /filtered_points?
- Symptoms: No pose pub? Wrong Z (neg)? No red detect? Cam TF fail?

**Code changes if needed:**
- Relax filters: min_z=-0.5, max_z=0.1, max_y=1.0
- Dynamic color (HSV red)
- Size filter: bounding box 0.02-0.06m

## 3. Fix Rotation (quat order & wrist override)
**main.py:**
- q_final = q_yaw * q_down (not reverse)
- Remove `pre_grasp_joints.position[idx] += np.pi / 2.0` block

**ik.py:**
- Match: q_final = q_yaw * q_pitch_down

## 4. Test Grasp
```
ros2 launch planning lab5_bringup.launch.py
# Place red cube, check gripper orients 90° to principal axis (not straight down)
```

Progress: [] Environment | [] Perception | [] Rotation | [] Test

