from std_srvs.srv import Trigger
import sys
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from planning.ik import IKPlanner


class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.rotation_applied = False

        self.cube_pub = self.create_subscription(
            PoseStamped, '/cube_pose_red', self.cube_callback, 1
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 1
        )

        self.exec_ac = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        self.start_move_srv = self.create_service(
            Trigger,
            '/start_robot_move',
            self.start_robot_move_callback
        )

        self.cube_pose = None
        self.latest_cube_pose = None
        self.current_plan = None
        self.joint_state = None

        self.board_pose_sub = self.create_subscription(
            PoseStamped,
            '/board_test_pose',
            self.board_pose_callback,
            1
        )

        self.board_pose = None

        self.ik_planner = IKPlanner()

        self.job_queue = []
        self.approach_offset = float(
            self.declare_parameter('approach_offset', 0.185).value
        )
        self.grasp_offset = float(
            self.declare_parameter('grasp_offset', 0.14).value
        )
        self.place_down_adjustment = float(
            self.declare_parameter('place_down_adjustment', 0.01).value
        )
        self.place_x_offset = float(
            self.declare_parameter('place_x_offset', 0.0).value
        )
        self.place_y_offset = float(
            self.declare_parameter('place_y_offset', 0.0).value
        )
        requested_auto_start = bool(
            self.declare_parameter('auto_start', False).value
        )
        self.auto_start = False
        if requested_auto_start:
            self.get_logger().warn(
                "Ignoring auto_start=true. Robot motion now requires an "
                "explicit /start_robot_move service call."
            )
        self.add_on_set_parameters_callback(self._on_parameter_update)
        self.ur7e_utils_commands = {
            'reset_state': [
                ['ros2', 'run', 'ur7e_utils', 'reset_state'],
                ['reset_state']
            ],
            'tuck': [
                ['ros2', 'run', 'ur7e_utils', 'tuck'],
                ['tuck']
            ]
        }

        self.target_marker_pub = self.create_publisher(
            Marker,
            '/place_target_marker',
            10
        )

    def publish_place_marker(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'place_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.target_marker_pub.publish(marker)

    def board_pose_callback(self, msg):
        self.board_pose = msg

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def _on_parameter_update(self, params):
        for param in params:
            if param.name == 'approach_offset':
                self.approach_offset = float(param.value)
            elif param.name == 'grasp_offset':
                self.grasp_offset = float(param.value)
            elif param.name == 'place_down_adjustment':
                self.place_down_adjustment = float(param.value)
            elif param.name == 'place_x_offset':
                self.place_x_offset = float(param.value)
            elif param.name == 'place_y_offset':
                self.place_y_offset = float(param.value)
            elif param.name == 'auto_start':
                self.auto_start = False
                if bool(param.value):
                    self.get_logger().warn(
                        "Ignoring auto_start=true. Use /start_robot_move to "
                        "begin motion."
                    )

        self.get_logger().info(
            "Updated calibration: "
            f"place_xy=({self.place_x_offset:.4f},{self.place_y_offset:.4f}), "
            f"approach={self.approach_offset:.4f}, "
            f"grasp={self.grasp_offset:.4f}, "
            f"place_down={self.place_down_adjustment:.4f}"
        )
        return SetParametersResult(successful=True)

    def start_robot_move_callback(self, request, response):
        if self.latest_cube_pose is None:
            response.success = False
            response.message = 'No red block pose available yet'
            return response

        if self.cube_pose is not None:
            response.success = False
            response.message = 'Robot move already in progress'
            return response

        self.get_logger().info('Starting robot move from game_manager')
        started = self.start_pick_place(self.latest_cube_pose)
        response.success = bool(started)
        response.message = (
            'Robot move started' if started else 'Robot move failed to start'
        )
        return response

    def cube_callback(self, cube_pose):
        self.latest_cube_pose = cube_pose

    def start_pick_place(self, cube_pose):
        if self.cube_pose is not None:
            return False

        self.rotation_applied = False

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return False

        # Wait until perception has published the four-corner board target.
        # Do this before latching cube_pose so a later cube message can retry.
        if self.board_pose is None:
            self.get_logger().info(
                "No board pose yet, waiting for /board_test_pose",
                throttle_duration_sec=2.0
            )
            return False

        self.cube_pose = cube_pose
        q = cube_pose.pose.orientation

        yaw_angle = 2.0 * np.arctan2(q.z, q.w)
        perpendicular_yaw = yaw_angle + (np.pi / 2.0)

        q_final_rot = R.from_euler('ZYX', [perpendicular_yaw, np.pi, 0.0])
        q_final = q_final_rot.as_quat()

        board_q = self.board_pose.pose.orientation
        board_yaw = 2.0 * np.arctan2(board_q.z, board_q.w)
        board_place_yaw = board_yaw
        q_place_rot = R.from_euler('ZYX', [board_place_yaw, np.pi, 0.0])
        q_place = q_place_rot.as_quat()

        pick_x = cube_pose.pose.position.x
        pick_y = cube_pose.pose.position.y
        pick_z = cube_pose.pose.position.z

        # --- PRE-GRASP ---
        # Seed from current robot state.
        pre_grasp_joints = self.ik_planner.compute_ik(
            self.joint_state,
            pick_x,
            pick_y,
            pick_z + self.approach_offset,
            qx=float(q_final[0]),
            qy=float(q_final[1]),
            qz=float(q_final[2]),
            qw=float(q_final[3])
        )
        if not pre_grasp_joints:
            self.get_logger().error("Pre-grasp IK failed, aborting")
            self.cube_pose = None
            return False

        # --- GRASP ---
        # Seed from pre_grasp: the robot will be at that config when it descends.
        grasp_joints = self.ik_planner.compute_ik(
            pre_grasp_joints,
            pick_x,
            pick_y,
            pick_z + self.grasp_offset,
            qx=float(q_final[0]),
            qy=float(q_final[1]),
            qz=float(q_final[2]),
            qw=float(q_final[3])
        )
        if not grasp_joints:
            self.get_logger().error("Grasp IK failed, aborting")
            self.cube_pose = None
            return False

        self.job_queue.append(pre_grasp_joints)
        self.job_queue.append(grasp_joints)
        self.job_queue.append('toggle_grip')
        self.job_queue.append(pre_grasp_joints)

        board_x = self.board_pose.pose.position.x + self.place_x_offset
        board_y = self.board_pose.pose.position.y + self.place_y_offset
        board_z = self.board_pose.pose.position.z
        place_hover_z = board_z + self.approach_offset
        place_z = board_z + self.grasp_offset - self.place_down_adjustment

        self.get_logger().info(
            f"Board divot target: x={board_x:.3f}, y={board_y:.3f}, "
            f"z={place_z:.3f}, yaw={board_yaw:.3f}"
        )

        self.publish_place_marker(board_x, board_y, place_z)

        # --- PLACE HOVER ---
        # Seed from pre_grasp: the robot returns there after lifting off the block.
        place_hover_joints = self.ik_planner.compute_ik(
            pre_grasp_joints,
            board_x,
            board_y,
            place_hover_z,
            qx=float(q_place[0]),
            qy=float(q_place[1]),
            qz=float(q_place[2]),
            qw=float(q_place[3])
        )
        if not place_hover_joints:
            self.get_logger().error("Place hover IK failed, aborting")
            self.job_queue = []
            self.cube_pose = None
            return False

        # --- PLACE ---
        # Seed from place_hover: the robot descends from that config.
        place_joints = self.ik_planner.compute_ik(
            place_hover_joints,
            board_x,
            board_y,
            place_z,
            qx=float(q_place[0]),
            qy=float(q_place[1]),
            qz=float(q_place[2]),
            qw=float(q_place[3])
        )
        if not place_joints:
            self.get_logger().error("Place IK failed, aborting")
            self.job_queue = []
            self.cube_pose = None
            return False

        self.get_logger().info("All IK solutions found, executing sequence")
        self.job_queue.append(place_hover_joints)
        self.job_queue.append(place_joints)
        self.job_queue.append('toggle_grip')
        self.job_queue.append(place_hover_joints)
        self.job_queue.append('reset_state')
        self.job_queue.append('tuck')

        self.execute_jobs()
        return True

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)

            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return

            self.get_logger().info("Planned to position")
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()

        elif next_job in ('reset_state', 'tuck'):
            self._run_command(next_job)

        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()

    def _run_command(self, command):
        candidate_cmds = self.ur7e_utils_commands.get(command, [[command]])

        for cmd in candidate_cmds:
            self.get_logger().info(f"Running {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    timeout=30.0
                )
            except FileNotFoundError:
                self.get_logger().warn(f"{cmd[0]} not found")
                continue
            except subprocess.TimeoutExpired:
                self.get_logger().warn(f"{' '.join(cmd)} timed out")
                continue

            if result.returncode == 0:
                self.get_logger().info(f"{' '.join(cmd)} complete")
                self.execute_jobs()
                return

            self.get_logger().warn(
                f"{' '.join(cmd)} failed with return code {result.returncode}"
            )

        self.get_logger().error(f"All {command} command attempts failed")
        self.execute_jobs()

    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
