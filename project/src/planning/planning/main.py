from std_srvs.srv import Trigger
import sys
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

from planning.ik import IKPlanner

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')
        self.rotation_applied = False
        self.cube_pub = self.create_subscription(PoseStamped, '/cube_pose_red', self.cube_callback, 1)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()

        self.job_queue = []

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def cube_callback(self, cube_pose):
        if self.cube_pose is not None:
            return

        self.rotation_applied = False
        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        self.cube_pose = cube_pose
        q = cube_pose.pose.orientation

        yaw_angle = 2.0 * np.arctan2(q.z, q.w)
        perpendicular_yaw = yaw_angle + (np.pi / 2.0)
        
        q_final_rot = R.from_euler('ZYX', [perpendicular_yaw, np.pi, 0.0])
        q_final = q_final_rot.as_quat()

        pre_grasp_joints = self.ik_planner.compute_ik(
            self.joint_state,
            cube_pose.pose.position.x,
            cube_pose.pose.position.y,
            cube_pose.pose.position.z + 0.185,
            qx=float(q_final[0]),
            qy=float(q_final[1]),
            qz=float(q_final[2]),
            qw=float(q_final[3])
        )
        
        if pre_grasp_joints:
            self.job_queue.append(pre_grasp_joints)

        print(cube_pose.pose.position.x, cube_pose.pose.position.y, cube_pose.pose.position.z)

        grasp_joints = self.ik_planner.compute_ik(
            self.joint_state,
            cube_pose.pose.position.x,
            cube_pose.pose.position.y,
            cube_pose.pose.position.z + 0.14,
            qx=float(q_final[0]),
            qy=float(q_final[1]),
            qz=float(q_final[2]),
            qw=float(q_final[3])
        )
        
        if grasp_joints:
            self.job_queue.append(grasp_joints)

        self.job_queue.append('toggle_grip')

        if pre_grasp_joints:
            self.job_queue.append(pre_grasp_joints)

        release_joints = self.ik_planner.compute_ik(
                self.joint_state,
                cube_pose.pose.position.x,
                cube_pose.pose.position.y + 0.1,
                cube_pose.pose.position.z + 0.185,
                qx=float(q_final[0]),
                qy=float(q_final[1]),
                qz=float(q_final[2]),
                qw=float(q_final[3])
            )
        
        if release_joints:
            self.job_queue.append(release_joints)

        self.job_queue.append('toggle_grip')

        self.execute_jobs()

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

    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
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