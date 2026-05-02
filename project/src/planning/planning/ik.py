import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation as R


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')

        self.cb_group = ReentrantCallbackGroup()

        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.cb_group)
        self.plan_client = self.create_client(
            GetMotionPlan, '/plan_kinematic_path', callback_group=self.cb_group)

        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')

        self.target_pose = None
        self.is_planning = False
        self.execution_lock = False

        self.subs = []
        for color in ['red', 'blue', 'green']:
            self.subs.append(
                self.create_subscription(
                    PoseStamped,
                    f'/cube_pose_{color}',
                    self.pose_callback,
                    10
                )
            )

        self.timer = self.create_timer(0.5, self.timer_callback, callback_group=self.cb_group)

    def pose_callback(self, msg: PoseStamped):
        if self.is_planning or self.execution_lock:
            return
        self.target_pose = msg

    def timer_callback(self):
        if self.target_pose is None or self.is_planning or self.execution_lock:
            return

        self.is_planning = True
        self.execution_lock = True

        msg = self.target_pose
        self.target_pose = None

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z + 0.15

        q_yaw = R.from_quat([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        q_down = R.from_quat([0.0, 0.0, 0.0, 1.0])

        q_final = (q_down * q_yaw).as_quat()

        current_state = JointState()
        current_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        current_state.position = [4.722, -1.850, -1.425, -1.405, 1.593, -3.141]

        ik_result = self.compute_ik(
            current_state, x, y, z,
            qx=float(q_final[0]),
            qy=float(q_final[1]),
            qz=float(q_final[2]),
            qw=float(q_final[3])
        )

        if ik_result:
            self.plan_to_joints(ik_result)

        self.is_planning = False

    def compute_ik(self, current_joint_state, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.robot_state.joint_state = current_joint_state
        ik_req.ik_request.ik_link_name = 'tool0'
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout = Duration(sec=5)
        ik_req.ik_request.group_name = 'ur_manipulator'

        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.execution_lock = False
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.execution_lock = False
            return None

        return result.solution.joint_state

    def plan_to_joints(self, target_joint_state):
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.planner_id = "RRTConnectkConfigDefault"

        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )

        req.motion_plan_request.goal_constraints.append(goal_constraints)
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.execution_lock = False
            return None

        result = future.result()
        if result.motion_plan_response.error_code.val != 1:
            self.execution_lock = False
            return None

        self.execution_lock = False
        return result.motion_plan_response.trajectory


def main(args=None):
    rclpy.init(args=args)
    node = IKPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()