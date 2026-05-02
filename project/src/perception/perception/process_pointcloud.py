import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from rcl_interfaces.msg import SetParametersResult


class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')

        self.target_frame = self.declare_parameter(
            'camera_depth_optical_frame', 'base_link'
        ).value

        self.max_y = float(self.declare_parameter('max_y', 0.67).value)
        self.min_z = float(self.declare_parameter('min_z', -0.18).value)
        self.max_z = float(self.declare_parameter('max_z', -0.15).value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        self.filtered_points_pub = self.create_publisher(
            PointCloud2, '/filtered_points', 1
        )

        self.pub_red = None
        self.pub_blue = None
        self.pub_green = None

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info("Multi-color point cloud tracker initialized.")

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}")
            return

        cloud = do_transform_cloud(msg, tf)

        raw = pc2.read_points(
            cloud,
            field_names=('x', 'y', 'z', 'rgb'),
            skip_nans=True
        )

        xyz = np.column_stack(
            (raw['x'], raw['y'], raw['z'])
        ).astype(np.float32, copy=False)

        rgb = raw['rgb'].view(np.uint32)
        r = (rgb >> 16) & 255
        g = (rgb >> 8) & 255
        b = rgb & 255

        spatial_mask = (
            (xyz[:, 1] <= self.max_y) &
            (xyz[:, 2] <= self.max_z) &
            (xyz[:, 2] >= self.min_z)
        )

        masks = {
            "red": (r > 150) & (g < 100) & (b < 100),
            "blue": (b > 150) & (r < 100) & (g < 100),
            "green": (g > 150) & (r < 100) & (b < 100)
        }

        for color, color_mask in masks.items():
            self.process_object(color, cloud, xyz, spatial_mask & color_mask)

    def process_object(self, color, cloud, xyz, mask):
        pts = xyz[mask]

        if len(pts) < 50:
            return

        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        std[std == 0] = 1e-6

        z = np.abs((pts - mean) / std)
        clean_mask = (z[:, 0] < 2.0) & (z[:, 1] < 2.0) & (z[:, 2] < 2.0)
        clean = pts[clean_mask]

        if len(clean) < 50:
            return

        filtered_cloud = pc2.create_cloud_xyz32(
            cloud.header,
            clean.tolist()
        )
        self.filtered_points_pub.publish(filtered_cloud)

        centroid = np.mean(clean, axis=0)

        xy = clean[:, :2]
        xy_centered = xy - centroid[:2]

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        principal = eigvecs[:, np.argmax(eigvals)]
        yaw = np.arctan2(principal[1], principal[0])
        pose = PoseStamped()
        pose.header = cloud.header

        pose.pose.position.x = float(centroid[0])
        pose.pose.position.y = float(centroid[1])
        pose.pose.position.z = float(centroid[2])

        half = yaw / 2.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        if color == "red":
            if self.pub_red is None:
                self.pub_red = self.create_publisher(PoseStamped, '/cube_pose_red', 1)
            self.pub_red.publish(pose)

        elif color == "blue":
            if self.pub_blue is None:
                self.pub_blue = self.create_publisher(PoseStamped, '/cube_pose_blue', 1)
            self.pub_blue.publish(pose)

        elif color == "green":
            if self.pub_green is None:
                self.pub_green = self.create_publisher(PoseStamped, '/cube_pose_green', 1)
            self.pub_green.publish(pose)

    def _on_parameter_update(self, params):
        for p in params:
            if p.name == 'min_z':
                self.min_z = float(p.value)
            elif p.name == 'max_z':
                self.max_z = float(p.value)
            elif p.name == 'max_y':
                self.max_y = float(p.value)

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()