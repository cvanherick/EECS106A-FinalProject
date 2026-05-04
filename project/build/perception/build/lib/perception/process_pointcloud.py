import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from rcl_interfaces.msg import SetParametersResult


class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')

        self.target_frame = self.declare_parameter('target_frame', 'base_link').value

        self.cluster_dist_thresh = 0.003
        self.cluster_min_pts = 50
        self.block_unit = 0.03

        self.board_origin = None
        self.board_z = None
        self.CELL_SIZE = 0.03

        self.row_dir = np.array([1.0, 0.0])
        self.col_dir = np.array([0.0, 1.0])
        
        self.row_dir = self.row_dir / np.linalg.norm(self.row_dir)
        self.col_dir = self.col_dir / np.linalg.norm(self.col_dir)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        self.pose_pub = self.create_publisher(PoseStamped, '/cube_pose_red', 10)
        self.block_info_pub = self.create_publisher(String, '/detected_blocks', 10)

        # NEW: publishes board cell target for IK
        self.board_test_pose_pub = self.create_publisher(
            PoseStamped,
            '/board_test_pose',
            10
        )

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info("Red block clustering + board test pose initialized.")

    def euclidean_clustering(self, points):
        clusters = []
        visited = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if visited[i]:
                continue

            queue = [i]
            cluster_idx = []

            while queue:
                idx = queue.pop()
                if visited[idx]:
                    continue

                visited[idx] = True
                cluster_idx.append(idx)

                dists = np.linalg.norm(points - points[idx], axis=1)
                neighbors = np.where(dists < self.cluster_dist_thresh)[0]

                for n in neighbors:
                    if not visited[n]:
                        queue.append(n)

            if len(cluster_idx) >= self.cluster_min_pts:
                clusters.append(points[cluster_idx])

        return clusters

    def estimate_shape(self, pts):
        xy = pts[:, :2]
        mean = np.mean(xy, axis=0)
        xy_centered = xy - mean

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        aligned = xy_centered @ eigvecs
        lengths = np.max(aligned, axis=0) - np.min(aligned, axis=0)

        long_side = max(lengths)
        short_side = min(lengths)

        n_long = max(1, int(round(long_side / self.block_unit)))
        n_short = max(1, int(round(short_side / self.block_unit)))

        return f"{n_long}x{n_short}"

    def set_board_origin(self, cluster):
        centroid = np.mean(cluster, axis=0)

        self.board_origin = np.array([centroid[0], centroid[1]])
        self.board_z = centroid[2]


        self.get_logger().info("===== BOARD ORIGIN SET =====")
        self.get_logger().info(f"Origin P00: x={self.board_origin[0]:.4f}, y={self.board_origin[1]:.4f}")
        self.get_logger().info(f"Board z: {self.board_z:.4f}")
        self.get_logger().info("============================")

        # DEBUG: check grid directions
        for r, c in [(0,0), (5,0), (0,5), (5,5)]:
            wx, wy, _ = self.board_to_world(r, c)
            self.get_logger().info(f"({r},{c}) maps to: ({wx:.3f}, {wy:.3f})")

    def board_to_world(self, row, col):
        if self.board_origin is None:
            return None

        xy = (
            self.board_origin
            + row * self.CELL_SIZE * self.row_dir
            + col * self.CELL_SIZE * self.col_dir
        )

        return float(xy[0]), float(xy[1]), float(self.board_z)

    def publish_board_test_pose(self, row=5, col=5):
        result = self.board_to_world(row, col)

        if result is None:
            return

        x, y, z = result

        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.board_test_pose_pub.publish(pose)

        self.get_logger().info(
            f"Published /board_test_pose for cell ({row},{col}): "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}",
            throttle_duration_sec=2.0
        )

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}", throttle_duration_sec=2.0)
            return

        cloud = do_transform_cloud(msg, tf)

        raw = pc2.read_points(
            cloud,
            field_names=('x', 'y', 'z', 'rgb'),
            skip_nans=True
        )

        xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32, copy=False)

        rgb = raw['rgb'].view(np.uint32)
        r = (rgb >> 16) & 255
        g = (rgb >> 8) & 255
        b = rgb & 255

        red_mask = (r > 150) & (g < 100) & (b < 100)
        pts = xyz[red_mask]

        self.get_logger().info(
            f"total={len(xyz)}, red={np.count_nonzero(red_mask)}",
            throttle_duration_sec=2.0
        )

        if len(pts) < self.cluster_min_pts:
            return

        clusters = self.euclidean_clustering(pts)

        self.get_logger().info(f"Detected {len(clusters)} clusters", throttle_duration_sec=2.0)

        for cluster in clusters:
            shape = self.estimate_shape(cluster)

            self.get_logger().info(f"Estimated shape before origin check: {shape}", throttle_duration_sec=2.0)

            if shape == "1x1":
                if self.board_origin is None:
                    self.get_logger().info("Found 1x1 marker, setting board origin")
                    self.set_board_origin(cluster)
                else:
                    self.get_logger().info("1x1 marker seen, origin already set", throttle_duration_sec=2.0)
                continue

            self.process_block(cluster, cloud.header)

        if self.board_origin is not None:
            test = self.board_to_world(5, 5)
            self.get_logger().info(f"(5,5) maps to: {test}", throttle_duration_sec=2.0)
            self.publish_board_test_pose(5, 5)

    def process_block(self, pts, header):
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        std[std == 0] = 1e-6

        z = np.abs((pts - mean) / std)
        clean = pts[(z < 2.0).all(axis=1)]

        if len(clean) < self.cluster_min_pts:
            return

        centroid = np.mean(clean, axis=0)

        xy = clean[:, :2]
        xy_centered = xy - centroid[:2]

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        principal = eigvecs[:, np.argmax(eigvals)]
        yaw = np.arctan2(principal[1], principal[0])

        aligned = xy_centered @ eigvecs
        min_xy = np.min(aligned, axis=0)
        max_xy = np.max(aligned, axis=0)

        lengths = max_xy - min_xy

        long_side = max(lengths)
        short_side = min(lengths)

        n_long = max(1, int(round(long_side / self.block_unit)))
        n_short = max(1, int(round(short_side / self.block_unit)))

        shape = f"{n_long}x{n_short}"

        self.get_logger().info(f"Block detected: {shape}")

        pose = PoseStamped()
        pose.header = header

        pose.pose.position.x = float(centroid[0])
        pose.pose.position.y = float(centroid[1])
        pose.pose.position.z = float(centroid[2])

        half = yaw / 2.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        self.pose_pub.publish(pose)

        msg = String()
        msg.data = f"{shape}|x:{centroid[0]:.3f},y:{centroid[1]:.3f},z:{centroid[2]:.3f}"
        self.block_info_pub.publish(msg)

    def _on_parameter_update(self, params):
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()