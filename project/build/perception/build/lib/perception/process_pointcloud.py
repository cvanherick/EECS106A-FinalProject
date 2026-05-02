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

        self.target_frame = self.declare_parameter(
            'camera_depth_optical_frame', 'base_link'
        ).value

        self.max_y = float(self.declare_parameter('max_y', 0.67).value)
        self.min_z = float(self.declare_parameter('min_z', -0.18).value)
        self.max_z = float(self.declare_parameter('max_z', -0.15).value)

        # 🔧 clustering + sizing params
        self.cluster_dist_thresh = 0.008   # meters
        self.cluster_min_pts = 50
        self.block_unit = 0.016  # meters per stud (MEASURE THIS!)

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

        self.pose_pub = self.create_publisher(
            PoseStamped, '/cube_pose_red', 10
        )

        self.block_info_pub = self.create_publisher(
            String, '/detected_blocks', 10
        )

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info("Red block clustering + sizing initialized.")

    # ----------------------------
    # Euclidean clustering (no ML)
    # ----------------------------
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

            if len(cluster_idx) > self.cluster_min_pts:
                clusters.append(points[cluster_idx])

        return clusters

    # ----------------------------
    # Main callback
    # ----------------------------
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

        # 🔴 Only RED now
        red_mask = (r > 150) & (g < 100) & (b < 100)

        pts = xyz[spatial_mask & red_mask]

        if len(pts) < self.cluster_min_pts:
            return

        # ----------------------------
        # CLUSTERING STEP
        # ----------------------------
        clusters = self.euclidean_clustering(pts)

        self.get_logger().info(f"Detected {len(clusters)} clusters")

        for cluster in clusters:
            self.process_block(cluster, cloud.header)

    # ----------------------------
    # Process individual block
    # ----------------------------
    def process_block(self, pts, header):

        # Remove outliers (z-score)
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        std[std == 0] = 1e-6

        z = np.abs((pts - mean) / std)
        clean = pts[(z < 2.0).all(axis=1)]

        if len(clean) < self.cluster_min_pts:
            return

        centroid = np.mean(clean, axis=0)

        # ----------------------------
        # PCA for orientation
        # ----------------------------
        xy = clean[:, :2]
        xy_centered = xy - centroid[:2]

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        principal = eigvecs[:, np.argmax(eigvals)]
        yaw = np.arctan2(principal[1], principal[0])

        # ----------------------------
        # Bounding box (PCA frame)
        # ----------------------------
        aligned = xy_centered @ eigvecs
        min_xy = np.min(aligned, axis=0)
        max_xy = np.max(aligned, axis=0)

        lengths = max_xy - min_xy

        long_side = max(lengths)
        short_side = min(lengths)

        # ----------------------------
        # Convert to block units
        # ----------------------------
        n_long = int(round(long_side / self.block_unit))
        n_short = int(round(short_side / self.block_unit))

        shape = f"{n_long}x{n_short}"

        self.get_logger().info(f"Block detected: {shape}")

        # ----------------------------
        # Publish pose (same as before)
        # ----------------------------
        pose = PoseStamped()
        pose.header = header

        pose.pose.position.x = float(centroid[0])
        pose.pose.position.y = float(centroid[1])
        pose.pose.position.z = float(centroid[2])

        half = yaw / 2.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        self.pose_pub.publish(pose)

        # ----------------------------
        # Publish block info
        # ----------------------------
        msg = String()
        msg.data = f"{shape}|x:{centroid[0]:.3f},y:{centroid[1]:.3f}"
        self.block_info_pub.publish(msg)

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