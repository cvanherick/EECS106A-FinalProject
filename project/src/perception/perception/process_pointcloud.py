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
            'target_frame',
            'base_link'
        ).value

        self.cluster_dist_thresh = 0.003
        self.cluster_min_pts = 50
        self.block_unit = 0.03175

        # NOTE: If the robot is overshooting, measure the physical
        # distance between two peg centers and update this number!
        self.CELL_SIZE = 0.03175

        self.board_rows = max(
            1,
            int(self.declare_parameter('board_rows', 12).value)
        )
        self.board_cols = max(
            1,
            int(self.declare_parameter('board_cols', 10).value)
        )

        self.board_origin = None
        self.board_z = None
        self.row_step = self.CELL_SIZE
        self.col_step = self.CELL_SIZE
        self.corner_span_rows = max(
            1,
            int(
                self.declare_parameter(
                    'corner_span_rows',
                    self.board_rows - 1
                ).value
            )
        )
        self.corner_span_cols = max(
            1,
            int(
                self.declare_parameter(
                    'corner_span_cols',
                    self.board_cols - 1
                ).value
            )
        )
        self.target_row = float(
            self.declare_parameter(
                'target_row',
                (self.board_rows - 1) / 2.0
            ).value
        )
        self.target_col = float(
            self.declare_parameter(
                'target_col',
                (self.board_cols - 1) / 2.0
            ).value
        )

        # Hardcoded axes. Y is flipped to 1.0 to prevent moving the wrong way.
        self.row_dir = np.array([-1.0, 0.0])
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

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/cube_pose_red',
            10
        )
        self.block_info_pub = self.create_publisher(
            String,
            '/detected_blocks',
            10
        )

        self.board_test_pose_pub = self.create_publisher(
            PoseStamped,
            '/board_test_pose',
            10
        )

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info("Red block marker clustering initialized.")
        self.get_logger().info(
            f"Board defaults: {self.board_rows}x{self.board_cols}, "
            f"target=({self.target_row:.1f},{self.target_col:.1f}), "
            f"cell={self.CELL_SIZE:.5f} m"
        )

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

    def set_board_from_corner_markers(self, corner_clusters):
        centroids = np.array([
            np.mean(cluster, axis=0)
            for cluster in corner_clusters
        ])

        if len(centroids) < 4:
            return False

        xy_all = centroids[:, :2]
        center = np.mean(xy_all, axis=0)

        # If extra 1x1 red blocks are visible, use the four outermost markers.
        distances = np.linalg.norm(xy_all - center, axis=1)
        corner_indices = np.argsort(distances)[-4:]
        corners = centroids[corner_indices]

        best = None
        best_score = -np.inf

        for origin_idx in range(4):
            origin_xy = corners[origin_idx, :2]
            others = [i for i in range(4) if i != origin_idx]
            vectors = [(i, corners[i, :2] - origin_xy) for i in others]
            diagonal_idx, _ = max(
                vectors,
                key=lambda item: np.linalg.norm(item[1])
            )
            adjacent = [(i, v) for i, v in vectors if i != diagonal_idx]

            if len(adjacent) != 2:
                continue

            _, first_vec = adjacent[0]
            _, second_vec = adjacent[1]

            first_len = np.linalg.norm(first_vec)
            second_len = np.linalg.norm(second_vec)

            if first_len == 0.0 or second_len == 0.0:
                continue

            first_dir = first_vec / first_len
            second_dir = second_vec / second_len

            first_as_row = (
                np.dot(first_dir, self.row_dir)
                + np.dot(second_dir, self.col_dir)
            )
            second_as_row = (
                np.dot(second_dir, self.row_dir)
                + np.dot(first_dir, self.col_dir)
            )

            if first_as_row >= second_as_row:
                row_vec = first_vec
                col_vec = second_vec
            else:
                row_vec = second_vec
                col_vec = first_vec

            row_len = np.linalg.norm(row_vec)
            col_len = np.linalg.norm(col_vec)

            if row_len == 0.0 or col_len == 0.0:
                continue

            row_dir = row_vec / row_len
            col_dir = col_vec / col_len
            score = (
                np.dot(row_dir, self.row_dir)
                + np.dot(col_dir, self.col_dir)
            )

            if score > best_score:
                best_score = score
                best = origin_xy, row_dir, col_dir, row_len, col_len

        if best is None:
            self.get_logger().warn(
                "Could not infer board frame from corner markers"
            )
            return False

        origin_xy, row_dir, col_dir, row_len, col_len = best

        self.board_origin = origin_xy
        self.board_z = float(np.mean(corners[:, 2]))
        self.row_dir = row_dir
        self.col_dir = col_dir
        self.row_step = row_len / self.corner_span_rows
        self.col_step = col_len / self.corner_span_cols

        self.get_logger().info("===== BOARD FRAME SET FROM 4 CORNERS =====")
        self.get_logger().info(
            f"Origin P00: x={self.board_origin[0]:.4f}, "
            f"y={self.board_origin[1]:.4f}"
        )
        self.get_logger().info(f"Board z: {self.board_z:.4f}")
        self.get_logger().info(
            f"Row dir: [{self.row_dir[0]:.4f}, {self.row_dir[1]:.4f}], "
            f"step={self.row_step:.4f}"
        )
        self.get_logger().info(
            f"Col dir: [{self.col_dir[0]:.4f}, {self.col_dir[1]:.4f}], "
            f"step={self.col_step:.4f}"
        )
        self.get_logger().info("============================")

        # DEBUG: check grid directions
        debug_cells = [
            (0, 0),
            (self.corner_span_rows, 0),
            (0, self.corner_span_cols),
            (self.corner_span_rows, self.corner_span_cols)
        ]
        for r, c in debug_cells:
            wx, wy, _ = self.board_to_world(r, c)
            self.get_logger().info(f"({r},{c}) maps to: ({wx:.3f}, {wy:.3f})")

        return True

    def board_to_world(self, row, col):
        if self.board_origin is None:
            return None

        xy = (
            self.board_origin
            + row * self.row_step * self.row_dir
            + col * self.col_step * self.col_dir
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

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                Time()
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF error: {ex}",
                throttle_duration_sec=2.0
            )
            return

        cloud = do_transform_cloud(msg, tf)

        raw = pc2.read_points(
            cloud,
            field_names=('x', 'y', 'z', 'rgb'),
            skip_nans=True
        )

        xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(
            np.float32,
            copy=False
        )

        rgb = raw['rgb'].view(np.uint32)
        r = (rgb >> 16) & 255
        g = (rgb >> 8) & 255
        b = rgb & 255

        red_mask = (r > 150) & (g < 100) & (b < 100)
        pts = xyz[red_mask]

        if len(pts) < self.cluster_min_pts:
            return

        clusters = self.euclidean_clustering(pts)

        one_by_one_clusters = [
            cluster for cluster in clusters
            if self.estimate_shape(cluster) == "1x1"
        ]

        # 1. PASS ONE: Use four 1x1 red corners to establish the board frame
        if self.board_origin is None:
            if len(one_by_one_clusters) >= 4:
                self.get_logger().info(
                    "Found at least four 1x1 markers, setting board frame"
                )
                self.set_board_from_corner_markers(one_by_one_clusters)
            else:
                self.get_logger().info(
                    "Waiting for 4 corner markers, currently seeing "
                    f"{len(one_by_one_clusters)}",
                    throttle_duration_sec=2.0
                )

        # 2. PASS TWO: Process pickable blocks after board calibration
        if self.board_origin is not None:
            # Publish board pose before cube poses so planning has the target
            # before the first pick callback fires.
            self.publish_board_test_pose(self.target_row, self.target_col)

            for cluster in clusters:
                shape = self.estimate_shape(cluster)

                # Do not publish the origin marker as a pickable block
                if shape == "1x1":
                    continue

                self.process_block(cluster, cloud.header)

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
        pose.pose.position.z = float(centroid[2]) - 0.005

        half = yaw / 2.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        self.pose_pub.publish(pose)

        msg = String()
        msg.data = (
            f"{shape}|x:{centroid[0]:.3f},"
            f"y:{centroid[1]:.3f},z:{centroid[2]:.3f}"
        )
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
