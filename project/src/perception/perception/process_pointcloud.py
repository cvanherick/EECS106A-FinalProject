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
        self.pointcloud_topic = self.declare_parameter(
            'pointcloud_topic',
            '/camera/camera/depth/color/points'
        ).value

        self.cluster_dist_thresh = 0.003
        self.cluster_min_pts = 50
        self.block_unit = 0.03175
        self.block_pick_z_offset = float(
            self.declare_parameter('block_pick_z_offset', -0.015).value
        )
        self.pick_x_offset = float(
            self.declare_parameter('pick_x_offset', 0.0).value
        )
        self.pick_y_offset = float(
            self.declare_parameter('pick_y_offset', 0.0).value
        )
        self.place_x_offset = float(
            self.declare_parameter('place_x_offset', 0.0).value
        )
        self.place_y_offset = float(
            self.declare_parameter('place_y_offset', 0.0).value
        )
        self.use_measured_grid = bool(
            self.declare_parameter('use_measured_grid', False).value
        )
        self.use_oriented_box_center = bool(
            self.declare_parameter('use_oriented_box_center', False).value
        )

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
        self.playable_row_offset = int(
            self.declare_parameter('playable_row_offset', 2).value
        )
        self.playable_col_offset = int(
            self.declare_parameter('playable_col_offset', 0).value
        )
        self.playable_rows = max(
            1,
            int(
                self.declare_parameter(
                    'playable_rows',
                    self.board_rows - 2 * self.playable_row_offset
                ).value
            )
        )
        self.playable_cols = max(
            1,
            int(
                self.declare_parameter(
                    'playable_cols',
                    self.board_cols - 2 * self.playable_col_offset
                ).value
            )
        )

        self.board_origin = None
        self.board_center = None
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
        default_place_row = float(
            self.declare_parameter('target_row', 5.0).value
        )
        default_place_col = float(
            self.declare_parameter('target_col', 5.0).value
        )
        self.place_row = float(
            self.declare_parameter('place_row', default_place_row).value
        )
        self.place_col = float(
            self.declare_parameter('place_col', default_place_col).value
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
            self.pointcloud_topic,
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
            f"Listening for point cloud on {self.pointcloud_topic}; "
            f"target frame is {self.target_frame}"
        )
        self.get_logger().info(
            f"Physical board: {self.board_rows}x{self.board_cols}; "
            f"game board: {self.playable_rows}x{self.playable_cols} "
            f"offset by ({self.playable_row_offset},{self.playable_col_offset}); "
            "red 1x1 corner blocks are calibration markers; "
            f"place=({self.place_row:.1f},{self.place_col:.1f}), "
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

        centered = corners[:, :2] - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_a = vh[0]
        axis_b = vh[1]

        span_a = np.ptp(centered @ axis_a)
        span_b = np.ptp(centered @ axis_b)

        if span_a == 0.0 or span_b == 0.0:
            self.get_logger().warn(
                "Could not infer board frame from corner markers"
            )
            return False

        if span_a >= span_b:
            long_dir = axis_a
            short_dir = axis_b
            long_span = span_a
            short_span = span_b
        else:
            long_dir = axis_b
            short_dir = axis_a
            long_span = span_b
            short_span = span_a

        row_dir = long_dir
        col_dir = short_dir

        if np.dot(row_dir, self.row_dir) < 0.0:
            row_dir = -row_dir

        if np.dot(col_dir, self.col_dir) < 0.0:
            col_dir = -col_dir

        row_step = self.CELL_SIZE
        col_step = self.CELL_SIZE
        if self.use_measured_grid:
            row_step = long_span / float(self.corner_span_rows)
            col_step = short_span / float(self.corner_span_cols)

        self.board_center = center
        self.board_origin = (
            center
            - (self.corner_span_rows / 2.0) * row_step * row_dir
            - (self.corner_span_cols / 2.0) * col_step * col_dir
        )

        self.board_z = float(np.mean(corners[:, 2]))
        self.row_dir = row_dir
        self.col_dir = col_dir
        self.row_step = row_step
        self.col_step = col_step

        self.get_logger().info("===== BOARD FRAME SET FROM 4 CORNERS =====")
        self.get_logger().info(
            f"Origin P00: x={self.board_origin[0]:.4f}, "
            f"y={self.board_origin[1]:.4f}"
        )
        self.get_logger().info(
            f"Center: x={self.board_center[0]:.4f}, "
            f"y={self.board_center[1]:.4f}"
        )
        self.get_logger().info(f"Board z: {self.board_z:.4f}")
        self.get_logger().info(
            f"Observed spans: long={long_span:.4f}, short={short_span:.4f}"
        )
        self.get_logger().info(
            f"Measured steps: row={self.row_step:.5f}, "
            f"col={self.col_step:.5f}; nominal={self.CELL_SIZE:.5f}"
        )
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
            wx, wy, _ = self.physical_board_to_world(r, c)
            self.get_logger().info(
                f"physical ({r},{c}) maps to: ({wx:.3f}, {wy:.3f})"
            )

        return True

    def board_to_world(self, row, col):
        if self.board_origin is None:
            return None

        if not self.is_valid_divot(row, col):
            self.get_logger().warn(
                f"Requested divot ({row:.2f},{col:.2f}) is outside "
                f"0-{self.playable_rows - 1} playable rows and "
                f"0-{self.playable_cols - 1} playable cols",
                throttle_duration_sec=2.0
            )
            return None

        physical_row = row + self.playable_row_offset
        physical_col = col + self.playable_col_offset

        return self.physical_board_to_world(physical_row, physical_col)

    def physical_board_to_world(self, row, col):
        xy = (
            self.board_origin
            + row * self.row_step * self.row_dir
            + col * self.col_step * self.col_dir
        )
        xy = xy + np.array([self.place_x_offset, self.place_y_offset])
        return float(xy[0]), float(xy[1]), float(self.board_z)


    def estimate_oriented_box(self, pts):
        xy = pts[:, :2]
        mean_xy = np.mean(xy, axis=0)
        xy_centered = xy - mean_xy

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        aligned = xy_centered @ eigvecs
        min_xy = np.min(aligned, axis=0)
        max_xy = np.max(aligned, axis=0)
        lengths = max_xy - min_xy

        center_aligned = (min_xy + max_xy) / 2.0
        center_xy = mean_xy + center_aligned @ eigvecs.T
        principal = eigvecs[:, 0]
        yaw = np.arctan2(principal[1], principal[0])

        return center_xy, yaw, lengths

    def is_valid_divot(self, row, col):
        return (
            0.0 <= row <= float(self.playable_rows - 1)
            and 0.0 <= col <= float(self.playable_cols - 1)
        )

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

        board_yaw = np.arctan2(self.row_dir[1], self.row_dir[0])
        half = board_yaw / 2.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        self.board_test_pose_pub.publish(pose)
        self.get_logger().info(
            f"Published place divot ({row:.2f},{col:.2f}) -> "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={board_yaw:.3f}",
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
            self.get_logger().warn(
                f"TF error: {ex}",
                throttle_duration_sec=2.0
            )
            return

        cloud = do_transform_cloud(msg, tf)

        cloud_fields = {field.name for field in cloud.fields}
        if 'rgb' in cloud_fields:
            color_field = 'rgb'
        elif 'rgba' in cloud_fields:
            color_field = 'rgba'
        else:
            self.get_logger().warn(
                f"Point cloud has no rgb/rgba field. Fields: {sorted(cloud_fields)}",
                throttle_duration_sec=2.0
            )
            return

        try:
            raw = pc2.read_points(
                cloud,
                field_names=('x', 'y', 'z', color_field),
                skip_nans=True
            )
        except Exception as ex:
            self.get_logger().warn(
                f"Could not read point cloud fields: {ex}",
                throttle_duration_sec=2.0
            )
            return

        xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(
            np.float32,
            copy=False
        )

        rgb = raw[color_field].view(np.uint32)
        r = (rgb >> 16) & 255
        g = (rgb >> 8) & 255
        b = rgb & 255

        red_mask = (r > 150) & (g < 100) & (b < 100)
        pts = xyz[red_mask]

        if len(pts) < self.cluster_min_pts:
            self.get_logger().info(
                f"Point cloud received from {msg.header.frame_id}, but only "
                f"{len(pts)} red points passed threshold",
                throttle_duration_sec=2.0
            )
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
            self.publish_board_test_pose(self.place_row, self.place_col)

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
        center_xy, yaw, lengths = self.estimate_oriented_box(clean)
        pick_xy = centroid[:2]
        if self.use_oriented_box_center:
            pick_xy = center_xy

        long_side = max(lengths)
        short_side = min(lengths)

        n_long = max(1, int(round(long_side / self.block_unit)))
        n_short = max(1, int(round(short_side / self.block_unit)))

        shape = f"{n_long}x{n_short}"

        self.get_logger().info(f"Block detected: {shape}")

        pose = PoseStamped()
        pose.header = header

        pose.pose.position.x = float(pick_xy[0] + self.pick_x_offset)
        pose.pose.position.y = float(pick_xy[1] + self.pick_y_offset)
        pose.pose.position.z = float(centroid[2] + self.block_pick_z_offset)

        half = yaw / 2.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        self.pose_pub.publish(pose)
        self.get_logger().info(
            "Published pick pose: "
            f"x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"z={pose.pose.position.z:.3f}; "
            f"centroid=({centroid[0]:.3f},{centroid[1]:.3f}), "
            f"box=({center_xy[0]:.3f},{center_xy[1]:.3f})",
            throttle_duration_sec=1.0
        )

        msg = String()
        msg.data = (
            f"{shape}|x:{pose.pose.position.x:.3f},"
            f"y:{pose.pose.position.y:.3f},z:{pose.pose.position.z:.3f}"
        )
        self.block_info_pub.publish(msg)

    def _on_parameter_update(self, params):
        for param in params:
            if param.name == 'place_row':
                self.place_row = float(param.value)
            elif param.name == 'place_col':
                self.place_col = float(param.value)
            elif param.name == 'target_row':
                self.place_row = float(param.value)
            elif param.name == 'target_col':
                self.place_col = float(param.value)
            elif param.name == 'block_pick_z_offset':
                self.block_pick_z_offset = float(param.value)
            elif param.name == 'pick_x_offset':
                self.pick_x_offset = float(param.value)
            elif param.name == 'pick_y_offset':
                self.pick_y_offset = float(param.value)
            elif param.name == 'place_x_offset':
                self.place_x_offset = float(param.value)
            elif param.name == 'place_y_offset':
                self.place_y_offset = float(param.value)
            elif param.name == 'use_measured_grid':
                self.use_measured_grid = bool(param.value)
            elif param.name == 'use_oriented_box_center':
                self.use_oriented_box_center = bool(param.value)

        self.get_logger().info(
            f"Place divot set to ({self.place_row:.2f},{self.place_col:.2f})",
            throttle_duration_sec=1.0
        )
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
