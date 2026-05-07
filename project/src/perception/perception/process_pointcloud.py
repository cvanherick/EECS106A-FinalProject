import rclpy
from itertools import combinations
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
from visualization_msgs.msg import Marker, MarkerArray


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
        self.place_along_piece_offset = float(
            self.declare_parameter('place_along_piece_offset', 0.0).value
        )
        self.place_across_piece_offset = float(
            self.declare_parameter('place_across_piece_offset', 0.0).value
        )
        self.place_z_offset = float(
            self.declare_parameter('place_z_offset', 0.0).value
        )
        self.use_measured_grid = bool(
            self.declare_parameter('use_measured_grid', False).value
        )
        self.use_oriented_box_center = bool(
            self.declare_parameter('use_oriented_box_center', False).value
        )
        self.grid_step_warn_ratio = float(
            self.declare_parameter('grid_step_warn_ratio', 0.15).value
        )

        # NOTE: If the robot is overshooting, measure the physical
        # distance between two peg centers and update this number!
        self.CELL_SIZE = 0.03175
        self.expected_board_long_span = (12 - 1) * self.CELL_SIZE
        self.expected_board_short_span = (10 - 1) * self.CELL_SIZE

        self.board_rows = max(
            1,
            int(self.declare_parameter('board_rows', 10).value)
        )
        self.board_cols = max(
            1,
            int(self.declare_parameter('board_cols', 12).value)
        )
        self.playable_row_offset = int(
            self.declare_parameter('playable_row_offset', 1).value
        )
        self.playable_col_offset = int(
            self.declare_parameter('playable_col_offset', 1).value
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
        self.invert_playable_rows = bool(
            self.declare_parameter('invert_playable_rows', False).value
        )
        self.invert_playable_cols = bool(
            self.declare_parameter('invert_playable_cols', True).value
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
        self.expected_board_long_span = (
            max(self.corner_span_rows, self.corner_span_cols) * self.CELL_SIZE
        )
        self.expected_board_short_span = (
            min(self.corner_span_rows, self.corner_span_cols) * self.CELL_SIZE
        )
        self.min_corner_span_ratio = float(
            self.declare_parameter('min_corner_span_ratio', 0.65).value
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
        self.target_is_set = bool(
            self.declare_parameter('target_is_set', False).value
        )
        self.robot_target_cells = self.parse_target_cells(
            self.declare_parameter('robot_target_cells', '').value
        )
        self.expected_robot_shape = str(
            self.declare_parameter('expected_robot_shape', '').value
        )
        self.piece_yaw_along_col = bool(
            self.declare_parameter('piece_yaw_along_col', False).value
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
            '/cube_pose_blue',
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
        self.board_divot_marker_pub = self.create_publisher(
            MarkerArray,
            '/board_divot_markers',
            10
        )
        self.board_marker_timer = self.create_timer(
            1.0,
            self.publish_board_divot_markers
        )

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info(
            "Color clustering initialized: red=board markers, "
            "blue=robot pickup blocks."
        )
        self.get_logger().info(
            f"Listening for point cloud on {self.pointcloud_topic}; "
            f"target frame is {self.target_frame}"
        )
        self.get_logger().info(
            f"Physical board: {self.board_rows}x{self.board_cols}; "
            f"game board: {self.playable_rows}x{self.playable_cols} "
            f"offset by ({self.playable_row_offset},{self.playable_col_offset}); "
            f"invert_rows={self.invert_playable_rows}, "
            f"invert_cols={self.invert_playable_cols}; "
            "red 1x1 corner blocks are calibration markers; "
            f"target_set={self.target_is_set}, "
            f"place=({self.place_row:.1f},{self.place_col:.1f}), "
            "piece_frame_offsets="
            f"({self.place_along_piece_offset:.4f},"
            f"{self.place_across_piece_offset:.4f},"
            f"{self.place_z_offset:.4f}), "
            f"cell={self.CELL_SIZE:.5f} m"
        )
        self.get_logger().info(
            "Expected board marker span: "
            f"long={self.expected_board_long_span:.4f} m "
            f"({self.expected_board_long_span / 0.0254:.2f} in), "
            f"short={self.expected_board_short_span:.4f} m "
            f"({self.expected_board_short_span / 0.0254:.2f} in)"
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

        corners, long_span, short_span, long_dir, short_dir = (
            self.select_board_corner_markers(centroids)
        )

        if corners is None:
            self.get_logger().warn(
                "Saw 1x1 red clusters, but none matched the expected "
                "10x12 board corner span. Move non-corner red pieces away "
                "from the board markers.",
                throttle_duration_sec=2.0
            )
            return False

        if self.corner_span_rows >= self.corner_span_cols:
            row_dir = long_dir
            col_dir = short_dir
            row_span = long_span
            col_span = short_span
        else:
            row_dir = short_dir
            col_dir = long_dir
            row_span = short_span
            col_span = long_span

        # if np.dot(row_dir, self.row_dir) < 0.0:
            # row_dir = -row_dir

        # if np.dot(col_dir, self.col_dir) < 0.0:
            # col_dir = -col_dir

        measured_row_step = row_span / float(self.corner_span_rows)
        measured_col_step = col_span / float(self.corner_span_cols)

        for axis_name, measured_step in (
            ('row', measured_row_step),
            ('col', measured_col_step)
        ):
            relative_error = abs(measured_step - self.CELL_SIZE) / self.CELL_SIZE
            if relative_error > self.grid_step_warn_ratio:
                self.get_logger().warn(
                    f"Measured {axis_name} cell size {measured_step:.5f} m "
                    f"differs from nominal {self.CELL_SIZE:.5f} m by "
                    f"{relative_error * 100.0:.1f}%. Board calibration may "
                    "be inaccurate.",
                    throttle_duration_sec=2.0
                )

        row_step = self.CELL_SIZE
        col_step = self.CELL_SIZE
        if self.use_measured_grid:
            row_step = measured_row_step
            col_step = measured_col_step

        center = np.mean(corners[:, :2], axis=0)
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

    def select_board_corner_markers(self, centroids):
        best = None
        min_long_span = self.expected_board_long_span * self.min_corner_span_ratio
        min_short_span = self.expected_board_short_span * self.min_corner_span_ratio

        for candidate in combinations(centroids, 4):
            corners = np.array(candidate)
            centered = corners[:, :2] - np.mean(corners[:, :2], axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis_a = vh[0]
            axis_b = vh[1]

            span_a = np.ptp(centered @ axis_a)
            span_b = np.ptp(centered @ axis_b)

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

            if long_span < min_long_span or short_span < min_short_span:
                continue

            span_score = (
                abs(long_span - self.expected_board_long_span)
                + abs(short_span - self.expected_board_short_span)
            )
            aspect_score = abs(
                (long_span / short_span)
                - (self.expected_board_long_span / self.expected_board_short_span)
            )
            score = span_score + 0.1 * aspect_score

            if best is None or score < best[0]:
                best = (score, corners, long_span, short_span, long_dir, short_dir)

        if best is None:
            xy = centroids[:, :2]
            self.get_logger().warn(
                "Rejected board marker candidates. "
                f"Expected at least long={min_long_span:.3f} m and "
                f"short={min_short_span:.3f} m. "
                f"Detected 1x1 centroids: {np.round(xy, 3).tolist()}",
                throttle_duration_sec=2.0
            )
            return None, None, None, None, None

        _, corners, long_span, short_span, long_dir, short_dir = best
        return corners, long_span, short_span, long_dir, short_dir

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

        physical_row, physical_col = self.game_to_physical_cell(row, col)

        return self.physical_board_to_world(physical_row, physical_col)

    def game_to_physical_cell(self, row, col):
        game_row = row
        game_col = col

        if self.invert_playable_rows:
            game_row = (self.playable_rows - 1) - game_row

        if self.invert_playable_cols:
            game_col = (self.playable_cols - 1) - game_col

        physical_row = game_row + self.playable_row_offset
        physical_col = game_col + self.playable_col_offset
        return physical_row, physical_col

    def physical_board_to_world(self, row, col):
        xy = (
            self.board_origin
            + row * self.row_step * self.row_dir
            + col * self.col_step * self.col_dir
        )
        xy = xy + np.array([self.place_x_offset, self.place_y_offset])
        return float(xy[0]), float(xy[1]), float(self.board_z)

    def world_to_physical_board_cell(self, xy):
        if self.board_origin is None:
            return None

        corrected_xy = (
            np.asarray(xy, dtype=float)
            - np.array([self.place_x_offset, self.place_y_offset])
        )
        delta = corrected_xy - self.board_origin
        row = float(np.dot(delta, self.row_dir) / self.row_step)
        col = float(np.dot(delta, self.col_dir) / self.col_step)
        return row, col

    def is_xy_on_physical_board(self, xy, margin_cells=0.75):
        cell = self.world_to_physical_board_cell(xy)
        if cell is None:
            return False

        row, col = cell
        return (
            -margin_cells
            <= row
            <= (self.board_rows - 1) + margin_cells
            and -margin_cells
            <= col
            <= (self.board_cols - 1) + margin_cells
        )

    def is_playable_physical_cell(self, row, col):
        return (
            self.playable_row_offset
            <= row
            < self.playable_row_offset + self.playable_rows
            and self.playable_col_offset
            <= col
            < self.playable_col_offset + self.playable_cols
        )

    def is_corner_physical_cell(self, row, col):
        return (
            row in (0, self.corner_span_rows)
            and col in (0, self.corner_span_cols)
        )

    def set_marker_color(self, marker, rgba):
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

    def parse_target_cells(self, text):
        cells = []
        for item in str(text).split(';'):
            item = item.strip()
            if not item:
                continue

            parts = item.split(',')
            if len(parts) != 2:
                self.get_logger().warn(
                    f"Ignoring malformed robot target cell '{item}'",
                    throttle_duration_sec=2.0
                )
                continue

            try:
                cells.append((float(parts[0]), float(parts[1])))
            except ValueError:
                self.get_logger().warn(
                    f"Ignoring malformed robot target cell '{item}'",
                    throttle_duration_sec=2.0
                )

        return cells

    def normalized_shape_key(self, shape):
        try:
            a, b = [int(part) for part in str(shape).lower().split('x')]
        except ValueError:
            return None

        return tuple(sorted((a, b)))

    def shape_matches_expected(self, shape):
        if not self.expected_robot_shape:
            return True

        detected = self.normalized_shape_key(shape)
        expected = self.normalized_shape_key(self.expected_robot_shape)
        return detected is not None and detected == expected

    def publish_board_divot_markers(self):
        if self.board_origin is None:
            return

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for row in range(self.board_rows):
            for col in range(self.board_cols):
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = stamp
                marker.ns = 'board_divots'
                marker.id = row * self.board_cols + col
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                x, y, z = self.physical_board_to_world(row, col)
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z + 0.006
                marker.pose.orientation.w = 1.0

                marker.scale.x = 0.010
                marker.scale.y = 0.010
                marker.scale.z = 0.010

                if self.is_corner_physical_cell(row, col):
                    self.set_marker_color(marker, (1.0, 0.05, 0.05, 1.0))
                    marker.scale.x = 0.018
                    marker.scale.y = 0.018
                    marker.scale.z = 0.018
                elif self.is_playable_physical_cell(row, col):
                    self.set_marker_color(marker, (0.1, 0.45, 1.0, 0.9))
                else:
                    self.set_marker_color(marker, (0.65, 0.65, 0.65, 0.45))

                marker_array.markers.append(marker)

        if self.target_is_set and self.is_valid_divot(self.place_row, self.place_col):
            physical_row, physical_col = self.game_to_physical_cell(
                self.place_row,
                self.place_col
            )
            target = Marker()
            target.header.frame_id = self.target_frame
            target.header.stamp = stamp
            target.ns = 'board_divots'
            target.id = self.board_rows * self.board_cols
            target.type = Marker.SPHERE
            target.action = Marker.ADD

            x, y, z = self.physical_board_to_world(physical_row, physical_col)
            target.pose.position.x = x
            target.pose.position.y = y
            target.pose.position.z = z + 0.018
            target.pose.orientation.w = 1.0
            target.scale.x = 0.026
            target.scale.y = 0.026
            target.scale.z = 0.026
            self.set_marker_color(target, (1.0, 0.85, 0.0, 1.0))
            marker_array.markers.append(target)

            old_adjusted = Marker()
            old_adjusted.header.frame_id = self.target_frame
            old_adjusted.header.stamp = stamp
            old_adjusted.ns = 'place_adjusted_target'
            old_adjusted.id = 0
            old_adjusted.action = Marker.DELETE
            marker_array.markers.append(old_adjusted)

            has_piece_frame_offset = any(
                abs(value) > 1e-6
                for value in (
                    self.place_along_piece_offset,
                    self.place_across_piece_offset,
                    self.place_z_offset
                )
            )
            if has_piece_frame_offset:
                adjusted = Marker()
                adjusted.header.frame_id = self.target_frame
                adjusted.header.stamp = stamp
                adjusted.ns = 'place_adjusted_target'
                adjusted.id = 0
                adjusted.type = Marker.SPHERE
                adjusted.action = Marker.ADD

                long_axis = self.get_piece_long_axis()
                ax, ay, az = self.apply_piece_frame_place_offsets(
                    x,
                    y,
                    z,
                    long_axis
                )
                adjusted.pose.position.x = ax
                adjusted.pose.position.y = ay
                adjusted.pose.position.z = az + 0.034
                adjusted.pose.orientation.w = 1.0
                adjusted.scale.x = 0.018
                adjusted.scale.y = 0.018
                adjusted.scale.z = 0.018
                self.set_marker_color(adjusted, (1.0, 0.0, 1.0, 1.0))
                marker_array.markers.append(adjusted)

        for marker_id in range(10):
            old_target = Marker()
            old_target.header.frame_id = self.target_frame
            old_target.header.stamp = stamp
            old_target.ns = 'robot_piece_target'
            old_target.id = marker_id
            old_target.action = Marker.DELETE
            marker_array.markers.append(old_target)

        if self.robot_target_cells:
            valid_cells = [
                c for c in self.robot_target_cells if self.is_valid_divot(*c)
            ]
            if valid_cells:
                # Compute the centroid and span of the full piece in world space.
                phys = [self.game_to_physical_cell(r, c) for r, c in valid_cells]
                world_pts = np.array([
                    self.physical_board_to_world(pr, pc)[:2] for pr, pc in phys
                ])
                cx, cy = world_pts.mean(axis=0)

                # Span along each board axis (number of cells x step).
                rows = [r for r, _ in valid_cells]
                cols = [c for _, c in valid_cells]
                n_rows = max(1, max(rows) - min(rows) + 1)
                n_cols = max(1, max(cols) - min(cols) + 1)

                span_along_row_dir = n_rows * self.row_step
                span_along_col_dir = n_cols * self.col_step

                # Marker local X is aligned to the piece long axis by yaw, so
                # scale.x must be the long-axis span, not world/base X span.
                long_axis = self.get_piece_long_axis()
                if self.piece_yaw_along_col:
                    marker_long_span = span_along_row_dir
                    marker_short_span = span_along_col_dir
                else:
                    marker_long_span = span_along_col_dir
                    marker_short_span = span_along_row_dir

                piece_yaw = np.arctan2(long_axis[1], long_axis[0])
                half_yaw = piece_yaw / 2.0

                piece_marker = Marker()
                piece_marker.header.frame_id = self.target_frame
                piece_marker.header.stamp = stamp
                piece_marker.ns = 'robot_piece_target'
                piece_marker.id = 0
                piece_marker.type = Marker.CUBE
                piece_marker.action = Marker.ADD

                piece_marker.pose.position.x = float(cx)
                piece_marker.pose.position.y = float(cy)
                piece_marker.pose.position.z = float(self.board_z) + 0.030
                piece_marker.pose.orientation.x = 0.0
                piece_marker.pose.orientation.y = 0.0
                piece_marker.pose.orientation.z = float(np.sin(half_yaw))
                piece_marker.pose.orientation.w = float(np.cos(half_yaw))
                piece_marker.scale.x = float(marker_long_span * 0.72)
                piece_marker.scale.y = float(marker_short_span * 0.72)
                piece_marker.scale.z = 0.010
                self.set_marker_color(piece_marker, (1.0, 0.28, 0.05, 0.85))
                marker_array.markers.append(piece_marker)

        self.board_divot_marker_pub.publish(marker_array)


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

    def get_piece_long_axis(self):
        # piece_yaw_along_col=True means the piece spans multiple rows (same
        # col), so its long axis is row_dir. Otherwise it spans multiple cols
        # and its long axis is col_dir.
        if self.piece_yaw_along_col:
            return self.row_dir

        return self.col_dir

    def apply_piece_frame_place_offsets(self, x, y, z, long_axis):
        across_axis = np.array([-long_axis[1], long_axis[0]])
        xy = (
            np.array([x, y])
            + self.place_along_piece_offset * long_axis
            + self.place_across_piece_offset * across_axis
        )
        return float(xy[0]), float(xy[1]), float(z + self.place_z_offset)

    def publish_board_test_pose(self, row=5, col=5):
        if not self.target_is_set:
            self.get_logger().info(
                "Board frame is calibrated, but no placement target has "
                "been set yet. Waiting for game_manager.",
                throttle_duration_sec=2.0
            )
            return

        result = self.board_to_world(row, col)

        if result is None:
            return

        x, y, z = result
        physical_row, physical_col = self.game_to_physical_cell(row, col)
        raw_x, raw_y, raw_z = x, y, z
        long_axis = self.get_piece_long_axis()
        x, y, z = self.apply_piece_frame_place_offsets(x, y, z, long_axis)

        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        board_yaw = np.arctan2(long_axis[1], long_axis[0])
        half = board_yaw / 2.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(np.sin(half))
        pose.pose.orientation.w = float(np.cos(half))

        self.board_test_pose_pub.publish(pose)
        self.get_logger().info(
            f"Published place divot ({row:.2f},{col:.2f}) -> "
            f"physical ({physical_row:.2f},{physical_col:.2f}) -> "
            f"raw=({raw_x:.3f},{raw_y:.3f},{raw_z:.3f}), "
            f"adjusted=({x:.3f},{y:.3f},{z:.3f}), "
            f"yaw={board_yaw:.3f}",
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

        red_mask = (r > 150) & (g < 110) & (b < 110)
        blue_mask = (b > 120) & (r < 120) & (g < 170)

        red_pts = xyz[red_mask]
        # Only keep points near the board region
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = -0.50, 1.0
        board_region_mask = (
            (red_pts[:, 0] > x_min) &
            (red_pts[:, 0] < x_max) &
            (red_pts[:, 1] > y_min) &
            (red_pts[:, 1] < y_max) &
            (red_pts[:, 2] > z_min) &
            (red_pts[:, 2] < z_max)
        )
        red_pts = red_pts[board_region_mask]

        blue_pts = xyz[blue_mask]

        red_clusters = []
        blue_clusters = []

        if len(red_pts) >= self.cluster_min_pts:
            red_clusters = self.euclidean_clustering(red_pts)
        elif self.board_origin is None:
            self.get_logger().info(
                f"Point cloud received from {msg.header.frame_id}, but only "
                f"{len(red_pts)} red marker points passed threshold",
                throttle_duration_sec=2.0
            )

        if len(blue_pts) >= self.cluster_min_pts:
            blue_clusters = self.euclidean_clustering(blue_pts)
        elif self.board_origin is not None:
            self.get_logger().info(
                f"Board is calibrated, but only {len(blue_pts)} blue robot "
                "piece points passed threshold",
                throttle_duration_sec=2.0
            )

        one_by_one_clusters = [
            cluster for cluster in red_clusters
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
            # before the first pick callback fires. This intentionally waits
            # for game_manager to set a real target instead of using a default.
            self.publish_board_test_pose(self.place_row, self.place_col)

            if not self.target_is_set or not self.expected_robot_shape:
                self.get_logger().info(
                    "Board is calibrated, but no robot target/shape has been "
                    "set yet. Not publishing blue pickup poses.",
                    throttle_duration_sec=2.0
                )
                return

            valid_blue_clusters = []
            for cluster in blue_clusters:
                centroid = np.mean(cluster, axis=0)
                if self.is_xy_on_physical_board(centroid[:2]):
                    row_col = self.world_to_physical_board_cell(centroid[:2])
                    self.get_logger().info(
                        "Skipping blue block already on board at physical "
                        f"cell ({row_col[0]:.2f},{row_col[1]:.2f})",
                        throttle_duration_sec=1.0
                    )
                    continue

                shape = self.estimate_shape(cluster)
                if not self.shape_matches_expected(shape):
                    self.get_logger().info(
                        f"Skipping blue {shape}; waiting for "
                        f"{self.expected_robot_shape}",
                        throttle_duration_sec=1.0
                    )
                    continue

                valid_blue_clusters.append((shape, cluster))

            if len(valid_blue_clusters) == 0:
                self.get_logger().info(
                    f"No off-board blue {self.expected_robot_shape} block "
                    "available for pickup.",
                    throttle_duration_sec=2.0
                )
                return

            if len(valid_blue_clusters) > 1:
                shapes = [shape for shape, _ in valid_blue_clusters]
                self.get_logger().warn(
                    "Multiple matching off-board blue robot blocks are "
                    f"visible ({shapes}); refusing to guess pickup target.",
                    throttle_duration_sec=2.0
                )
                return

            _, cluster = valid_blue_clusters[0]
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

        self.get_logger().info(f"Blue robot block detected: {shape}")

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
            "Published blue pick pose: "
            f"x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"z={pose.pose.position.z:.3f}; "
            f"centroid=({centroid[0]:.3f},{centroid[1]:.3f}), "
            f"box=({center_xy[0]:.3f},{center_xy[1]:.3f})",
            throttle_duration_sec=1.0
        )

        msg = String()
        msg.data = (
            f"blue:{shape}|x:{pose.pose.position.x:.3f},"
            f"y:{pose.pose.position.y:.3f},z:{pose.pose.position.z:.3f}"
        )
        self.block_info_pub.publish(msg)

    def _on_parameter_update(self, params):
        for param in params:
            if param.name == 'place_row':
                self.place_row = float(param.value)
                self.target_is_set = True
            elif param.name == 'place_col':
                self.place_col = float(param.value)
                self.target_is_set = True
            elif param.name == 'target_row':
                self.place_row = float(param.value)
                self.target_is_set = True
            elif param.name == 'target_col':
                self.place_col = float(param.value)
                self.target_is_set = True
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
            elif param.name == 'place_along_piece_offset':
                self.place_along_piece_offset = float(param.value)
            elif param.name == 'place_across_piece_offset':
                self.place_across_piece_offset = float(param.value)
            elif param.name == 'place_z_offset':
                self.place_z_offset = float(param.value)
            elif param.name == 'use_measured_grid':
                self.use_measured_grid = bool(param.value)
            elif param.name == 'use_oriented_box_center':
                self.use_oriented_box_center = bool(param.value)
            elif param.name == 'grid_step_warn_ratio':
                self.grid_step_warn_ratio = float(param.value)
            elif param.name == 'invert_playable_rows':
                self.invert_playable_rows = bool(param.value)
            elif param.name == 'invert_playable_cols':
                self.invert_playable_cols = bool(param.value)
            elif param.name == 'robot_target_cells':
                self.robot_target_cells = self.parse_target_cells(param.value)
                self.target_is_set = True
            elif param.name == 'expected_robot_shape':
                self.expected_robot_shape = str(param.value)
            elif param.name == 'piece_yaw_along_col':
                self.piece_yaw_along_col = bool(param.value)
            elif param.name == 'target_is_set':
                self.target_is_set = bool(param.value)

        self.get_logger().info(
            f"Place divot set to ({self.place_row:.2f},{self.place_col:.2f}); "
            f"target_set={self.target_is_set}; "
            f"robot_cells={self.robot_target_cells}; "
            f"expected_shape={self.expected_robot_shape}; "
            "piece_frame_offsets="
            f"({self.place_along_piece_offset:.4f},"
            f"{self.place_across_piece_offset:.4f},"
            f"{self.place_z_offset:.4f}); "
            f"invert_rows={self.invert_playable_rows}, "
            f"invert_cols={self.invert_playable_cols}",
            throttle_duration_sec=1.0
        )
        if self.board_origin is not None and self.target_is_set:
            self.publish_board_test_pose(self.place_row, self.place_col)
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
