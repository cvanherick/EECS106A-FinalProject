import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
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
        
        # --- SPATIAL PARAMETERS (Bounding Box) ---
        self.target_frame = self.declare_parameter('camera_depth_optical_frame', 'base_link').value
        self.max_y = float(self.declare_parameter('max_y', 0.67).value)
        self.min_z = float(self.declare_parameter('min_z', -0.18).value)
        self.max_z = float(self.declare_parameter('max_z', -0.15).value)

        # --- COLOR PARAMETERS (Defaults set for a bright RED block) ---
        self.r_min = int(self.declare_parameter('r_min', 150).value)
        self.r_max = int(self.declare_parameter('r_max', 255).value)
        self.g_min = int(self.declare_parameter('g_min', 0).value)
        self.g_max = int(self.declare_parameter('g_max', 100).value)
        self.b_min = int(self.declare_parameter('b_min', 0).value)
        self.b_max = int(self.declare_parameter('b_max', 100).value)

        self.add_on_set_parameters_callback(self._on_parameter_update)

        # --- TF SETUP ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- SUBSCRIBERS & PUBLISHERS ---
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )
        self.cube_pose_pub = self.create_publisher(PoseStamped, '/cube_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info("Subscribed to PointCloud2. Color Filtering & PCA Pose publisher ready.")

    def pointcloud_callback(self, msg: PointCloud2):
        source_frame = msg.header.frame_id 
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, source_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {self.target_frame}: {ex}')
            return

        transformed_cloud = do_transform_cloud(msg, tf)

        # 1. READ POINTS (Including RGB)
        raw_points = pc2.read_points(
            transformed_cloud,
            field_names=('x', 'y', 'z', 'rgb'),
            skip_nans=True,
        )
        
        points_xyz = np.column_stack(
                (raw_points['x'], raw_points['y'], raw_points['z'])
            ).astype(np.float32, copy=False)

        # 2. UNPACK RGB
        rgb_floats = raw_points['rgb']
        rgb_bytes = rgb_floats.view(np.uint32)
        r = (rgb_bytes >> 16) & 255
        g = (rgb_bytes >> 8) & 255
        b = rgb_bytes & 255

        # 3. CREATE MASKS
        # Color Mask
        color_mask = (
            (r >= self.r_min) & (r <= self.r_max) &
            (g >= self.g_min) & (g <= self.g_max) &
            (b >= self.b_min) & (b <= self.b_max)
        )
        # Spatial Mask
        spatial_mask = (
            (points_xyz[:, 1] <= self.max_y) & 
            (points_xyz[:, 2] <= self.max_z) & 
            (points_xyz[:, 2] >= self.min_z)
        )

        # Combine and Filter
        filtered_points = points_xyz[spatial_mask & color_mask]

        if len(filtered_points) < 50:
            # Silently return if no target block is found to prevent log spam
            return

        # 4. STATISTICAL OUTLIER REJECTION
        mean_pt = np.mean(filtered_points, axis=0)
        std_pt = np.std(filtered_points, axis=0)
        std_pt[std_pt == 0] = 1e-6 # Prevent div by zero
        
        z_scores = np.abs((filtered_points - mean_pt) / std_pt)
        clean_mask = (z_scores[:, 0] < 2.0) & (z_scores[:, 1] < 2.0) & (z_scores[:, 2] < 2.0)
        clean_points = filtered_points[clean_mask]

        if len(clean_points) < 50:
            return

        # Republish clean cloud for RViz
        filtered_cloud = pc2.create_cloud_xyz32(transformed_cloud.header, clean_points.tolist())
        self.filtered_points_pub.publish(filtered_cloud)

        # 5. COMPUTE CENTROID & PCA
        centroid = np.mean(clean_points, axis=0)
        xy_points = clean_points[:, :2]
        centered_xy = xy_points - centroid[:2]
        
        cov_matrix = np.cov(centered_xy.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
        
        yaw = np.arctan2(principal_vector[1], principal_vector[0])

        # 6. CONVERT YAW TO QUATERNION & PUBLISH
        half_yaw = yaw / 2.0
        pose_msg = PoseStamped()
        pose_msg.header = transformed_cloud.header
        
        pose_msg.pose.position.x = float(centroid[0])
        pose_msg.pose.position.y = float(centroid[1])
        pose_msg.pose.position.z = float(centroid[2])
        
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = float(np.sin(half_yaw))
        pose_msg.pose.orientation.w = float(np.cos(half_yaw))

        self.cube_pose_pub.publish(pose_msg)

    def _on_parameter_update(self, params):
        # Update dynamically without restarting the node!
        for param in params:
            if param.name == 'min_z': self.min_z = float(param.value)
            elif param.name == 'max_z': self.max_z = float(param.value)
            elif param.name == 'max_y': self.max_y = float(param.value)
            elif param.name == 'r_min': self.r_min = int(param.value)
            elif param.name == 'r_max': self.r_max = int(param.value)
            elif param.name == 'g_min': self.g_min = int(param.value)
            elif param.name == 'g_max': self.g_max = int(param.value)
            elif param.name == 'b_min': self.b_min = int(param.value)
            elif param.name == 'b_max': self.b_max = int(param.value)
        
        self.get_logger().info("Parameters updated!")
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()