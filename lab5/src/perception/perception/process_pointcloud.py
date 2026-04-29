import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from rcl_interfaces.msg import SetParametersResult

class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')
        self.target_frame = self.declare_parameter('camera_depth_optical_frame', 'base_link').value
        self.max_y = float(self.declare_parameter('max_y', 0.67).value)

        self.min_z = float(self.declare_parameter('min_z', -0.18).value)
        self.max_z = float(self.declare_parameter('max_z', -0.15).value)
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.cube_pose_pub = self.create_publisher(PointStamped, '/cube_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info("Subscribed to PointCloud2 topic and marker publisher ready")

    def pointcloud_callback(self, msg: PointCloud2):

        # Transform the pointcloud from its original frame to base_link
        # Lookup Transform and use library function to transform cloud

        # Filter points between z coords between min_z and max_z and max_y
        # Call the numpy array filtered_points

        source_frame = 'camera_depth_optical_frame' # TODO: Fill in the source frame based on what you implemented in your static TF broadcaster 
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, source_frame, Time()) # TODO: the entire tf lookup params should be filled in
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {self.target_frame}: {ex}')
            return

        transformed_cloud = do_transform_cloud(msg, tf) # TODO: look what do_transform_cloud takes in and outputs

        raw_points = pc2.read_points(
            transformed_cloud,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        )
        
        points_base = np.column_stack(
                (raw_points['x'], raw_points['y'], raw_points['z'])
            ).astype(np.float32, copy=False)

        # TODO: Create masks based on the specified min, max y and z parameters above in order to filter points
        filtered_points = points_base[(points_base[:, 1] <= self.max_y) & (points_base[:, 2] <= self.max_z) & (points_base[:, 2] >= self.min_z)]

        if filtered_points.size == 0:
            self.get_logger().warn(
                f'No points after filters: z in [{self.min_z:.3f}, {self.max_z:.3f}] m, y <= {self.max_y:.3f} m'
            )
            return

        filtered_cloud = pc2.create_cloud_xyz32(
            transformed_cloud.header,
            filtered_points.tolist(),
        )
        self.filtered_points_pub.publish(filtered_cloud)

        centroid = np.mean(filtered_points, axis=0)
        cube_x, cube_y, cube_z = centroid[0], centroid[1], centroid[2]

    
        cube_pose = PointStamped()
        cube_pose.header = transformed_cloud.header
        cube_pose.point.x = float(cube_x)
        cube_pose.point.y = float(cube_y)
        cube_pose.point.z = float(cube_z)

        self.cube_pose_pub.publish(cube_pose)

    def _on_parameter_update(self, params):
        new_min_z = self.min_z
        new_max_z = self.max_z

        for param in params:
            if param.name == 'min_z' and param.type_ == Parameter.Type.DOUBLE:
                new_min_z = float(param.value)
            elif param.name == 'max_z' and param.type_ == Parameter.Type.DOUBLE:
                new_max_z = float(param.value)

        if new_min_z > new_max_z:
            return SetParametersResult(
                successful=False,
                reason='min_z must be <= max_z',
            )

        self.min_z = new_min_z
        self.max_z = new_max_z
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
