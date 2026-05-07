from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # -------------------------
    # Declare args
    # -------------------------

    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="false")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")

    # -------------------------
    # Includes & Nodes
    # -------------------------
    # RealSense (include rs_launch.py)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )

    # Perception node
    perception_node = Node(
        package='perception',
        executable='process_pointcloud',
        name='process_pointcloud',
        output='screen',
        parameters=[{
            'pointcloud_topic': pointcloud_topic,
            'board_rows': 12,
            'board_cols': 10,
            'playable_row_offset': 2,
            'playable_col_offset': 0,
            'playable_rows': 8,
            'playable_cols': 10,
        }]
    )

    # Planning TF node
    planning_tf_node = Node(
        package='planning',
        executable='tf',
        name='tf_node',
        output='screen'
    )

    # MoveIt include
    moveit_launch_file = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py"
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz
        }.items(),
    )

    cube_grasp_node = Node(
        package='planning',
        executable='main',
        name='cube_grasp',
        output='screen'
    )

    # -------------------------
    # LaunchDescription
    # -------------------------
    return LaunchDescription([

        # Actions
        DeclareLaunchArgument(
            "pointcloud_topic",
            default_value="/camera/camera/depth/color/points"
        ),
        realsense_launch,
        perception_node,
        planning_tf_node,
        moveit_launch, 
        cube_grasp_node, 
    ])
