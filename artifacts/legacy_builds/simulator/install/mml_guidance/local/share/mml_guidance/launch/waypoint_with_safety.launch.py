from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Boundary parameters
        DeclareLaunchArgument('boundary_x_min', default_value='-2.0'),
        DeclareLaunchArgument('boundary_x_max', default_value='2.0'),
        DeclareLaunchArgument('boundary_y_min', default_value='-1.5'),
        DeclareLaunchArgument('boundary_y_max', default_value='1.5'),
        DeclareLaunchArgument('boundary_z_min', default_value='0.8'),
        DeclareLaunchArgument('boundary_z_max', default_value='2.0'),
        DeclareLaunchArgument('safe_zone_radius', default_value='0.5'),
        DeclareLaunchArgument('warning_distance', default_value='0.3'),
        DeclareLaunchArgument('emergency_distance', default_value='0.15'),
        DeclareLaunchArgument('boundary_return_speed', default_value='0.2'),
        
        LogInfo(msg="=" * 60),
        LogInfo(msg="WAYPOINT NAVIGATION WITH BOUNDARY SAFETY"),
        LogInfo(msg="=" * 60),
        LogInfo(msg="- Cycles through 5 waypoints: [-1, -0.5, 0, 0.5, 1]"),
        LogInfo(msg="- Boundary safety continuously active"),
        LogInfo(msg="- Emergency return if boundary violated"),
        LogInfo(msg="=" * 60),
        
        # Waypoint Navigation Node (go_here with safety)
        Node(
            package='mml_guidance',
            executable='go_here',
            name='waypoint_navigator',
            output='screen',
            parameters=[{
                'boundary_x_min': LaunchConfiguration('boundary_x_min'),
                'boundary_x_max': LaunchConfiguration('boundary_x_max'),
                'boundary_y_min': LaunchConfiguration('boundary_y_min'),
                'boundary_y_max': LaunchConfiguration('boundary_y_max'),
                'boundary_z_min': LaunchConfiguration('boundary_z_min'),
                'boundary_z_max': LaunchConfiguration('boundary_z_max'),
                'safe_zone_radius': LaunchConfiguration('safe_zone_radius'),
                'warning_distance': LaunchConfiguration('warning_distance'),
                'emergency_distance': LaunchConfiguration('emergency_distance'),
                'boundary_return_speed': LaunchConfiguration('boundary_return_speed'),
            }],
            emulate_tty=True,
        ),
        
        # Boundary Visualization Node
        Node(
            package='mml_guidance',
            executable='boundary_visualization',
            name='boundary_visualization',
            output='screen',
            parameters=[{
                'boundary_x_min': LaunchConfiguration('boundary_x_min'),
                'boundary_x_max': LaunchConfiguration('boundary_x_max'),
                'boundary_y_min': LaunchConfiguration('boundary_y_min'),
                'boundary_y_max': LaunchConfiguration('boundary_y_max'),
                'boundary_z_min': LaunchConfiguration('boundary_z_min'),
                'boundary_z_max': LaunchConfiguration('boundary_z_max'),
                'safe_zone_radius': LaunchConfiguration('safe_zone_radius'),
                'trajectory_max_points': 1000,
                'velocity_arrow_scale': 2.0,
                'show_corners': False,
            }],
        ),
        
        LogInfo(msg="=" * 60),
        LogInfo(msg="Nodes started:"),
        LogInfo(msg="  1. Waypoint Navigator (go_here with boundary safety)"),
        LogInfo(msg="  2. Boundary Visualization"),
        LogInfo(msg="=" * 60),
        LogInfo(msg="Open RViz to visualize boundaries and trajectory"),
    ])