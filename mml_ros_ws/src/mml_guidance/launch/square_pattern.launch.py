from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Pattern parameters
        DeclareLaunchArgument('pattern_type', default_value='square',
                             description="Pattern type: 'square', 'circle', or 'figure8'"),
        DeclareLaunchArgument('pattern_size', default_value='1.5',
                             description='Pattern size in meters (square side / circle diameter)'),
        DeclareLaunchArgument('pattern_center_x', default_value='0.0'),
        DeclareLaunchArgument('pattern_center_y', default_value='0.0'),
        DeclareLaunchArgument('flight_altitude', default_value='1.4'),
        DeclareLaunchArgument('waypoint_update_rate', default_value='0.5',
                             description='Waypoint update frequency in Hz'),
        DeclareLaunchArgument('waypoints_per_loop', default_value='40',
                             description='Total number of waypoints in complete pattern'),
        DeclareLaunchArgument('face_direction', default_value='true'),
        DeclareLaunchArgument('continuous_loop', default_value='true'),
        
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
        LogInfo(msg="PATTERN NAVIGATION WITH BOUNDARY SAFETY"),
        LogInfo(msg="=" * 60),
        
        # Pattern Navigator
        Node(
            package='mml_guidance',
            executable='go_here_square',
            name='pattern_navigator',
            output='screen',
            parameters=[{
                # Pattern configuration
                'pattern_type': LaunchConfiguration('pattern_type'),
                'pattern_size': LaunchConfiguration('pattern_size'),
                'pattern_center_x': LaunchConfiguration('pattern_center_x'),
                'pattern_center_y': LaunchConfiguration('pattern_center_y'),
                'flight_altitude': LaunchConfiguration('flight_altitude'),
                'waypoint_update_rate': LaunchConfiguration('waypoint_update_rate'),
                'waypoints_per_loop': LaunchConfiguration('waypoints_per_loop'),
                'face_direction': LaunchConfiguration('face_direction'),
                'continuous_loop': LaunchConfiguration('continuous_loop'),
                # Boundary safety
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
        
        # Boundary Visualization
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
                'show_corners': True,
            }],
        ),
        
        LogInfo(msg="=" * 60),
        LogInfo(msg="Pattern navigation with boundary safety active"),
    ])
