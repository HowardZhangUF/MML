from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    return LaunchDescription([
        # Model and tracking parameters
        DeclareLaunchArgument('model_path', default_value='',
                            description='Path to trained PPO model (.zip file)'),
        DeclareLaunchArgument('target_topic', default_value='/leo/enu/pose',
                            description='Topic for target pose'),
        DeclareLaunchArgument('max_vel', default_value='0.5',
                            description='Maximum velocity (m/s)'),
        DeclareLaunchArgument('drone_height', default_value='1.0',
                            description='Target altitude (m)'),
        DeclareLaunchArgument('num_particles', default_value='200',
                            description='Number of particles for filter'),
        DeclareLaunchArgument('prediction_method', default_value='Velocity',
                            description='Prediction method'),
        
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
        DeclareLaunchArgument('boundary_return_speed', default_value='0.3'),
        
        # Visualization parameters
        DeclareLaunchArgument('enable_visualization', default_value='true',
                            description='Enable RViz visualization'),
        DeclareLaunchArgument('velocity_arrow_scale', default_value='2.0',
                            description='Scale factor for velocity arrows'),
        
        LogInfo(msg="=== RL Offboard Tracker Launch ==="),
        LogInfo(msg="Deploying PyGame-trained policy with boundary safety + visualization"),
        
        # RL Tracker Node
        Node(
            package='mml_guidance',
            executable='rl_tracker',
            name='rl_offboard_tracker',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'target_topic': LaunchConfiguration('target_topic'),
                'max_vel': LaunchConfiguration('max_vel'),
                'drone_height': LaunchConfiguration('drone_height'),
                'num_particles': LaunchConfiguration('num_particles'),
                'prediction_method': LaunchConfiguration('prediction_method'),
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
                'velocity_arrow_scale': LaunchConfiguration('velocity_arrow_scale'),
                'show_corners': False,
            }],
            condition=IfCondition(LaunchConfiguration('enable_visualization')),
        ),
        
        LogInfo(msg="=== Nodes started (RL Tracker + Visualization) ==="),
    ])