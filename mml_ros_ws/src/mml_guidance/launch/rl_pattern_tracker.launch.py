from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ═══════════════════════════════════════════════════════════
        # PATTERN PARAMETERS
        # ═══════════════════════════════════════════════════════════
        DeclareLaunchArgument('pattern_type', default_value='square',
                             description="Pattern type: 'square', 'circle', 'figure8', or 'none' for target tracking"),
        DeclareLaunchArgument('pattern_size', default_value='1.5',
                             description='Pattern size in meters (square side / circle diameter)'),
        DeclareLaunchArgument('pattern_center_x', default_value='0.0',
                             description='Pattern center X coordinate'),
        DeclareLaunchArgument('pattern_center_y', default_value='0.0',
                             description='Pattern center Y coordinate'),
        DeclareLaunchArgument('waypoints_per_loop', default_value='40',
                             description='Total number of waypoints in complete pattern'),
        DeclareLaunchArgument('waypoint_update_rate', default_value='0.5',
                             description='Waypoint advancement frequency in Hz'),
        DeclareLaunchArgument('continuous_loop', default_value='true',
                             description='Keep looping through waypoints'),
        
        # ═══════════════════════════════════════════════════════════
        # RL MODEL PARAMETERS
        # ═══════════════════════════════════════════════════════════
        DeclareLaunchArgument('model_path', 
                             default_value='',
                             description='Path to RL policy model'),
        DeclareLaunchArgument('target_topic', default_value='/leo/enu/pose',
                             description='Target topic (only used when pattern_type=none)'),
        DeclareLaunchArgument('drone_height', default_value='1.4',
                             description='Flight altitude in meters'),
        DeclareLaunchArgument('max_vel', default_value='0.5',
                             description='Maximum velocity command from RL policy'),
        
        # ═══════════════════════════════════════════════════════════
        # CONTROL PARAMETERS
        # ═══════════════════════════════════════════════════════════
        DeclareLaunchArgument('waypoint_threshold', default_value='0.3',
                             description='Distance to consider waypoint reached (m)'),
        DeclareLaunchArgument('waypoint_velocity', default_value='0.3',
                             description='Velocity toward waypoint (m/s)'),
        DeclareLaunchArgument('integration_steps', default_value='10',
                             description='Integration steps for velocity'),
        DeclareLaunchArgument('integration_dt', default_value='0.05',
                             description='Integration time step'),
        
        # ═══════════════════════════════════════════════════════════
        # BOUNDARY SAFETY PARAMETERS
        # ═══════════════════════════════════════════════════════════
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
        
        # ═══════════════════════════════════════════════════════════
        # PARTICLE FILTER PARAMETERS
        # ═══════════════════════════════════════════════════════════
        DeclareLaunchArgument('use_particle_filter', default_value='false',
                             description='Enable particle filter (only for target tracking mode, pattern_type=none)'),
        DeclareLaunchArgument('num_particles', default_value='200',
                             description='Number of particles for prediction'),
        DeclareLaunchArgument('prediction_method', default_value='Velocity',
                             description='Prediction method: Velocity, ConstantVelocity, etc.'),
        
        LogInfo(msg="=" * 80),
        LogInfo(msg="RL OFFBOARD TRACKER - PATTERN FOLLOWING MODE"),
        LogInfo(msg="=" * 80),
        
        # ═══════════════════════════════════════════════════════════
        # RL OFFBOARD TRACKER NODE
        # ═══════════════════════════════════════════════════════════
        Node(
            package='mml_guidance',
            executable='rl_tracker',
            name='rl_pattern_tracker',
            output='screen',
            parameters=[{
                # Pattern configuration
                'pattern_type': LaunchConfiguration('pattern_type'),
                'pattern_size': LaunchConfiguration('pattern_size'),
                'pattern_center_x': LaunchConfiguration('pattern_center_x'),
                'pattern_center_y': LaunchConfiguration('pattern_center_y'),
                'waypoints_per_loop': LaunchConfiguration('waypoints_per_loop'),
                'waypoint_update_rate': LaunchConfiguration('waypoint_update_rate'),
                'continuous_loop': LaunchConfiguration('continuous_loop'),
                
                # RL model
                'model_path': LaunchConfiguration('model_path'),
                'target_topic': LaunchConfiguration('target_topic'),
                'drone_height': LaunchConfiguration('drone_height'),
                'max_vel': LaunchConfiguration('max_vel'),
                
                # Control parameters
                'waypoint_threshold': LaunchConfiguration('waypoint_threshold'),
                'waypoint_velocity': LaunchConfiguration('waypoint_velocity'),
                'integration_steps': LaunchConfiguration('integration_steps'),
                'integration_dt': LaunchConfiguration('integration_dt'),
                
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
                
                # Particle filter
                'use_particle_filter': LaunchConfiguration('use_particle_filter'),
                'num_particles': LaunchConfiguration('num_particles'),
                'prediction_method': LaunchConfiguration('prediction_method'),
            }],
            emulate_tty=True,
        ),
        
        # ═══════════════════════════════════════════════════════════
        # BOUNDARY VISUALIZATION NODE
        # ═══════════════════════════════════════════════════════════
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
        
        LogInfo(msg="=" * 80),
        LogInfo(msg="Launch parameters:"),
        LogInfo(msg="  Pattern: Use 'pattern_type' to select square/circle/figure8/none"),
        LogInfo(msg="  Size: Use 'pattern_size' to set pattern dimensions"),
        LogInfo(msg="  Update rate: Use 'waypoint_update_rate' to control advancement speed"),
        LogInfo(msg="  Model: Use 'model_path' to specify RL policy"),
        LogInfo(msg="=" * 80),
        LogInfo(msg="💡 Example usage:"),
        LogInfo(msg="  ros2 launch mml_guidance rl_pattern_tracker.launch.py pattern_type:=circle pattern_size:=2.0"),
        LogInfo(msg="  ros2 launch mml_guidance rl_pattern_tracker.launch.py pattern_type:=square waypoints_per_loop:=60"),
        LogInfo(msg="  ros2 launch mml_guidance rl_pattern_tracker.launch.py pattern_type:=none  # Target tracking mode"),
        LogInfo(msg="=" * 80),
    ])
