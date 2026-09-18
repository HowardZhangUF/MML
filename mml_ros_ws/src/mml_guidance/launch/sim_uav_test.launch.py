from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ★ NEW: Mode selection
        DeclareLaunchArgument('control_mode', default_value='rl_policy',
                            description='Control mode: rl_policy or predefined'),
        DeclareLaunchArgument('waypoint_update_rate', default_value='0.05',
                            description='Waypoint update rate (Hz) - affects both real and simulated drones'),
        
        # Model and tracking parameters 
        DeclareLaunchArgument('model_path', default_value='',
                            description='Path to trained PPO model'),
        DeclareLaunchArgument('target_topic', default_value='/howard/enu/pose'),
        DeclareLaunchArgument('max_vel', default_value='0.3'),
        DeclareLaunchArgument('drone_height', default_value='1.0'),
        
        # Position control parameters
        DeclareLaunchArgument('integration_steps', default_value='10'),
        DeclareLaunchArgument('integration_dt', default_value='0.05'),
        DeclareLaunchArgument('waypoint_threshold', default_value='0.1'),
        
        # Simulation parameters
        DeclareLaunchArgument('sim_dt', default_value='0.05'),
        DeclareLaunchArgument('sim_velocity', default_value='0.3'),
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        
        # Turtlebot parameters
        DeclareLaunchArgument('robot_name', default_value='howard'),
        DeclareLaunchArgument('pattern', default_value='circle'),
        DeclareLaunchArgument('turtlebot_velocity', default_value='0.15'),
        DeclareLaunchArgument('radius', default_value='0.8'),
        DeclareLaunchArgument('goal_tolerance', default_value='0.15'),
        
        # Real UAV specific parameters
        DeclareLaunchArgument('num_particles', default_value='200'),
        DeclareLaunchArgument('prediction_method', default_value='Velocity'),
        DeclareLaunchArgument('boundary_return_speed', default_value='0.3'),
        
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
        
        LogInfo(msg="=== SIMULATED UAV TRACKER TEST ==="),
        LogInfo(msg="Pure software simulation - no MAVROS required"),
        
        # 1. Markov Goal Generator
        Node(
            package='markov_package',
            executable='markov_node',
            name='howard_markov_node',
            output='screen',
            parameters=[{
                'robot_name': LaunchConfiguration('robot_name'),
                'is_sim': False,  
                'is_time_mode': False,
            }],
            emulate_tty=True,
        ),
        
        # Simulated UAV Tracker Node
        Node(
            package='mml_guidance',
            executable='simulated_uav_tracker',
            name='simulated_uav_tracker',
            output='screen',
            parameters=[{
                # ★ NEW: Add mode parameters
                'control_mode': LaunchConfiguration('control_mode'),
                'waypoint_update_rate': LaunchConfiguration('waypoint_update_rate'),
                
                # Existing parameters
                'model_path': LaunchConfiguration('model_path'),
                'target_topic': LaunchConfiguration('target_topic'),
                'max_vel': LaunchConfiguration('max_vel'),
                'drone_height': LaunchConfiguration('drone_height'),
                'integration_steps': LaunchConfiguration('integration_steps'),
                'integration_dt': LaunchConfiguration('integration_dt'),
                'waypoint_threshold': LaunchConfiguration('waypoint_threshold'),
                'sim_dt': LaunchConfiguration('sim_dt'),
                'sim_velocity': LaunchConfiguration('sim_velocity'),
                'initial_x': LaunchConfiguration('initial_x'),
                'initial_y': LaunchConfiguration('initial_y'),
                'boundary_x_min': LaunchConfiguration('boundary_x_min'),
                'boundary_x_max': LaunchConfiguration('boundary_x_max'),
                'boundary_y_min': LaunchConfiguration('boundary_y_min'),
                'boundary_y_max': LaunchConfiguration('boundary_y_max'),
                'boundary_z_min': LaunchConfiguration('boundary_z_min'),
                'boundary_z_max': LaunchConfiguration('boundary_z_max'),
                'safe_zone_radius': LaunchConfiguration('safe_zone_radius'),
                'warning_distance': LaunchConfiguration('warning_distance'),
                'emergency_distance': LaunchConfiguration('emergency_distance'),
            }],
            emulate_tty=True,
        ),

        # RL Tracker Node (real UAV)
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
                'waypoint_update_rate': LaunchConfiguration('waypoint_update_rate'),
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

        # Simulated Turtlebot (Howard)
        Node(
            package='mml_guidance',
            executable='simulated_turtlebot',
            name='simulated_turtlebot',
            output='screen',
            parameters=[{
                'robot_name': LaunchConfiguration('robot_name'),
                'pattern': LaunchConfiguration('pattern'),
                'velocity': LaunchConfiguration('turtlebot_velocity'),
                'radius': LaunchConfiguration('radius'),
                'goal_tolerance': LaunchConfiguration('goal_tolerance'),
                'start_x': 0.8,
                'start_y': 0.0,
                'start_z': 0.0,
                'update_rate': 20.0,
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
                'velocity_arrow_scale': 2.0,
                'show_corners': False,
            }],
        ),
        
        LogInfo(msg="=== Nodes started - Open RViz to visualize ==="),
    ])