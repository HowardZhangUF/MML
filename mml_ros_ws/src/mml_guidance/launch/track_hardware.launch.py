"""
ROS 2 Hardware Launch File for Drone + Mocap System

This launch file connects:
1. OptiTrack Motion Capture system (via VRPN/mocap_transform)
2. Drone autopilot (via rosflight/mavros)
3. Reef estimator for pose estimation
4. MML guidance system

Based on the ROS 1 track_hardware.launch file.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def _pkg_share(pkg_name: str):
    """Helper to safely get package share directory"""
    try:
        return get_package_share_directory(pkg_name)
    except Exception:
        return None


def generate_launch_description():
    # Launch arguments
    use_camera = LaunchConfiguration("use_camera")
    vehicle = LaunchConfiguration("vehicle")
    vehicle_type = LaunchConfiguration("vehicle_type")
    run_estimator = LaunchConfiguration("run_estimator")
    is_sim = LaunchConfiguration("is_sim")
    
    # Mocap settings
    mocap_server_ip = LaunchConfiguration("mocap_server_ip")
    rigid_body_name = LaunchConfiguration("rigid_body_name")  # For turtlebot
    
    declare_use_camera = DeclareLaunchArgument(
        "use_camera", 
        default_value="true",
        description="Launch RealSense camera node"
    )
    
    declare_vehicle = DeclareLaunchArgument(
        "vehicle",
        default_value="takahe",
        description="Vehicle name (drone rigid body name in OptiTrack)"
    )
    
    declare_vehicle_type = DeclareLaunchArgument(
        "vehicle_type",
        default_value="quad",
        description="Vehicle type for PID parameters"
    )
    
    declare_run_estimator = DeclareLaunchArgument(
        "run_estimator",
        default_value="true",
        description="Launch reef_estimator node"
    )
    
    declare_is_sim = DeclareLaunchArgument(
        "is_sim",
        default_value="false",
        description="Running in simulation mode"
    )
    
    declare_mocap_server_ip = DeclareLaunchArgument(
        "mocap_server_ip",
        default_value="192.168.1.104",
        description="OptiTrack/VRPN server IP address"
    )
    
    declare_rigid_body_name = DeclareLaunchArgument(
        "rigid_body_name",
        default_value="rail",
        description="Turtlebot rigid body name in OptiTrack"
    )

    # Check for optional packages
    reef_estimator_share = _pkg_share("reef_estimator")
    position_to_velocity_share = _pkg_share("position_to_velocity")
    reef_control_share = _pkg_share("reef_control")
    mocap_transform_share = _pkg_share("mocap_transform")
    realsense2_camera_share = _pkg_share("realsense2_camera")

    warn_actions = []
    if reef_estimator_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Package 'reef_estimator' not found"))
    if position_to_velocity_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Package 'position_to_velocity' not found"))
    if reef_control_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Package 'reef_control' not found"))
    if mocap_transform_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Package 'mocap_transform' not found - OptiTrack connection unavailable"))

    # ===== Core Nodes =====
    
    # MML Guidance node (always available)
    drone_guidance_node = Node(
        package="mml_guidance",
        executable="guidance",
        name="drone_guidance",
        output="screen",
        parameters=[{
            "is_viz": True,
            "is_sim": is_sim
        }],
        remappings=[
            ("/turtle_pose_stamped", ["/", rigid_body_name, "/nwu/pose_stamped"]),
            ("/quad_pose_stamped", ["/", vehicle, "/nwu/pose_stamped"])
        ]
    )

    # MML PF Visualization node
    mml_pf_visualization_node = Node(
        package="mml_guidance",
        executable="mml_pf_visualization",
        name="mml_pf_visualization",
        output="screen",
        remappings=[("odom", ["/", rigid_body_name, "/nwu/pose_stamped"])],
        parameters=[{"is_sim": is_sim}]
    )

    # ===== Optional Hardware Nodes =====
    optional_nodes = []

    # OptiTrack Mocap Transform nodes (for both drone and turtlebot)
    if mocap_transform_share is not None:
        # Drone mocap node
        drone_mocap_node = Node(
            package="mocap_transform",
            executable="mocap_transform_node",
            name=vehicle,
            output="screen",
            parameters=[{
                "rigid_body_name": vehicle,
                "vrpn_server_ip": mocap_server_ip
            }],
            remappings=[
                (f"/{vehicle}/ned/pose_stamped", "/pose_stamped")
            ]
        )
        optional_nodes.append(drone_mocap_node)
        
        # Turtlebot mocap node
        turtlebot_mocap_node = Node(
            package="mocap_transform",
            executable="mocap_transform_node",
            name=rigid_body_name,
            output="screen",
            parameters=[{
                "rigid_body_name": rigid_body_name,
                "vrpn_server_ip": mocap_server_ip
            }]
        )
        optional_nodes.append(turtlebot_mocap_node)

    # Reef Estimator (for drone state estimation)
    if reef_estimator_share is not None:
        xy_est_params = os.path.join(reef_estimator_share, "params", "xy_est_params.yaml")
        z_est_params = os.path.join(reef_estimator_share, "params", "z_est_params.yaml")
        basic_params = os.path.join(reef_estimator_share, "params", "basic_params.yaml")
        
        estimator_params = []
        for p in [xy_est_params, z_est_params, basic_params]:
            if os.path.isfile(p):
                estimator_params.append(p)
        
        estimator_params.append({
            "enable_rgbd": False,
            "enable_sonar": False,
            "enable_mocap_xy": True,
            "enable_mocap_z": True
        })
        
        reef_estimator_node = Node(
            package="reef_estimator",
            executable="reef_estimator",
            name="reef_estimator",
            output="log",
            parameters=estimator_params,
            remappings=[
                ("mocap_ned", "pose_stamped"),
                ("mocap_velocity/body_level_frame", "velocity/body_level_frame"),
                ("rgbd_velocity_body_frame", "rgbd_velocity/body_level_frame")
            ],
            condition=IfCondition(run_estimator)
        )
        optional_nodes.append(reef_estimator_node)

    # Position to Velocity node
    if position_to_velocity_share is not None:
        pose_to_vel_params = []
        basic_yaml = os.path.join(position_to_velocity_share, "params", "basic.yaml")
        if os.path.isfile(basic_yaml):
            pose_to_vel_params.append(basic_yaml)
        
        pose_to_vel_node = Node(
            package="position_to_velocity",
            executable="position_to_velocity_node",
            name="pose_to_vel",
            output="log",
            parameters=pose_to_vel_params
        )
        optional_nodes.append(pose_to_vel_node)

    # Reef Control PID
    if reef_control_share is not None:
        # Try to find vehicle-specific PID params
        reef_adaptive_control_share = _pkg_share("reef_adaptive_control")
        control_params = []
        if reef_adaptive_control_share is not None:
            pid_yaml = os.path.join(reef_adaptive_control_share, "params", f"{vehicle_type}_pid.yaml")
            if os.path.isfile(pid_yaml):
                control_params.append(pid_yaml)
        
        reef_control_node = Node(
            package="reef_control",
            executable="pid_control_node",
            name="reef_control_pid",
            output="screen",
            parameters=control_params
        )
        optional_nodes.append(reef_control_node)

    # RealSense Camera (optional)
    if realsense2_camera_share is not None:
        realsense_node = Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="realsense2_camera",
            output="log",
            condition=IfCondition(use_camera)
        )
        optional_nodes.append(realsense_node)

    # Turtlebot control group
    turtlebot_pid_share = _pkg_share("andres_turtlebot_pid")
    turtlebot_group_actions = []
    
    if turtlebot_pid_share is not None:
        turtlebot_group_actions.append(
            Node(
                package="andres_turtlebot_pid",
                executable="turtlebot_pid.py",
                name="turtlebot_pid",
                output="screen",
                parameters=[{"is_sim": False}],
                remappings=[
                    ("agent_pose", ["/", rigid_body_name, "/nwu/pose_stamped"]),
                    ("target_pose", ["/", rigid_body_name, "/goal_pose"]),
                    ("pub_info", "/mobile_base/commands/velocity")
                ]
            )
        )
    
    # Markov goal pose (trajectory generator)
    turtlebot_group_actions.append(
        Node(
            package="mml_guidance",
            executable="markov_goal_pose",
            name="markov_goal_pose",
            output="screen",
            remappings=[
                ("agent_pose", ["/", rigid_body_name, "/nwu/pose_stamped"]),
                ("/goal_pose", ["/", rigid_body_name, "/goal_pose"])
            ]
        )
    )
    
    turtlebot_group = GroupAction([
        PushRosNamespace(rigid_body_name),
        *turtlebot_group_actions
    ])

    return LaunchDescription([
        LogInfo(msg="=== MML Hardware Launch (ROS2) ==="),
        declare_use_camera,
        declare_vehicle,
        declare_vehicle_type,
        declare_run_estimator,
        declare_is_sim,
        declare_mocap_server_ip,
        declare_rigid_body_name,
        *warn_actions,
        drone_guidance_node,
        mml_pf_visualization_node,
        *optional_nodes,
        turtlebot_group,
        LogInfo(msg="=== All nodes launched ===")
    ])
