import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    def _pkg_share(pkg_name: str):
        try:
            return get_package_share_directory(pkg_name)
        except Exception:
            return None

    def _existing_file(path: str):
        return path if path and os.path.isfile(path) else None

    # Arguments
    is_visualization = LaunchConfiguration("is_visualization")
    is_simulation = LaunchConfiguration("is_simulation")
    vehicle = LaunchConfiguration("vehicle")
    run_estimator = LaunchConfiguration("run_estimator")
    run_pose_to_vel = LaunchConfiguration("run_pose_to_vel")
    run_reef_control = LaunchConfiguration("run_reef_control")
    run_robot0 = LaunchConfiguration("run_robot0")

    declare_is_visualization = DeclareLaunchArgument(
        "is_visualization",
        default_value="true",
        description="Whether to enable visualization",
    )
    declare_is_simulation = DeclareLaunchArgument(
        "is_simulation",
        default_value="true",
        description="Whether to run in simulation mode",
    )
    declare_vehicle = DeclareLaunchArgument(
        "vehicle",
        default_value="sim",
        description="Vehicle type (used for sim_helper pid yaml naming)",
    )
    declare_run_estimator = DeclareLaunchArgument(
        "run_estimator",
        default_value="true",
        description="Launch reef_estimator if available",
    )
    declare_run_pose_to_vel = DeclareLaunchArgument(
        "run_pose_to_vel",
        default_value="true",
        description="Launch position_to_velocity if available",
    )
    declare_run_reef_control = DeclareLaunchArgument(
        "run_reef_control",
        default_value="true",
        description="Launch reef_control if available",
    )
    declare_run_robot0 = DeclareLaunchArgument(
        "run_robot0",
        default_value="true",
        description="Launch robot0 turtlebot-related nodes if available",
    )

    # Optional dependency package shares
    reef_estimator_share = _pkg_share("reef_estimator")
    sim_helper_share = _pkg_share("sim_helper")
    position_to_velocity_share = _pkg_share("position_to_velocity")
    reef_control_share = _pkg_share("reef_control")
    andres_turtlebot_pid_share = _pkg_share("andres_turtlebot_pid")

    # Collect warnings for missing optional deps
    warn_actions = []
    if reef_estimator_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Optional package 'reef_estimator' not found; skipping reef_estimator node."))
    if sim_helper_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Optional package 'sim_helper' not found; skipping sim_helper param usage."))
    if position_to_velocity_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Optional package 'position_to_velocity' not found; skipping pose_to_vel node."))
    if reef_control_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Optional package 'reef_control' not found; skipping reef_control_pid node."))
    if andres_turtlebot_pid_share is None:
        warn_actions.append(LogInfo(msg="[WARNING] Optional package 'andres_turtlebot_pid' not found; skipping robot0 turtlebot pid node."))

    # Always-available nodes (within this package)
    drone_guidance_node = Node(
        package="mml_guidance",
        executable="guidance",
        name="drone_guidance",
        output="screen",
        parameters=[{"is_viz": is_visualization, "is_sim": is_simulation}],
    )

    mml_pf_visualization_node = Node(
        package="mml_guidance",
        executable="mml_pf_visualization",
        name="mml_pf_visualization",
        output="screen",
        remappings=[("odom", "/robot0/odom")],
        condition=IfCondition(is_visualization),
    )

    # Optional nodes
    optional_nodes = []

    if reef_estimator_share is not None:
        xy_est_params = _existing_file(os.path.join(reef_estimator_share, "params", "xy_est_params.yaml"))
        z_est_params = _existing_file(os.path.join(reef_estimator_share, "params", "z_est_params.yaml"))
        basic_params = _existing_file(os.path.join(reef_estimator_share, "params", "basic_params.yaml"))
        missing_params = [p for p in (xy_est_params, z_est_params, basic_params) if p is None]
        if missing_params:
            warn_actions.append(LogInfo(msg="[WARNING] reef_estimator param yaml(s) not found; launching reef_estimator without those files."))

        estimator_params = [p for p in (xy_est_params, z_est_params, basic_params) if p is not None]
        estimator_params.append(
            {
                "enable_rgbd": False,
                "enable_sonar": True,
                "enable_mocap_xy": True,
                "enable_mocap_z": False,
            }
        )

        optional_nodes.append(
            Node(
                package="reef_estimator",
                executable="reef_estimator",
                name="reef_estimator",
                output="screen",
                parameters=estimator_params,
                remappings=[
                    ("mocap_ned", "pose_stamped"),
                    ("mocap_velocity/body_level_frame", "velocity/body_level_frame"),
                    ("rgbd_velocity_body_frame", "rgbd_velocity/body_level_frame"),
                ],
                condition=IfCondition(run_estimator),
            )
        )

    if position_to_velocity_share is not None:
        pose_to_vel_params = []
        if sim_helper_share is not None:
            sim_camera_to_body = _existing_file(os.path.join(sim_helper_share, "params", "sim_camera_to_body.yaml"))
            if sim_camera_to_body is not None:
                pose_to_vel_params.append(sim_camera_to_body)
            else:
                warn_actions.append(LogInfo(msg="[WARNING] sim_helper/params/sim_camera_to_body.yaml not found; pose_to_vel will use defaults."))

        basic_yaml = _existing_file(os.path.join(position_to_velocity_share, "params", "basic.yaml"))
        if basic_yaml is not None:
            pose_to_vel_params.append(basic_yaml)
        else:
            warn_actions.append(LogInfo(msg="[WARNING] position_to_velocity/params/basic.yaml not found; pose_to_vel will use defaults."))

        optional_nodes.append(
            Node(
                package="position_to_velocity",
                executable="position_to_velocity_node",
                name="pose_to_vel",
                output="screen",
                parameters=pose_to_vel_params,
                remappings=[("odom", "multirotor/truth/NED")],
                condition=IfCondition(run_pose_to_vel),
            )
        )

    if reef_control_share is not None and sim_helper_share is not None:
        # We can't reliably build a dynamic yaml path from LaunchConfiguration and check existence here;
        # instead we pass the file only if it exists for the *default* vehicle, else rely on node defaults.
        default_vehicle_pid = _existing_file(os.path.join(sim_helper_share, "params", "sim_pid.yaml"))
        reef_control_params = [default_vehicle_pid] if default_vehicle_pid is not None else []
        if default_vehicle_pid is None:
            warn_actions.append(LogInfo(msg="[WARNING] sim_helper/params/sim_pid.yaml not found; reef_control_pid will use defaults."))

        optional_nodes.append(
            Node(
                package="reef_control",
                executable="reef_control_node",
                name="reef_control_pid",
                output="screen",
                parameters=reef_control_params,
                condition=IfCondition(run_reef_control),
            )
        )
    elif reef_control_share is not None and sim_helper_share is None:
        # reef_control exists but sim_helper doesn't; still allow launching reef_control with defaults.
        optional_nodes.append(
            Node(
                package="reef_control",
                executable="reef_control_node",
                name="reef_control_pid",
                output="screen",
                parameters=[],
                condition=IfCondition(run_reef_control),
            )
        )

    # Robot0 Group (optional turtlebot PID + always-available markov goal pose)
    robot0_actions = [
        PushRosNamespace("robot0"),
        Node(
            package="mml_guidance",
            executable="markov_goal_pose",
            name="markov_goal_pose",
            output="screen",
            remappings=[("/odom", "/robot0/odom"), ("/goal_pose", "/robot0/goal_pose")],
        ),
    ]

    if andres_turtlebot_pid_share is not None:
        # Note: executable name varies by package; this is best-effort and gated by run_robot0.
        robot0_actions.insert(
            1,
            Node(
                package="andres_turtlebot_pid",
                executable="turtlebot_pid.py",
                name="turtlebot_pid",
                output="screen",
                remappings=[
                    ("/cmd_vel", "/robot0/cmd_vel"),
                    ("/odom", "/robot0/odom"),
                    ("/goal_pose", "/robot0/goal_pose"),
                ],
            ),
        )

    robot0_group = GroupAction(robot0_actions, condition=IfCondition(run_robot0))

    return LaunchDescription(
        [
            declare_is_visualization,
            declare_is_simulation,
            declare_vehicle,
            declare_run_estimator,
            declare_run_pose_to_vel,
            declare_run_reef_control,
            declare_run_robot0,
            LogInfo(msg="Launching mml_sim_estimator (ROS2)"),
            *warn_actions,
            drone_guidance_node,
            mml_pf_visualization_node,
            *optional_nodes,
            robot0_group,
        ]
    )
