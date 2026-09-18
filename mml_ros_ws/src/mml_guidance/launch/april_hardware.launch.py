import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Declare the argument for bag name
    declare_bag_name_arg = DeclareLaunchArgument(
        "bag_name", default_value="mml_guidance", description="Name of the bag to use."
    )

    # Declare the argument for simulation mode
    declare_is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="false",
        description="Whether to run in simulation mode.",
    )
    # Declare the argument for prediction method
    declare_prediction_method_arg = DeclareLaunchArgument(
        "prediction_method",
        default_value="Transformer",
        description="Prediction method: 'Transformer', 'NN', 'DMMN', 'KF', 'Velocity' or 'Unicycle'.",
    )

    # Declare the argument for guidance mode
    declare_guidance_mode_arg = DeclareLaunchArgument(
        "guidance_mode",
        default_value="standard",
        description="Guidance mode: 'Information', 'Lawnmower', 'WeightedMean', 'Estimator'",
    )

    # Declare the argument for number of particles
    declare_num_particles_arg = DeclareLaunchArgument(
        "N", default_value="500", description="Number of particles for the filter."
    )

    # Launch guidance node
    guidance_node = Node(
        package="mml_guidance",
        executable="guidance",
        name="drone_guidance",
        output="screen",
        remappings=[
            ("/turtle_pose_stamped", "/leo/enu/pose"),
            ("/quad_pose_stamped", "/pop/enu/pose"),
        ],
        parameters=[{"is_viz": True}, {"is_sim": LaunchConfiguration("is_sim")}],
    )

    # Launch mml_pf_visualization node
    mml_pf_visualization_node = Node(
        package="mml_guidance",
        executable="mml_pf_visualization",
        name="mml_pf_visualization",
        output="screen",
        remappings=[("odom", "/leo/enu/pose")],
        parameters=[{"is_sim": LaunchConfiguration("is_sim")}],
    )

    return LaunchDescription(
        [
            declare_bag_name_arg,
            declare_is_sim_arg,
            declare_prediction_method_arg,
            declare_guidance_mode_arg,
            declare_num_particles_arg,
            guidance_node,
            mml_pf_visualization_node,
        ]
    )
