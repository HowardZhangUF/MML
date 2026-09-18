#!/bin/bash
# filepath: /home/andrespulido8/Documents/PhD_Stuff/deep_adp_mml/mml_guidance/mml_guidance/repeated_sim.sh

seconds=190  # this should be longer than the sim shutdown time
N_sims=10

# Launch ROS2 nodes in the background
ros2 launch mml_guidance mml_sim_estimator.launch.py &
ros2 launch mml_guidance bag_sim.launch.py prefix_name:="KF" bag_selected:="true" &
sleep $seconds

# Loop to launch the second ROS2 launch file N_sims times
for ((i=1; i<=N_sims-1; i++))
do
    echo "Iteration $i"
    ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/Pose "{
        position: {
            x: 0.0,
            y: 0.0,
            z: 0.0
        }, orientation: {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0
        }
    }"
    sleep 5
    ros2 launch mml_guidance sim_experiments.launch.py &
    ros2 launch mml_guidance bag_sim.launch.py prefix_name:="PFVelocity" bag_selected:="true" &
    sleep $seconds
done

echo "killing processes"
# Kill ROS2 specific processes
pkill -f "ros2 bag record"
pkill -f "ros2 launch"
pkill -f "ros2 run"
pkill python3
pkill reef_estimator
pkill reef_control
sleep 5

echo "DONE WITH REPEATED SIMS"