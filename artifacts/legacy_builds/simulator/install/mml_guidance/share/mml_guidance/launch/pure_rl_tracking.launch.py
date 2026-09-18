#!/usr/bin/env python3
"""
Pure RL Tracking Launch File
=============================
Launches the complete two-node RL tracking system with:
  - Howard (simulated target)
  - RL Policy Node (inference + integration)
  - Drone Commander (MAVROS interface)
  - Boundary Visualization
  - Optional monitoring terminals

Usage:
  # Velocity mode (direct control)
  ros2 launch mml_guidance pure_rl_tracking.launch.py control_mode:=velocity

  # Position mode (integrated waypoints)
  ros2 launch mml_guidance pure_rl_tracking.launch.py control_mode:=position

  # With custom control rate
  ros2 launch mml_guidance pure_rl_tracking.launch.py control_mode:=position control_rate:=20.0
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os
from pathlib import Path


def generate_launch_description():
    
    # Get workspace directory
    workspace_dir = str(Path.home() / 'mml_python_sim')
    
    return LaunchDescription([
        
        # ═══════════════════════════════════════════════════════════════
        # LAUNCH ARGUMENTS
        # ═══════════════════════════════════════════════════════════════
        DeclareLaunchArgument(
            'control_mode',
            default_value='position',
            description="Control mode: 'velocity' (direct) or 'position' (integrated)"
        ),
        
        DeclareLaunchArgument(
            'control_rate',
            default_value='10.0',
            description='RL policy update rate (Hz)'
        ),
        
        DeclareLaunchArgument(
            'model_path',
            default_value=f'{workspace_dir}/models_recurrent/recurrent_ppo_tracking_final_20251216_151239.zip',
            description='Path to trained RL model'
        ),
        
        DeclareLaunchArgument(
            'enable_monitoring_terminals',
            default_value='false',
            description='Enable separate xterm windows for topic monitoring'
        ),
        
        # ═══════════════════════════════════════════════════════════════
        # 1. HOWARD (Target Robot - Circle Motion)
        # ═══════════════════════════════════════════════════════════════
        Node(
            package='mml_guidance',
            executable='Sim_turtle',
            name='howard_sim',
            output='screen',
            parameters=[{
                'robot_name': 'howard',
                'pattern': 'circle',
                'velocity': 0.3,
                'radius': 0.8,
                'start_x': 0.0,
                'start_y': 0.0,
                'update_rate': 20.0,
            }]
        ),
        
        # ═══════════════════════════════════════════════════════════════
        # 2. RL POLICY NODE (Inference + Integration)
        # ═══════════════════════════════════════════════════════════════
        Node(
            package='mml_guidance',
            executable='rl_policy_node',
            name='rl_policy',
            output='screen',
            parameters=[{
                'control_mode': LaunchConfiguration('control_mode'),
                'control_rate': LaunchConfiguration('control_rate'),
                'model_path': LaunchConfiguration('model_path'),
                'fixed_altitude': 1.4,
                'max_tracked_targets': 10,
                
                # Integration parameters
                'max_position_jump': 0.5,
                'boundary_x_min': -2.0,
                'boundary_x_max': 2.0,
                'boundary_y_min': -1.5,
                'boundary_y_max': 1.5,
            }],
            emulate_tty=True,
        ),
        
        # ═══════════════════════════════════════════════════════════════
        # 3. DRONE COMMANDER (MAVROS Interface)
        # ═══════════════════════════════════════════════════════════════
        Node(
            package='mml_guidance',
            executable='drone_commander',
            name='drone_commander',
            output='screen',
            parameters=[{
                'control_mode': LaunchConfiguration('control_mode'),
                'fixed_altitude': 1.4,
                'altitude_kp': 0.5,
                'altitude_kd': 0.2,
                'max_climb_rate': 0.3,
                
                # Safety boundaries
                'boundary_x_min': -2.0,
                'boundary_x_max': 2.0,
                'boundary_y_min': -1.5,
                'boundary_y_max': 1.5,
                'boundary_z_min': 0.8,
                'boundary_z_max': 2.0,
                'warning_distance': 0.3,
                'emergency_distance': 0.15,
                
                # Position mode safety
                'max_waypoint_distance': 0.5,
            }],
            emulate_tty=True,
        ),
        
        # ═══════════════════════════════════════════════════════════════
        # 4. BOUNDARY VISUALIZATION
        # ═══════════════════════════════════════════════════════════════
        Node(
            package='mml_guidance',
            executable='boundary_visualization',
            name='boundary_viz',
            parameters=[{
                'boundary_x_min': -2.0,
                'boundary_x_max': 2.0,
                'boundary_y_min': -1.5,
                'boundary_y_max': 1.5,
                'boundary_z_min': 0.8,
                'boundary_z_max': 2.0,
            }]
        ),
    ])
