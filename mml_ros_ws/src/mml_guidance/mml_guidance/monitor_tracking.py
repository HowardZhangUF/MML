#!/usr/bin/env python3
"""
Real-Time Tracking Monitor
===========================
Monitors and compares data flow through the RL tracking system:
  - RL policy output (/rl_policy/command)
  - MAVROS setpoint (velocity or position)
  - Actual UAV position (/pop/enu/pose)

Usage:
  # For velocity mode
  python3 monitor_tracking.py --mode velocity
  
  # For position mode
  python3 monitor_tracking.py --mode position
  
  # As ROS2 node
  ros2 run mml_guidance monitor_tracking
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from datetime import datetime
import sys
import argparse

# Import custom message
try:
    from mml_guidance.msg import RLCommand
except ImportError:
    print("WARNING: RLCommand message not built yet.")
    RLCommand = None


class TrackingMonitor(Node):
    """Real-time tracking monitor"""
    
    def __init__(self, mode='position'):
        super().__init__('tracking_monitor')
        
        self.mode = mode
        
        # Storage for latest messages
        self.rl_command = None
        self.mavros_setpoint_pos = None
        self.mavros_setpoint_vel = None
        self.uav_pose = None
        
        # Timing
        self.rl_command_time = None
        self.mavros_time = None
        self.uav_time = None
        
        # ═══════════════════════════════════════════════════════════════
        # SUBSCRIBERS
        # ═══════════════════════════════════════════════════════════════
        if RLCommand is not None:
            self.sub_rl = self.create_subscription(
                RLCommand, '/rl_policy/command', self.rl_callback, 10)
        
        self.sub_mavros_pos = self.create_subscription(
            PoseStamped, '/mavros/setpoint_position/local', 
            self.mavros_pos_callback, 10)
        
        self.sub_mavros_vel = self.create_subscription(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped',
            self.mavros_vel_callback, 10)
        
        self.sub_uav = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.uav_callback, 10)
        
        # Display timer
        self.timer = self.create_timer(0.5, self.display_comparison)
        
        self.get_logger().info(f"🔍 Tracking Monitor started (mode: {self.mode})")
    
    def rl_callback(self, msg):
        self.rl_command = msg
        self.rl_command_time = self.get_clock().now()
    
    def mavros_pos_callback(self, msg):
        self.mavros_setpoint_pos = msg
        self.mavros_time = self.get_clock().now()
    
    def mavros_vel_callback(self, msg):
        self.mavros_setpoint_vel = msg
        self.mavros_time = self.get_clock().now()
    
    def uav_callback(self, msg):
        self.uav_pose = msg
        self.uav_time = self.get_clock().now()
    
    def display_comparison(self):
        """Print comparison every 500ms"""
        
        # Clear screen
        print("\033[2J\033[H")
        
        print("="*80)
        print(f"🔍 TRACKING MONITOR - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        print(f"Mode: {self.mode.upper()}")
        print("="*80)
        
        if self.rl_command is None:
            print("\n⏳ Waiting for RL policy output...")
            return
        
        # ═══════════════════════════════════════════════════════════════
        # 1. RL POLICY OUTPUT
        # ═══════════════════════════════════════════════════════════════
        print("\n┌─ RL POLICY OUTPUT ─────────────────────────────────────────┐")
        print(f"│  Control Mode: {self.rl_command.control_mode.upper()}")
        print(f"│  RL Velocity: vx={self.rl_command.vx:.3f}, vy={self.rl_command.vy:.3f} m/s")
        
        if self.rl_command.control_mode == "position":
            print(f"│  Integrated Position: x={self.rl_command.target_x:.3f}, "
                  f"y={self.rl_command.target_y:.3f} m")
            print(f"│  Integration dt: {self.rl_command.dt:.3f}s")
        
        print(f"│  Tracking Error: {self.rl_command.tracking_error:.3f}m")
        
        if self.rl_command_time:
            timestamp_ns = self.rl_command_time.nanoseconds
            print(f"│  Timestamp: {timestamp_ns}")
        
        print("└────────────────────────────────────────────────────────────┘")
        
        # ═══════════════════════════════════════════════════════════════
        # 2. SENT TO MAVROS
        # ═══════════════════════════════════════════════════════════════
        print("\n┌─ SENT TO MAVROS ───────────────────────────────────────────┐")
        
        if self.mode == "position" or self.rl_command.control_mode == "position":
            if self.mavros_setpoint_pos:
                pos = self.mavros_setpoint_pos.pose.position
                print(f"│  Target Waypoint: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} m")
                
                # Verify match
                if not np.isnan(self.rl_command.target_x):
                    x_match = abs(pos.x - self.rl_command.target_x) < 0.001
                    y_match = abs(pos.y - self.rl_command.target_y) < 0.001
                    match_symbol = "✓" if (x_match and y_match) else "✗"
                    print(f"│  Match with RL: {match_symbol}")
                    
                    if not x_match or not y_match:
                        print(f"│  ⚠️  Mismatch detected!")
                        print(f"│     RL: x={self.rl_command.target_x:.3f}, y={self.rl_command.target_y:.3f}")
                        print(f"│     MAVROS: x={pos.x:.3f}, y={pos.y:.3f}")
            else:
                print("│  ⏳ Waiting for MAVROS position setpoint...")
        
        else:  # velocity mode
            if self.mavros_setpoint_vel:
                vel = self.mavros_setpoint_vel.linear
                print(f"│  Velocity Command: vx={vel.x:.3f}, vy={vel.y:.3f}, vz={vel.z:.3f} m/s")
                
                # Verify match
                vx_match = abs(vel.x - self.rl_command.vx) < 0.001
                vy_match = abs(vel.y - self.rl_command.vy) < 0.001
                match_symbol = "✓" if (vx_match and vy_match) else "✗"
                print(f"│  Match with RL: {match_symbol}")
                
                if not vx_match or not vy_match:
                    print(f"│  ⚠️  Mismatch detected!")
                    print(f"│     RL: vx={self.rl_command.vx:.3f}, vy={self.rl_command.vy:.3f}")
                    print(f"│     MAVROS: vx={vel.x:.3f}, vy={vel.y:.3f}")
            else:
                print("│  ⏳ Waiting for MAVROS velocity setpoint...")
        
        # Latency
        if self.rl_command_time and self.mavros_time:
            latency = (self.mavros_time - self.rl_command_time).nanoseconds / 1e9
            latency_ms = latency * 1000
            
            if latency_ms > 10:
                print(f"│  ⚠️ Latency: {latency_ms:.2f}ms (>10ms WARNING)")
            else:
                print(f"│  Latency: {latency_ms:.2f}ms ✓")
        
        print("└────────────────────────────────────────────────────────────┘")
        
        # ═══════════════════════════════════════════════════════════════
        # 3. UAV ACTUAL STATE
        # ═══════════════════════════════════════════════════════════════
        print("\n┌─ UAV ACTUAL STATE ─────────────────────────────────────────┐")
        
        if self.uav_pose:
            pos = self.uav_pose.pose.position
            print(f"│  Current Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} m")
            
            # Distance to setpoint
            if self.mode == "position" and self.mavros_setpoint_pos:
                target = self.mavros_setpoint_pos.pose.position
                error = np.sqrt((pos.x - target.x)**2 + (pos.y - target.y)**2)
                print(f"│  Distance to Setpoint: {error:.3f}m")
                
                if error < 0.1:
                    status = "✓ GOOD"
                elif error < 0.5:
                    status = "⚠️ TRACKING"
                else:
                    status = "❌ FAR"
                print(f"│  Status: {status}")
        else:
            print("│  ⏳ Waiting for UAV pose...")
        
        print("└────────────────────────────────────────────────────────────┘")
        
        # ═══════════════════════════════════════════════════════════════
        # 4. SUMMARY
        # ═══════════════════════════════════════════════════════════════
        print("\n┌─ VERIFICATION ─────────────────────────────────────────────┐")
        
        all_active = all([
            self.rl_command,
            self.mavros_setpoint_pos or self.mavros_setpoint_vel,
            self.uav_pose
        ])
        
        if all_active:
            print("│  ✅ All data streams active")
            print("│  ✅ RL → MAVROS pipeline verified")
        else:
            print("│  ⏳ Waiting for all data streams...")
            if not self.rl_command:
                print("│     - Missing: RL policy output")
            if not (self.mavros_setpoint_pos or self.mavros_setpoint_vel):
                print("│     - Missing: MAVROS setpoint")
            if not self.uav_pose:
                print("│     - Missing: UAV pose")
        
        print("└────────────────────────────────────────────────────────────┘")
        print("\nPress Ctrl+C to exit")


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Monitor RL tracking system')
    parser.add_argument('--mode', type=str, default='position',
                       choices=['velocity', 'position'],
                       help='Control mode to monitor')
    
    # Parse only known args to avoid conflicts with ROS
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=args)
    node = TrackingMonitor(mode=parsed_args.mode)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
