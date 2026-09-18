#!/usr/bin/env python3
"""
Drone Commander Node (NODE 2)
==============================
MAVROS interface node that:
  - Subscribes to /rl_policy/command
  - Applies safety checks (boundaries, velocity limits)
  - MODE 1 (velocity): Publishes to /mavros/setpoint_velocity/cmd_vel_unstamped
  - MODE 2 (position): Publishes to /mavros/setpoint_position/local
  - Logs all commands sent to MAVROS for verification
  
Subscribed Topics:
  - /rl_policy/command (RLCommand): Commands from RL policy
  - /pop/enu/pose (PoseStamped): Current UAV pose for safety checks
  
Published Topics:
  - /mavros/setpoint_velocity/cmd_vel_unstamped (Twist): Velocity commands
  - /mavros/setpoint_position/local (PoseStamped): Position commands
  - /drone_commander/markers (MarkerArray): Visualization
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import custom message
try:
    from mml_guidance.msg import RLCommand
except ImportError:
    print("WARNING: RLCommand message not built yet. Run: colcon build --packages-select mml_guidance")
    RLCommand = None


class DroneCommanderNode(Node):
    """Drone Commander - MAVROS Interface with Safety"""
    
    def __init__(self):
        super().__init__('drone_commander')
        
        # ═══════════════════════════════════════════════════════════════
        # DECLARE PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        self.declare_parameter('control_mode', 'velocity')
        self.declare_parameter('fixed_altitude', 1.4)
        self.declare_parameter('altitude_kp', 0.5)
        self.declare_parameter('altitude_kd', 0.2)
        self.declare_parameter('max_climb_rate', 0.3)
        
        # Boundaries
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        self.declare_parameter('boundary_z_min', 0.8)
        self.declare_parameter('boundary_z_max', 2.0)
        self.declare_parameter('warning_distance', 0.3)
        self.declare_parameter('emergency_distance', 0.15)
        
        # Position mode safety
        self.declare_parameter('max_waypoint_distance', 0.5)
        
        # Get parameters
        self.control_mode = self.get_parameter('control_mode').value
        self.fixed_altitude = self.get_parameter('fixed_altitude').value
        self.altitude_kp = self.get_parameter('altitude_kp').value
        self.altitude_kd = self.get_parameter('altitude_kd').value
        self.max_climb_rate = self.get_parameter('max_climb_rate').value
        
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        self.boundary_z_min = self.get_parameter('boundary_z_min').value
        self.boundary_z_max = self.get_parameter('boundary_z_max').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.emergency_distance = self.get_parameter('emergency_distance').value
        self.max_waypoint_distance = self.get_parameter('max_waypoint_distance').value
        
        self.get_logger().info("="*80)
        self.get_logger().info("📡 DRONE COMMANDER NODE INITIALIZED")
        self.get_logger().info("="*80)
        self.get_logger().info(f"Control Mode: {self.control_mode.upper()}")
        self.get_logger().info(f"Fixed Altitude: {self.fixed_altitude}m")
        self.get_logger().info("="*80)
        
        # ═══════════════════════════════════════════════════════════════
        # STATE VARIABLES
        # ═══════════════════════════════════════════════════════════════
        self.uav_pos = None
        self.uav_vel = None
        self.prev_uav_pos = None
        self.prev_uav_time = None
        
        self.in_emergency = False
        self.boundary_status = "UNKNOWN"
        
        self.last_waypoint = None
        
        # ═══════════════════════════════════════════════════════════════
        # PUBLISHERS
        # ═══════════════════════════════════════════════════════════════
        self.vel_publisher = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        
        self.pos_publisher = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)
        
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/drone_commander/markers', 10)
        
        # ═══════════════════════════════════════════════════════════════
        # SUBSCRIBERS
        # ═══════════════════════════════════════════════════════════════
        if RLCommand is not None:
            self.rl_cmd_sub = self.create_subscription(
                RLCommand, '/rl_policy/command', self.rl_command_callback, 10)
        else:
            self.get_logger().error("❌ RLCommand message not available!")
            self.rl_cmd_sub = None
        
        self.uav_sub = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.uav_pose_callback, 10)
        
        self.get_logger().info("✅ Drone Commander ready!")
    
    # ═══════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════
    
    def uav_pose_callback(self, msg):
        """Callback for UAV pose (for safety checks)"""
        current_time = self.get_clock().now()
        
        self.uav_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Compute velocity
        if self.prev_uav_pos is not None and self.prev_uav_time is not None:
            dt = (current_time - self.prev_uav_time).nanoseconds / 1e9
            if dt > 0:
                self.uav_vel = (self.uav_pos - self.prev_uav_pos) / dt
            else:
                self.uav_vel = np.zeros(3)
        else:
            self.uav_vel = np.zeros(3)
        
        self.prev_uav_pos = self.uav_pos.copy()
        self.prev_uav_time = current_time
    
    def rl_command_callback(self, msg):
        """Callback for RL commands"""
        
        if self.uav_pos is None:
            self.get_logger().warn("⏳ Waiting for UAV pose...", throttle_duration_sec=2.0)
            return
        
        receive_time = self.get_clock().now()
        
        # ───────────────────────────────────────────────────────────────
        # 1. Check Safety
        # ───────────────────────────────────────────────────────────────
        self.boundary_status = self.check_boundary_safety()
        
        # ───────────────────────────────────────────────────────────────
        # 2. Process Commands Based on Mode
        # ───────────────────────────────────────────────────────────────
        if msg.control_mode == "velocity":
            self.process_velocity_command(msg, receive_time)
        elif msg.control_mode == "position":
            self.process_position_command(msg, receive_time)
        else:
            self.get_logger().error(f"❌ Unknown control mode: {msg.control_mode}")
    
    # ═══════════════════════════════════════════════════════════════════
    # MODE 1: VELOCITY CONTROL
    # ═══════════════════════════════════════════════════════════════════
    
    def process_velocity_command(self, msg, receive_time):
        """Process velocity command from RL policy"""
        
        vx_cmd = msg.vx
        vy_cmd = msg.vy
        
        # Emergency override
        if self.in_emergency:
            vx_cmd, vy_cmd = self.compute_return_to_center()
        
        # Simplified altitude control (proportional only)
        altitude_error = self.fixed_altitude - self.uav_pos[2]
        vz_cmd = np.clip(altitude_error * 0.8, -self.max_climb_rate, self.max_climb_rate)
        
        # Create Twist message
        twist_msg = Twist()
        twist_msg.linear.x = float(vx_cmd)
        twist_msg.linear.y = float(vy_cmd)
        twist_msg.linear.z = float(vz_cmd)
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0  # Ignore omega
        
        # Publish to MAVROS
        self.vel_publisher.publish(twist_msg)
        
        send_time = self.get_clock().now()
        latency = (send_time - receive_time).nanoseconds / 1e9
        
        # Log
        self.log_sent_command(
            "VELOCITY", vx_cmd, vy_cmd, vz_cmd, 
            float('nan'), float('nan'), latency
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # MODE 2: POSITION CONTROL
    # ═══════════════════════════════════════════════════════════════════
    
    def process_position_command(self, msg, receive_time):
        """Process position command from RL policy"""
        
        target_x = msg.target_x
        target_y = msg.target_y
        # Use fixed altitude directly (no PID computation)
        target_z = self.fixed_altitude
        
        # Check for NaN
        if np.isnan(target_x) or np.isnan(target_y):
            self.get_logger().error("❌ Received NaN position from RL policy!")
            return
        
        # Check waypoint jump distance
        if self.last_waypoint is not None:
            jump_dist = np.sqrt(
                (target_x - self.last_waypoint[0])**2 + 
                (target_y - self.last_waypoint[1])**2
            )
            if jump_dist > self.max_waypoint_distance:
                self.get_logger().warn(
                    f"⚠️ Large waypoint jump: {jump_dist:.3f}m "
                    f"(max: {self.max_waypoint_distance}m)"
                )
        
        # Emergency override
        if self.in_emergency:
            target_x, target_y = 0.0, 0.0  # Return to center
            self.get_logger().warn("🔴 EMERGENCY: Overriding waypoint to center")
        
        # Check boundaries
        target_x = np.clip(target_x, self.boundary_x_min, self.boundary_x_max)
        target_y = np.clip(target_y, self.boundary_y_min, self.boundary_y_max)
        # target_z already set to fixed_altitude (no clipping needed)
        
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = float(target_x)
        pose_msg.pose.position.y = float(target_y)
        pose_msg.pose.position.z = float(target_z)
        
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        # Publish to MAVROS
        self.pos_publisher.publish(pose_msg)
        
        send_time = self.get_clock().now()
        latency = (send_time - receive_time).nanoseconds / 1e9
        
        # Update last waypoint
        self.last_waypoint = np.array([target_x, target_y])
        
        # Compute distance to waypoint
        distance_to_waypoint = np.sqrt(
            (target_x - self.uav_pos[0])**2 + 
            (target_y - self.uav_pos[1])**2
        )
        
        # Log
        self.log_sent_command(
            "POSITION", msg.vx, msg.vy, 0.0,
            target_x, target_y, latency, distance_to_waypoint
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # SAFETY CHECKS
    # ═══════════════════════════════════════════════════════════════════
    
    def check_boundary_safety(self):
        """Check distance to boundaries"""
        
        if self.uav_pos is None:
            return "UNKNOWN"
        
        dist_x_min = self.uav_pos[0] - self.boundary_x_min
        dist_x_max = self.boundary_x_max - self.uav_pos[0]
        dist_y_min = self.uav_pos[1] - self.boundary_y_min
        dist_y_max = self.boundary_y_max - self.uav_pos[1]
        dist_z_min = self.uav_pos[2] - self.boundary_z_min
        dist_z_max = self.boundary_z_max - self.uav_pos[2]
        
        min_dist = min(dist_x_min, dist_x_max, dist_y_min, 
                       dist_y_max, dist_z_min, dist_z_max)
        
        if min_dist < self.emergency_distance:
            self.in_emergency = True
            return "EMERGENCY"
        elif min_dist < self.warning_distance:
            return "WARNING"
        else:
            self.in_emergency = False
            return "SAFE"
    
    def compute_return_to_center(self):
        """Compute velocity to return to center"""
        center = np.array([0.0, 0.0])
        current = self.uav_pos[:2]
        direction = center - current
        
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction_norm = direction / distance
            return_speed = 0.3
            return direction_norm[0] * return_speed, direction_norm[1] * return_speed
        else:
            return 0.0, 0.0
    
    # def compute_altitude_control(self):
    #     """PD controller for altitude"""
    #     altitude_error = self.fixed_altitude - self.uav_pos[2]
    #     vz_cmd = (self.altitude_kp * altitude_error + 
    #               self.altitude_kd * (-self.uav_vel[2]))
    #     vz_cmd = np.clip(vz_cmd, -self.max_climb_rate, self.max_climb_rate)
    #     return vz_cmd
    
    # ═══════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════
    
    def log_sent_command(self, mode, vx, vy, vz, target_x, target_y, 
                        latency, dist_to_waypoint=None):
        """Log command sent to MAVROS with FULL DIAGNOSTIC INFO"""
        
        timestamp = self.get_clock().now().nanoseconds / 1e9
        
        print("\n" + "="*80)
        print(f"📡 DRONE COMMANDER - Command @ {timestamp:.3f}s")
        print("="*80)
        print(f"⚙️  CONTROL MODE: {mode}")
        print(f"⏱️  Latency (RL→Commander): {latency*1000:.2f}ms")
        print("-"*80)
        
        print(f"📥 RECEIVED FROM RL POLICY:")
        print(f"   Control Mode: {mode}")
        if mode == "VELOCITY":
            print(f"   RL Velocity: vx={vx:+.4f}, vy={vy:+.4f} m/s")
        else:  # POSITION
            print(f"   RL Velocity (raw): vx={vx:+.4f}, vy={vy:+.4f} m/s")
            print(f"   Integrated Position: x={target_x:+.4f}, y={target_y:+.4f} m")
        
        print("-"*80)
        print(f"🛡️  SAFETY CHECKS:")
        print(f"   Current UAV Position: x={self.uav_pos[0]:+.4f}, y={self.uav_pos[1]:+.4f}, z={self.uav_pos[2]:+.4f} m")
        print(f"   Boundary Status: {self.boundary_status}")
        
        if mode == "POSITION" and dist_to_waypoint is not None:
            print(f"   Distance to Target Waypoint: {dist_to_waypoint:.4f}m")
            if self.last_waypoint is not None:
                jump = np.sqrt((target_x - self.last_waypoint[0])**2 + 
                             (target_y - self.last_waypoint[1])**2)
                print(f"   Waypoint Jump Distance: {jump:.4f}m (max: {self.max_waypoint_distance}m)")
        
        print("-"*80)
        
        if mode == "VELOCITY":
            print(f"📤 SENT TO MAVROS (/mavros/setpoint_velocity/cmd_vel_unstamped):")
            print(f"   Topic: Twist message")
            print(f"   linear.x (vx): {vx:+.4f} m/s")
            print(f"   linear.y (vy): {vy:+.4f} m/s")
            print(f"   linear.z (vz): {vz:+.4f} m/s (simple proportional altitude control)")
            print(f"   angular.z: 0.0 rad/s")
            print(f"   🚀 MAVROS WILL APPLY THIS VELOCITY IMMEDIATELY")
            
        else:  # POSITION
            print(f"📤 SENT TO MAVROS (/mavros/setpoint_position/local):")
            print(f"   Topic: PoseStamped message")
            print(f"   position.x: {target_x:+.4f} m")
            print(f"   position.y: {target_y:+.4f} m")
            print(f"   position.z: {self.fixed_altitude:+.4f} m")
            print(f"   orientation: (0, 0, 0, 1) - quaternion")
            print(f"   🎯 PX4 WILL MOVE TO THIS WAYPOINT USING INTERNAL PID")
            print(f"   📊 PX4 will compute velocity based on distance to waypoint")
        
        print("="*80 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = DroneCommanderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
