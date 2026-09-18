#!/usr/bin/env python3
"""
PID Controller for TurtleBot to follow goal poses from markov_goal_pose node
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class TurtleBotPIDController(Node):
    def __init__(self):
        super().__init__("turtlebot_pid_controller")
        
        # Parameters
        self.declare_parameter("robot_name", "leo")
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 1.0)
        self.declare_parameter("goal_tolerance", 0.15)
        
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        
        self.get_logger().info(f"=== TurtleBot PID Controller for {self.robot_name} ===")
        self.get_logger().info(f"Max linear vel: {self.max_linear_vel} m/s")
        self.get_logger().info(f"Max angular vel: {self.max_angular_vel} rad/s")
        self.get_logger().info(f"Goal tolerance: {self.goal_tolerance} m")
        
        # PID gains
        self.kp_linear = 1.0
        self.kp_angular = 2.0
        
        # State
        self.current_position = None
        self.current_yaw = None
        self.goal_position = None
        self.goal_yaw = None
        
        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Subscribers - USE /leo/odom (Odometry message)
        self.odom_sub = self.create_subscription(
            Odometry,
            f"/{self.robot_name}/odom",
            self.odom_callback,
            qos_profile
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            f"/{self.robot_name}/goal_pose",
            self.goal_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f"/{self.robot_name}/cmd_vel",
            10
        )
        
        self.get_logger().info(f"Subscribed to /{self.robot_name}/odom")
        self.get_logger().info(f"Subscribed to /{self.robot_name}/goal_pose")
        self.get_logger().info(f"Publishing to /{self.robot_name}/cmd_vel")
        
        # Control loop timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Controller initialized and running!")
    
    def odom_callback(self, msg):
        """Update current position and orientation from Odometry"""
        self.current_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        
        # Log first position
        if not hasattr(self, '_first_position_logged'):
            self._first_position_logged = True
            self.get_logger().info(
                f"✓ First position: [{self.current_position[0]:.2f}, {self.current_position[1]:.2f}]m, "
                f"yaw={np.rad2deg(self.current_yaw):.1f}°"
            )
    
    def goal_callback(self, msg):
        """Update goal position and orientation"""
        self.goal_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])
        
        q = msg.pose.orientation
        self.goal_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        
        self.get_logger().info(
            f"New goal: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}]m, "
            f"yaw={np.rad2deg(self.goal_yaw):.1f}°",
            throttle_duration_sec=2.0
        )
    
    def control_loop(self):
        """PID control loop to drive robot to goal"""
        cmd = Twist()
        
        # Wait until we have both current state and goal
        if self.current_position is None or self.goal_position is None:
            self.cmd_vel_pub.publish(cmd)  # Publish zero velocity
            return
        
        # Calculate distance to goal
        error_vector = self.goal_position - self.current_position
        distance = np.linalg.norm(error_vector)
        
        # If within tolerance, stop and align to goal orientation
        if distance < self.goal_tolerance:
            # Only rotate to match goal yaw
            yaw_error = self.angle_diff(self.goal_yaw, self.current_yaw)
            
            if abs(yaw_error) > 0.1:  # 5.7 degrees
                cmd.angular.z = np.clip(
                    self.kp_angular * yaw_error,
                    -self.max_angular_vel,
                    self.max_angular_vel
                )
            else:
                cmd.angular.z = 0.0
            
            cmd.linear.x = 0.0
        else:
            # Drive toward goal
            desired_yaw = np.arctan2(error_vector[1], error_vector[0])
            yaw_error = self.angle_diff(desired_yaw, self.current_yaw)
            
            # Angular velocity (turn toward goal)
            cmd.angular.z = np.clip(
                self.kp_angular * yaw_error,
                -self.max_angular_vel,
                self.max_angular_vel
            )
            
            # Linear velocity (move forward if roughly aligned)
            if abs(yaw_error) < np.pi / 4:  # Within 45 degrees
                cmd.linear.x = np.clip(
                    self.kp_linear * distance,
                    0.0,
                    self.max_linear_vel
                )
            else:
                cmd.linear.x = 0.0  # Rotate in place first
        
        self.cmd_vel_pub.publish(cmd)
    
    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    @staticmethod
    def angle_diff(target, current):
        """Compute shortest angular difference (wraps around ±π)"""
        diff = target - current
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff


def main(args=None):
    rclpy.init(args=args)
    controller = TurtleBotPIDController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        controller.cmd_vel_pub.publish(Twist())
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()