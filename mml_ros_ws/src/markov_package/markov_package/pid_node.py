#!/usr/bin/env python3
"""
PID Controller for TurtleBot - MOCAP VERSION
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class TurtleBotPIDController(Node):
    def __init__(self):
        super().__init__("turtlebot_pid_controller")
        
        self.get_logger().info("=== PID Controller (MoCap Version) Started ===")
        
        # Parameters
        self.declare_parameter("robot_name", "leo")
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 1.0)
        self.declare_parameter("goal_tolerance", 0.15)
        
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        
        # PID gains
        self.kp_linear = 1.0
        self.kp_angular = 2.0
        
        # State
        self.current_position = None
        self.current_yaw = None
        self.goal_position = None
        
        # QoS - Must be BEST_EFFORT to hear /mocap
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Subscriber 1: LISTEN TO MOCAP 
        self.mocap_sub = self.create_subscription(
            PoseStamped,
            "/mocap/leo/pose", 
            self.mocap_callback,
            qos_profile
        )
        self.get_logger().info("Subscribed to /mocap/leo/pose (Best Effort)")
        
        # Subscriber 2: LISTEN TO GOAL
        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/leo/goal_pose",
            self.goal_callback,
            10
        )
        
        # Publisher: Drive Wheels
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f"/{self.robot_name}/cmd_vel",
            10
        )
        
        self.timer = self.create_timer(0.05, self.control_loop)
    
    def mocap_callback(self, msg):
        # Apply the same NEGATIVE fix as your Markov Node!
        self.current_position = np.array([
            -msg.pose.position.x, 
            -msg.pose.position.y
        ])
        
        q = msg.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
    
    def goal_callback(self, msg):
        self.goal_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])
        self.get_logger().info(f"New Goal: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}]")
    
    def control_loop(self):
        if self.current_position is None or self.goal_position is None:
            return
        
        # Calculate Error
        error_vector = self.goal_position - self.current_position
        distance = np.linalg.norm(error_vector)
        
        cmd = Twist()
        
        if distance < self.goal_tolerance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            desired_yaw = np.arctan2(error_vector[1], error_vector[0])
            yaw_error = self.angle_diff(desired_yaw, self.current_yaw)
            
            cmd.angular.z = np.clip(self.kp_angular * yaw_error, -self.max_angular_vel, self.max_angular_vel)
            
            # Only drive forward if roughly facing the target
            if abs(yaw_error) < 0.5:
                cmd.linear.x = np.clip(self.kp_linear * distance, 0.0, self.max_linear_vel)
            else:
                cmd.linear.x = 0.0
        
        self.cmd_vel_pub.publish(cmd)
    
    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    @staticmethod
    def angle_diff(target, current):
        diff = target - current
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        return diff

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
