#!/usr/bin/env python3
"""
Simulated Turtlebot "Howard" for RViz Testing
- Moves in configurable patterns (circle, square, random, figure-8)
- Publishes pose for UAV to track
- Publishes visualization markers
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
from enum import Enum


class MovementPattern(Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    FIGURE_8 = "figure8"
    RANDOM = "random"
    WAYPOINT = "waypoint"


class SimulatedTurtlebot(Node):
    """
    Simulated turtlebot that moves in RViz for UAV tracking tests
    """
    def __init__(self):
        super().__init__('simulated_turtlebot')

        # Parameters
        self.declare_parameter('robot_name', 'howard')
        self.declare_parameter('pattern', 'circle')  # circle, square, figure8, random, waypoint
        self.declare_parameter('velocity', 0.3)
        self.declare_parameter('radius', 0.8)
        self.declare_parameter('goal_tolerance', 0.15)      
        self.declare_parameter('markov_speed', 0.3)
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_z', 0.0)
        self.declare_parameter('update_rate', 20.0)  # Hz
        
        # Get parameters
        self.robot_name = self.get_parameter('robot_name').value
        pattern_str = self.get_parameter('pattern').value
        self.velocity = self.get_parameter('velocity').value
        self.radius = self.get_parameter('radius').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # Robot state
        self.position = np.array([
            self.get_parameter('start_x').value,
            self.get_parameter('start_y').value,
            self.get_parameter('start_z').value
        ])
        self.velocity_vec = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        
        # Pattern control
        try:
            self.pattern = MovementPattern(pattern_str)
        except ValueError:
            self.get_logger().warn(f"Invalid pattern '{pattern_str}', using circle")
            self.pattern = MovementPattern.CIRCLE

        # Markov navigation state
        self.markov_goal = None
        self.markov_goal_tolerance = self.get_parameter("goal_tolerance").value
        self.markov_speed = self.get_parameter("markov_speed").value
        
        self.time = 0.0
        self.dt = 1.0 / self.update_rate
        
        # Square waypoints
        self.square_waypoints = [
            np.array([self.radius, self.radius]),
            np.array([-self.radius, self.radius]),
            np.array([-self.radius, -self.radius]),
            np.array([self.radius, -self.radius])
        ]
        self.current_waypoint_idx = 0
        
        # Random walk
        self.random_target = None
        self.random_change_time = 3.0  # Change direction every 3s
        self.last_random_change = 0.0
        
        # Trajectory for visualization
        self.trajectory_points = []
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"SIMULATED TURTLEBOT: {self.robot_name.upper()}")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Pattern: {self.pattern.value}")
        self.get_logger().info(f"Velocity: {self.velocity} m/s")
        self.get_logger().info(f"Radius: {self.radius} m")
        self.get_logger().info(f"Start position: [{self.position[0]:.2f}, {self.position[1]:.2f}]")
        self.get_logger().info(f"Update rate: {self.update_rate} Hz")
        self.get_logger().info("=" * 60)

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped, f'/{self.robot_name}/enu/pose', 10)
        
        self.marker_pub = self.create_publisher(
            MarkerArray, f'/{self.robot_name}/markers', 10)

        # Timer
        self.timer = self.create_timer(self.dt, self.update_loop)
        
        self.get_logger().info(f"✓ {self.robot_name} ready - moving in {self.pattern.value} pattern")

    # ═══════════════════════════════════════════════════════════════════
    # MOVEMENT PATTERNS
    # ═══════════════════════════════════════════════════════════════════

    def update_circle_motion(self):
        """Move in circular pattern"""
        omega = self.velocity / self.radius
        self.position[0] = self.radius * np.cos(omega * self.time)
        self.position[1] = self.radius * np.sin(omega * self.time)
        
        # Velocity (tangent to circle)
        self.velocity_vec[0] = -self.radius * omega * np.sin(omega * self.time)
        self.velocity_vec[1] = self.radius * omega * np.cos(omega * self.time)
        
        # Yaw (tangent direction)
        self.yaw = np.arctan2(self.velocity_vec[1], self.velocity_vec[0])

    def update_figure8_motion(self):
        """Move in figure-8 pattern"""
        omega = self.velocity / self.radius
        self.position[0] = self.radius * np.sin(omega * self.time)
        self.position[1] = self.radius * np.sin(2 * omega * self.time) / 2
        
        # Velocity
        self.velocity_vec[0] = self.radius * omega * np.cos(omega * self.time)
        self.velocity_vec[1] = self.radius * omega * np.cos(2 * omega * self.time)
        
        self.yaw = np.arctan2(self.velocity_vec[1], self.velocity_vec[0])

    def update_square_motion(self):
        """Move in square pattern"""
        target = self.square_waypoints[self.current_waypoint_idx]
        direction = target - self.position[:2]
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            # Reached waypoint, go to next
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % 4
            target = self.square_waypoints[self.current_waypoint_idx]
            direction = target - self.position[:2]
            distance = np.linalg.norm(direction)
        
        if distance > 0.01:
            # Move toward waypoint
            direction_norm = direction / distance
            self.velocity_vec[:2] = direction_norm * self.velocity
            self.position[:2] += self.velocity_vec[:2] * self.dt
            self.yaw = np.arctan2(direction_norm[1], direction_norm[0])
        else:
            self.velocity_vec[:2] = [0.0, 0.0]

    def update_random_motion(self):
        """Random walk with smooth direction changes"""
        # Change direction periodically
        if self.time - self.last_random_change > self.random_change_time:
            # Pick random target within radius
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.3, self.radius)
            self.random_target = np.array([
                dist * np.cos(angle),
                dist * np.sin(angle)
            ])
            self.last_random_change = self.time
            self.get_logger().info(f"New random target: [{self.random_target[0]:.2f}, {self.random_target[1]:.2f}]")
        
        if self.random_target is not None:
            direction = self.random_target - self.position[:2]
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                direction_norm = direction / distance
                self.velocity_vec[:2] = direction_norm * self.velocity
                self.position[:2] += self.velocity_vec[:2] * self.dt
                self.yaw = np.arctan2(direction_norm[1], direction_norm[0])
            else:
                # Reached target, pick new one immediately
                self.last_random_change = self.time - self.random_change_time

    def update_waypoint_motion(self):
        """Stay at fixed position (for manual control testing)"""
        self.velocity_vec = np.array([0.0, 0.0, 0.0])

    # ═══════════════════════════════════════════════════════════════════
    # UPDATE LOOP
    # ═══════════════════════════════════════════════════════════════════

    def update_loop(self):
        """Main update loop"""
        # Update position based on pattern
        if self.pattern == MovementPattern.CIRCLE:
            self.update_circle_motion()
        elif self.pattern == MovementPattern.FIGURE_8:
            self.update_figure8_motion()
        elif self.pattern == MovementPattern.SQUARE:
            self.update_square_motion()
        elif self.pattern == MovementPattern.RANDOM:
            self.update_random_motion()
        elif self.pattern == MovementPattern.WAYPOINT:
            self.update_waypoint_motion()
        
        self.time += self.dt
        
        # Add to trajectory
        if len(self.trajectory_points) == 0 or \
           np.linalg.norm(self.position - self.trajectory_points[-1]) > 0.05:
            self.trajectory_points.append(self.position.copy())
            if len(self.trajectory_points) > 1000:
                self.trajectory_points.pop(0)
        
        # Publish
        self.publish_pose()
        self.publish_markers()

    # ═══════════════════════════════════════════════════════════════════
    # PUBLISHING
    # ═══════════════════════════════════════════════════════════════════

    def publish_pose(self):
        """Publish pose for UAV to track"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(self.position[0])
        msg.pose.position.y = float(self.position[1])
        msg.pose.position.z = float(self.position[2])
        
        # Quaternion from yaw
        msg.pose.orientation.z = np.sin(self.yaw / 2.0)
        msg.pose.orientation.w = np.cos(self.yaw / 2.0)
        
        self.pose_pub.publish(msg)

    def publish_markers(self):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # 1. Robot body (orange cube)
        body_marker = Marker()
        body_marker.header.frame_id = "map"
        body_marker.header.stamp = self.get_clock().now().to_msg()
        body_marker.ns = f"{self.robot_name}_body"
        body_marker.id = marker_id
        body_marker.type = Marker.CUBE
        body_marker.action = Marker.ADD
        body_marker.pose.position.x = float(self.position[0])
        body_marker.pose.position.y = float(self.position[1])
        body_marker.pose.position.z = float(self.position[2] + 0.1)  # Slightly above ground
        body_marker.pose.orientation.z = np.sin(self.yaw / 2.0)
        body_marker.pose.orientation.w = np.cos(self.yaw / 2.0)
        body_marker.scale.x = 0.3
        body_marker.scale.y = 0.2
        body_marker.scale.z = 0.15
        body_marker.color.r = 1.0
        body_marker.color.g = 0.5
        body_marker.color.b = 0.0
        body_marker.color.a = 1.0
        marker_array.markers.append(body_marker)
        marker_id += 1
        
        # 2. Direction arrow (red arrow)
        if np.linalg.norm(self.velocity_vec[:2]) > 0.01:
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "map"
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = f"{self.robot_name}_direction"
            arrow_marker.id = marker_id
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            from geometry_msgs.msg import Point
            start = Point()
            start.x = self.position[0]
            start.y = self.position[1]
            start.z = self.position[2] + 0.1
            
            end = Point()
            end.x = self.position[0] + self.velocity_vec[0] * 0.5
            end.y = self.position[1] + self.velocity_vec[1] * 0.5
            end.z = self.position[2] + 0.1
            
            arrow_marker.points.append(start)
            arrow_marker.points.append(end)
            
            arrow_marker.scale.x = 0.03
            arrow_marker.scale.y = 0.06
            arrow_marker.scale.z = 0.08
            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 0.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0
            marker_array.markers.append(arrow_marker)
            marker_id += 1
        
        # 3. Trajectory (orange line)
        if len(self.trajectory_points) > 1:
            traj_marker = Marker()
            traj_marker.header.frame_id = "map"
            traj_marker.header.stamp = self.get_clock().now().to_msg()
            traj_marker.ns = f"{self.robot_name}_trajectory"
            traj_marker.id = marker_id
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD
            traj_marker.scale.x = 0.03
            traj_marker.color.r = 1.0
            traj_marker.color.g = 0.5
            traj_marker.color.b = 0.0
            traj_marker.color.a = 0.6
            
            from geometry_msgs.msg import Point
            for point in self.trajectory_points:
                p = Point()
                p.x, p.y, p.z = point[0], point[1], point[2] + 0.05
                traj_marker.points.append(p)
            
            marker_array.markers.append(traj_marker)
            marker_id += 1
        
        # 4. Name label (text)
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = f"{self.robot_name}_label"
        text_marker.id = marker_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(self.position[0])
        text_marker.pose.position.y = float(self.position[1])
        text_marker.pose.position.z = float(self.position[2] + 0.4)
        text_marker.scale.z = 0.15
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = self.robot_name.upper()
        marker_array.markers.append(text_marker)
        marker_id += 1
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedTurtlebot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()