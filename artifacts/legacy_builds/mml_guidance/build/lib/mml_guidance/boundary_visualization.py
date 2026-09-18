#!/usr/bin/env python3
"""
Boundary Visualization Node for RViz - Enhanced for RL Tracker
Visualizes:
- UAV position (sphere with color based on boundary proximity)
- UAV velocity vector (arrow)
- UAV trajectory (line strip)
- Leo position (different colored sphere)
- Leo velocity vector (arrow)
- Leo trajectory (line strip)
- Safety boundaries (box wireframe)
- Safe zone (transparent sphere)
- Corner markers (for boundary test mode)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np


class BoundaryVisualization(Node):
    def __init__(self):
        super().__init__('boundary_visualization')
        
        # Parameters
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        self.declare_parameter('boundary_z_min', 0.8)
        self.declare_parameter('boundary_z_max', 2.0)
        self.declare_parameter('safe_zone_radius', 0.5)
        self.declare_parameter('trajectory_max_points', 1000)
        self.declare_parameter('velocity_arrow_scale', 2.0)  # Scale factor for velocity arrows
        self.declare_parameter('show_corners', False)  # Show corner markers (for boundary test)
        
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        self.boundary_z_min = self.get_parameter('boundary_z_min').value
        self.boundary_z_max = self.get_parameter('boundary_z_max').value
        self.safe_zone_radius = self.get_parameter('safe_zone_radius').value
        self.trajectory_max_points = self.get_parameter('trajectory_max_points').value
        self.velocity_arrow_scale = self.get_parameter('velocity_arrow_scale').value
        self.show_corners = self.get_parameter('show_corners').value
        
        # Calculate center
        self.center_x = (self.boundary_x_min + self.boundary_x_max) / 2.0
        self.center_y = (self.boundary_y_min + self.boundary_y_max) / 2.0
        self.center_z = (self.boundary_z_min + self.boundary_z_max) / 2.0
        
        # Corner positions (for boundary test visualization)
        margin = 0.05
        self.corners = [
            (self.boundary_x_min + margin, self.boundary_y_min + margin, "BL"),
            (self.boundary_x_max - margin, self.boundary_y_min + margin, "BR"),
            (self.boundary_x_max - margin, self.boundary_y_max - margin, "TR"),
            (self.boundary_x_min + margin, self.boundary_y_max - margin, "TL"),
        ]
        
        # State storage
        self.uav_pos = None
        self.uav_vel = None
        self.uav_trajectory_points = []
        
        self.leo_pos = None
        self.leo_vel = None
        self.leo_trajectory_points = []
        self.last_leo_pos = None
        self.last_leo_time = None
        
        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.sub_uav_pose = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.uav_pose_cb, qos_profile)
        
        self.sub_leo_pose = self.create_subscription(
            PoseStamped, '/leo/enu/pose', self.leo_pose_cb, qos_profile)
        
        # Try to subscribe to velocity topics (they might not always be available)
        try:
            from geometry_msgs.msg import TwistStamped, Twist
            self.sub_uav_vel = self.create_subscription(
                Twist, '/mavros/local_position/velocity_local', self.uav_vel_cb, qos_profile)
        except:
            self.get_logger().warn("Could not subscribe to UAV velocity topic")
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/boundary_viz/markers', 10)
        
        # Timer for publishing markers
        self.viz_timer = self.create_timer(0.1, self.publish_visualization)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("BOUNDARY VISUALIZATION NODE - ENHANCED FOR RL TRACKER")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Boundaries: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] "
                              f"Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}] "
                              f"Z[{self.boundary_z_min:.1f}, {self.boundary_z_max:.1f}]")
        self.get_logger().info(f"Safe center: ({self.center_x:.1f}, {self.center_y:.1f})")
        self.get_logger().info(f"Velocity arrow scale: {self.velocity_arrow_scale}x")
        self.get_logger().info(f"Show corners: {self.show_corners}")
        self.get_logger().info("=" * 60)

    def uav_pose_cb(self, msg):
        """Update UAV position and trajectory"""
        self.uav_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        
        # Add to trajectory
        if len(self.uav_trajectory_points) == 0:
            self.uav_trajectory_points.append(self.uav_pos.copy())
        else:
            last_point = self.uav_trajectory_points[-1]
            dist = np.linalg.norm(np.array(self.uav_pos) - np.array(last_point))
            if dist > 0.05:  # Only add if moved 5cm
                self.uav_trajectory_points.append(self.uav_pos.copy())
                
                # Limit trajectory length
                if len(self.uav_trajectory_points) > self.trajectory_max_points:
                    self.uav_trajectory_points.pop(0)

    def uav_vel_cb(self, msg):
        """Update UAV velocity"""
        if hasattr(msg, 'twist'):
            v = msg.twist.linear
        else:
            v = msg.linear
        self.uav_vel = [v.x, v.y, v.z]

    def leo_pose_cb(self, msg):
        """Update Leo position and estimate velocity"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        new_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        
        # Estimate velocity from position change
        if self.last_leo_pos is not None and self.last_leo_time is not None:
            dt = current_time - self.last_leo_time
            if dt > 0.01:  # Avoid division by zero
                vel = [(new_pos[i] - self.last_leo_pos[i]) / dt for i in range(3)]
                # Simple exponential smoothing
                if self.leo_vel is None:
                    self.leo_vel = vel
                else:
                    alpha = 0.3
                    self.leo_vel = [alpha * vel[i] + (1 - alpha) * self.leo_vel[i] for i in range(3)]
        
        self.leo_pos = new_pos
        self.last_leo_pos = new_pos.copy()
        self.last_leo_time = current_time
        
        # Add to trajectory
        if len(self.leo_trajectory_points) == 0:
            self.leo_trajectory_points.append(self.leo_pos.copy())
        else:
            last_point = self.leo_trajectory_points[-1]
            dist = np.linalg.norm(np.array(self.leo_pos) - np.array(last_point))
            if dist > 0.05:  # Only add if moved 5cm
                self.leo_trajectory_points.append(self.leo_pos.copy())
                
                # Limit trajectory length
                if len(self.leo_trajectory_points) > self.trajectory_max_points:
                    self.leo_trajectory_points.pop(0)

    def publish_visualization(self):
        """Publish all visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # 1. Boundary Box (wireframe) - RED
        marker_array.markers.append(self.create_boundary_box(marker_id))
        marker_id += 1
        
        # 2. Safe Zone (transparent sphere) - GREEN
        marker_array.markers.append(self.create_safe_zone(marker_id))
        marker_id += 1
        
        # 3. Center Point - WHITE
        marker_array.markers.append(self.create_center_marker(marker_id))
        marker_id += 1
        
        # 4. Corner markers (optional, for boundary test mode)
        if self.show_corners:
            for i, (x, y, name) in enumerate(self.corners):
                marker_array.markers.append(self.create_corner_marker(marker_id, x, y, name))
                marker_id += 1
        
        # ═══════════════════════════════════════════════════════════
        # UAV VISUALIZATION
        # ═══════════════════════════════════════════════════════════
        
        if self.uav_pos is not None:
            # 5. UAV position (sphere) - Color based on boundary proximity
            marker_array.markers.append(self.create_uav_marker(marker_id))
            marker_id += 1
            
            # 6. UAV velocity vector (arrow) - CYAN
            if self.uav_vel is not None:
                vel_mag = np.linalg.norm(self.uav_vel[:2])
                if vel_mag > 0.05:  # Only show if moving
                    marker_array.markers.append(self.create_velocity_arrow(
                        marker_id, self.uav_pos, self.uav_vel, 
                        color=[0.0, 1.0, 1.0, 1.0], ns="uav_velocity"
                    ))
                    marker_id += 1
        
        # 7. UAV trajectory (line strip) - BLUE
        if len(self.uav_trajectory_points) > 1:
            marker_array.markers.append(self.create_trajectory_marker(
                marker_id, self.uav_trajectory_points, 
                color=[0.0, 0.5, 1.0, 0.8], ns="uav_trajectory"
            ))
            marker_id += 1
        
        # ═══════════════════════════════════════════════════════════
        # LEO VISUALIZATION
        # ═══════════════════════════════════════════════════════════
        
        if self.leo_pos is not None:
            # 8. Leo position (sphere) - ORANGE
            marker_array.markers.append(self.create_leo_marker(marker_id))
            marker_id += 1
            
            # 9. Leo velocity vector (arrow) - MAGENTA
            if self.leo_vel is not None:
                vel_mag = np.linalg.norm(self.leo_vel[:2])
                if vel_mag > 0.05:  # Only show if moving
                    marker_array.markers.append(self.create_velocity_arrow(
                        marker_id, self.leo_pos, self.leo_vel,
                        color=[1.0, 0.0, 1.0, 1.0], ns="leo_velocity"
                    ))
                    marker_id += 1
        
        # 10. Leo trajectory (line strip) - YELLOW
        if len(self.leo_trajectory_points) > 1:
            marker_array.markers.append(self.create_trajectory_marker(
                marker_id, self.leo_trajectory_points,
                color=[1.0, 0.8, 0.0, 0.8], ns="leo_trajectory"
            ))
            marker_id += 1
        
        # Publish
        self.marker_pub.publish(marker_array)

    def create_boundary_box(self, marker_id):
        """Create wireframe box for boundaries"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "boundary"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # 8 corners
        x_min, x_max = self.boundary_x_min, self.boundary_x_max
        y_min, y_max = self.boundary_y_min, self.boundary_y_max
        z_min, z_max = self.boundary_z_min, self.boundary_z_max
        
        corners = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ]
        
        # 12 edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical
        ]
        
        from geometry_msgs.msg import Point
        for edge in edges:
            p1 = Point()
            p1.x, p1.y, p1.z = corners[edge[0]]
            p2 = Point()
            p2.x, p2.y, p2.z = corners[edge[1]]
            marker.points.append(p1)
            marker.points.append(p2)
        
        return marker

    def create_safe_zone(self, marker_id):
        """Create transparent sphere for safe zone"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safe_zone"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.center_x
        marker.pose.position.y = self.center_y
        marker.pose.position.z = self.center_z
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = self.safe_zone_radius * 2
        marker.scale.y = self.safe_zone_radius * 2
        marker.scale.z = self.safe_zone_radius * 2
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.2
        
        return marker

    def create_center_marker(self, marker_id):
        """Create marker for center point"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "center"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.center_x
        marker.pose.position.y = self.center_y
        marker.pose.position.z = self.center_z
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        return marker

    def create_corner_marker(self, marker_id, x, y, name):
        """Create marker for corner targets"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "corners"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = self.center_z
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.7
        
        return marker

    def create_uav_marker(self, marker_id):
        """Create marker for UAV position - color based on boundary proximity"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "uav"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.uav_pos[0]
        marker.pose.position.y = self.uav_pos[1]
        marker.pose.position.z = self.uav_pos[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        
        # Color based on boundary distance
        dist = self.calculate_min_boundary_distance(self.uav_pos)
        
        if dist < 0:  # Violation - RED
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        elif dist < 0.15:  # Emergency - ORANGE
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        elif dist < 0.3:  # Warning - YELLOW
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        else:  # Safe - GREEN
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        
        marker.color.a = 1.0
        
        return marker

    def create_leo_marker(self, marker_id):
        """Create marker for Leo position - ORANGE"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "leo"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.leo_pos[0]
        marker.pose.position.y = self.leo_pos[1]
        marker.pose.position.z = self.leo_pos[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Orange color for Leo
        marker.color.r = 1.0
        marker.color.g = 0.6
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker

    def create_velocity_arrow(self, marker_id, pos, vel, color, ns):
        """Create arrow marker for velocity vector"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Arrow from position to position + velocity
        from geometry_msgs.msg import Point
        start = Point()
        start.x, start.y, start.z = pos[0], pos[1], pos[2]
        
        end = Point()
        end.x = pos[0] + vel[0] * self.velocity_arrow_scale
        end.y = pos[1] + vel[1] * self.velocity_arrow_scale
        end.z = pos[2] + vel[2] * self.velocity_arrow_scale
        
        marker.points.append(start)
        marker.points.append(end)
        
        # Arrow dimensions
        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1   # Head diameter
        marker.scale.z = 0.15  # Head length
        
        # Color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker

    def create_trajectory_marker(self, marker_id, trajectory_points, color, ns):
        """Create line strip for trajectory"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.03  # Line width
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        from geometry_msgs.msg import Point
        for point in trajectory_points:
            p = Point()
            p.x, p.y, p.z = point
            marker.points.append(p)
        
        return marker

    def calculate_min_boundary_distance(self, pos):
        """Calculate minimum distance to any boundary"""
        x, y, z = pos
        dist_x_min = x - self.boundary_x_min
        dist_x_max = self.boundary_x_max - x
        dist_y_min = y - self.boundary_y_min
        dist_y_max = self.boundary_y_max - y
        dist_z_min = z - self.boundary_z_min
        dist_z_max = self.boundary_z_max - z
        
        return min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)


def main(args=None):
    rclpy.init(args=args)
    node = BoundaryVisualization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()