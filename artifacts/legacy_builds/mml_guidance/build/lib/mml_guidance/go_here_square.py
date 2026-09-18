import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


def quaternion_to_euler(z, w):
    t0 = +2.0 * (w * z + 0.0 * 0.0)
    t1 = +1.0 - 2.0 * (0.0 * 0.0 + z * z)
    euler_z = math.atan2(t0, t1)
    return euler_z


def euler_to_quaternion(z):
    w = math.cos(z / 2)
    z = math.sin(z / 2)
    return z, w


def deg_to_rad(deg):
    return deg * np.pi / 180


class PatternWaypointNavigator(Node):
    """Multi-Pattern Waypoint Navigation with Boundary Safety (Square, Circle, Figure-8)"""

    def __init__(self):
        super().__init__("pattern_waypoint_navigator")

        # ═══════════════════════════════════════════════════════════
        # WAYPOINT PATTERN PARAMETERS
        # ═══════════════════════════════════════════════════════════
        self.declare_parameter('pattern_type', 'square')  # 'square', 'circle', or 'figure8'
        self.declare_parameter('waypoint_update_rate', 1.0)  # Hz - frequency of waypoint updates
        self.declare_parameter('pattern_size', 1.5)  # meters - size of pattern (side/diameter)
        self.declare_parameter('pattern_center_x', 0.0)
        self.declare_parameter('pattern_center_y', 0.0)
        self.declare_parameter('flight_altitude', 1.4)  # meters
        self.declare_parameter('face_direction', True)  # Whether to orient towards travel direction
        self.declare_parameter('continuous_loop', True)  # Keep looping through waypoints
        self.declare_parameter('waypoints_per_loop', 40)  # Total waypoints in complete pattern
        
        self.pattern_type = self.get_parameter('pattern_type').value
        self.waypoint_rate = self.get_parameter('waypoint_update_rate').value
        self.pattern_size = self.get_parameter('pattern_size').value
        self.center_x = self.get_parameter('pattern_center_x').value
        self.center_y = self.get_parameter('pattern_center_y').value
        self.altitude = self.get_parameter('flight_altitude').value
        self.face_direction = self.get_parameter('face_direction').value
        self.continuous_loop = self.get_parameter('continuous_loop').value
        self.waypoints_per_loop = self.get_parameter('waypoints_per_loop').value

        # ═══════════════════════════════════════════════════════════
        # BOUNDARY SAFETY PARAMETERS
        # ═══════════════════════════════════════════════════════════
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        self.declare_parameter('boundary_z_min', 0.8)
        self.declare_parameter('boundary_z_max', 2.0)
        self.declare_parameter('safe_zone_radius', 0.5)
        self.declare_parameter('warning_distance', 0.3)
        self.declare_parameter('emergency_distance', 0.15)
        self.declare_parameter('boundary_return_speed', 0.3)
        
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        self.boundary_z_min = self.get_parameter('boundary_z_min').value
        self.boundary_z_max = self.get_parameter('boundary_z_max').value
        self.safe_zone_radius = self.get_parameter('safe_zone_radius').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.emergency_distance = self.get_parameter('emergency_distance').value
        self.boundary_return_speed = self.get_parameter('boundary_return_speed').value
        
        # Calculate safe center
        self.safe_center_x = (self.boundary_x_min + self.boundary_x_max) / 2.0
        self.safe_center_y = (self.boundary_y_min + self.boundary_y_max) / 2.0
        self.safe_center_z = (self.boundary_z_min + self.boundary_z_max) / 2.0
        
        # ═══════════════════════════════════════════════════════════
        # GENERATE PATTERN WAYPOINTS
        # ═══════════════════════════════════════════════════════════
        self.generate_waypoints()
        
        # ═══════════════════════════════════════════════════════════
        # STATE TRACKING
        # ═══════════════════════════════════════════════════════════
        self.drone_pos = None
        self.current_state = State()
        self.offboard_enabled = False
        self.armed = False
        self.in_emergency_return = False
        self.boundary_violations = 0
        self.waypoint_index = 0
        self.mission_complete = False
        
        # ═══════════════════════════════════════════════════════════
        # QOS AND SUBSCRIBERS
        # ═══════════════════════════════════════════════════════════
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.sub_drone_pose = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.drone_pose_cb, qos_profile)
        self.sub_state = self.create_subscription(
            State, '/mavros/state', self.state_cb, 10)
        
        # ═══════════════════════════════════════════════════════════
        # PUBLISHERS
        # ═══════════════════════════════════════════════════════════
        self.position_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        
        # Waypoint Visualization Publishers
        self.waypoint_marker_pub = self.create_publisher(
            Marker, '/waypoint/current', 10)
        self.waypoint_path_pub = self.create_publisher(
            MarkerArray, '/waypoint/path', 10)
        self.waypoint_arrow_pub = self.create_publisher(
            Marker, '/waypoint/arrow', 10)
        self.waypoint_text_pub = self.create_publisher(
            Marker, '/waypoint/distance', 10)
        
        # ═══════════════════════════════════════════════════════════
        # SERVICES
        # ═══════════════════════════════════════════════════════════
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        
        self.get_logger().info("Waiting for MAVROS services...")
        self.arming_client.wait_for_service(timeout_sec=5.0)
        self.set_mode_client.wait_for_service(timeout_sec=5.0)
        
        # ═══════════════════════════════════════════════════════════
        # INITIALIZATION
        # ═══════════════════════════════════════════════════════════
        self.offboard_setpoint_counter = 0
        self.init_timer = self.create_timer(0.05, self.initialize_offboard)
        self.waypoint_timer = None
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"PATTERN WAYPOINT NAVIGATION: {self.pattern_type.upper()}")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Pattern type: {self.pattern_type}")
        self.get_logger().info(f"Pattern size: {self.pattern_size}m")
        self.get_logger().info(f"Center: ({self.center_x:.1f}, {self.center_y:.1f})")
        self.get_logger().info(f"Altitude: {self.altitude}m")
        self.get_logger().info(f"Waypoint update rate: {self.waypoint_rate} Hz")
        self.get_logger().info(f"Waypoints per loop: {self.waypoints_per_loop}")
        self.get_logger().info(f"Total waypoints: {len(self.waypoints)}")
        self.get_logger().info(f"Continuous loop: {self.continuous_loop}")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Boundaries: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] "
                              f"Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}] "
                              f"Z[{self.boundary_z_min:.1f}, {self.boundary_z_max:.1f}]")
        self.get_logger().info("=" * 60)
        
        # Publish complete waypoint path for visualization
        self.publish_waypoint_path()

    def generate_waypoints(self):
        """Dispatcher: Generate waypoints based on pattern type"""
        pattern_generators = {
            'square': self.generate_square_waypoints,
            'circle': self.generate_circle_waypoints,
            'figure8': self.generate_figure8_waypoints
        }
        
        if self.pattern_type not in pattern_generators:
            self.get_logger().error(
                f"Unknown pattern type: {self.pattern_type}. "
                f"Valid options: {list(pattern_generators.keys())}"
            )
            self.get_logger().info("Defaulting to 'square' pattern")
            self.pattern_type = 'square'
        
        pattern_generators[self.pattern_type]()
        
        # Log summary
        total_time = self.waypoints_per_loop / self.waypoint_rate
        self.get_logger().info("="*60)
        self.get_logger().info("WAYPOINT GENERATION SUMMARY")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Pattern: {self.pattern_type.upper()}")
        self.get_logger().info(f"Total waypoints: {len(self.waypoints)}")
        self.get_logger().info(f"Update rate: {self.waypoint_rate} Hz")
        self.get_logger().info(f"Complete loop time: {total_time:.1f}s")
        self.get_logger().info("="*60)
        
        # Log sample waypoints
        self.get_logger().info("Sample waypoints:")
        sample_indices = [0, len(self.waypoints)//4, len(self.waypoints)//2, 3*len(self.waypoints)//4]
        for idx in sample_indices:
            if idx < len(self.waypoints):
                wp = self.waypoints[idx]
                self.get_logger().info(
                    f"  WP {idx+1}/{len(self.waypoints)}: ({wp['x']:.3f}, {wp['y']:.3f}) @ {wp['ori_deg']:.0f}°"
                )

    def generate_square_waypoints(self):
        """Generate waypoints for a square pattern"""
        half_size = self.pattern_size / 2.0
        
        # Define corners of square (counter-clockwise from bottom-left)
        corners = [
            (self.center_x - half_size, self.center_y - half_size),  # Bottom-left
            (self.center_x + half_size, self.center_y - half_size),  # Bottom-right
            (self.center_x + half_size, self.center_y + half_size),  # Top-right
            (self.center_x - half_size, self.center_y + half_size),  # Top-left
        ]
        
        self.waypoints = []
        
        # Distribute waypoints evenly across all 4 sides
        for i in range(self.waypoints_per_loop):
            t = i / float(self.waypoints_per_loop)  # [0, 1)
            
            # Which side (0-3) and position within side
            segment = int(t * 4)
            if segment >= 4:  # Handle edge case
                segment = 3
            local_t = (t * 4) - segment  # [0, 1)
            
            # Get current and next corner
            curr_x, curr_y = corners[segment]
            next_x, next_y = corners[(segment + 1) % 4]
            
            # Linear interpolation
            x = curr_x + local_t * (next_x - curr_x)
            y = curr_y + local_t * (next_y - curr_y)
            
            # Calculate orientation
            if self.face_direction:
                dx = next_x - curr_x
                dy = next_y - curr_y
                ori_rad = math.atan2(dy, dx)
                ori_deg = math.degrees(ori_rad)
            else:
                ori_deg = 0.0
            
            self.waypoints.append({
                'x': x,
                'y': y,
                'z': self.altitude,
                'ori_deg': ori_deg,
                'index': i
            })
    
    def generate_circle_waypoints(self):
        """Generate waypoints for a circular pattern"""
        radius = self.pattern_size / 2.0
        
        self.waypoints = []
        
        for i in range(self.waypoints_per_loop):
            # Angle around circle (0 to 2π)
            theta = (i / float(self.waypoints_per_loop)) * 2.0 * np.pi
            
            # Parametric circle equations
            x = self.center_x + radius * np.cos(theta)
            y = self.center_y + radius * np.sin(theta)
            
            # Orientation: tangent to circle (perpendicular to radius)
            if self.face_direction:
                # Tangent direction is θ + 90°
                ori_rad = theta + np.pi / 2.0
                ori_deg = math.degrees(ori_rad)
            else:
                ori_deg = 0.0
            
            self.waypoints.append({
                'x': x,
                'y': y,
                'z': self.altitude,
                'ori_deg': ori_deg,
                'index': i
            })
    
    def generate_figure8_waypoints(self):
        """Generate waypoints for a figure-8 (lemniscate) pattern"""
        scale = self.pattern_size / 2.0
        
        self.waypoints = []
        
        for i in range(self.waypoints_per_loop):
            # Parameter t from 0 to 2π
            t = (i / float(self.waypoints_per_loop)) * 2.0 * np.pi
            
            # Lemniscate of Gerono (figure-8 curve)
            # x = a * sin(t)
            # y = a * sin(t) * cos(t) = a * sin(2t) / 2
            x = self.center_x + scale * np.sin(t)
            y = self.center_y + scale * np.sin(t) * np.cos(t)
            
            # Calculate orientation from velocity direction
            if self.face_direction:
                # Derivatives: dx/dt, dy/dt
                dx_dt = scale * np.cos(t)
                dy_dt = scale * (np.cos(t)**2 - np.sin(t)**2)  # d/dt[sin(t)*cos(t)]
                ori_rad = math.atan2(dy_dt, dx_dt)
                ori_deg = math.degrees(ori_rad)
            else:
                ori_deg = 0.0
            
            self.waypoints.append({
                'x': x,
                'y': y,
                'z': self.altitude,
                'ori_deg': ori_deg,
                'index': i
            })

    def state_cb(self, msg):
        self.current_state = msg
        self.offboard_enabled = (msg.mode == "GUIDED")
        self.armed = msg.armed

    def drone_pose_cb(self, msg):
        self.drone_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Check boundary safety continuously
        if self.drone_pos is not None and self.waypoint_timer is not None:
            self.check_boundary_safety()

    # ═══════════════════════════════════════════════════════════════
    # BOUNDARY SAFETY SYSTEM
    # ═══════════════════════════════════════════════════════════════
    
    def calculate_boundary_status(self, pos):
        """Calculate how close drone is to boundaries"""
        x, y, z = pos
        
        # Calculate distance to each boundary
        dist_x_min = x - self.boundary_x_min
        dist_x_max = self.boundary_x_max - x
        dist_y_min = y - self.boundary_y_min
        dist_y_max = self.boundary_y_max - y
        dist_z_min = z - self.boundary_z_min
        dist_z_max = self.boundary_z_max - z
        
        # Find minimum distance to any boundary
        min_dist = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)
        
        # Determine status
        if min_dist < 0:
            status = 'VIOLATION'
        elif min_dist < self.emergency_distance:
            status = 'EMERGENCY'
        elif min_dist < self.warning_distance:
            status = 'WARNING'
        else:
            status = 'SAFE'
        
        # Calculate direction to safe center
        direction = np.array([
            self.safe_center_x - x,
            self.safe_center_y - y,
            self.safe_center_z - z
        ])
        
        distance_to_center = np.linalg.norm(direction[:2])
        
        if distance_to_center > 0.01:
            direction_norm = direction / np.linalg.norm(direction)
        else:
            direction_norm = np.array([0.0, 0.0, 0.0])
        
        return status, direction_norm, min_dist, distance_to_center
    
    def check_boundary_safety(self):
        """Continuously monitor and enforce boundary safety"""
        if self.drone_pos is None:
            return
        
        status, direction_to_center, min_dist, dist_to_center = \
            self.calculate_boundary_status(self.drone_pos)
        
        if status == 'VIOLATION' or status == 'EMERGENCY':
            if not self.in_emergency_return:
                self.boundary_violations += 1
                self.in_emergency_return = True
                
                self.get_logger().warn(
                    f"🚨 BOUNDARY {status}! Distance: {min_dist:.2f}m - "
                    f"Activating emergency return (violation #{self.boundary_violations})"
                )
            
            # Override with emergency return velocity
            cmd = Twist()
            return_speed = 0.4 if status == 'VIOLATION' else self.boundary_return_speed
            cmd.linear.x = float(direction_to_center[0] * return_speed)
            cmd.linear.y = float(direction_to_center[1] * return_speed)
            cmd.linear.z = float(direction_to_center[2] * 0.2)
            self.vel_pub.publish(cmd)
            
        elif status == 'WARNING':
            self.get_logger().warn(
                f"⚡ WARNING! Approaching boundary: {min_dist:.2f}m",
                throttle_duration_sec=1.0
            )
            
        else:  # SAFE
            if self.in_emergency_return:
                if dist_to_center < self.safe_zone_radius:
                    self.in_emergency_return = False
                    self.get_logger().info(
                        f"✓ Returned to safe zone. Resuming waypoint navigation."
                    )

    # ═══════════════════════════════════════════════════════════════
    # OFFBOARD INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    
    def initialize_offboard(self):
        """Send setpoints before enabling OFFBOARD"""
        # Send initial position (first waypoint)
        wp = self.waypoints[0]
        self.publish_waypoint(wp)
        
        self.offboard_setpoint_counter += 1
        
        if self.offboard_setpoint_counter > 40:
            if not self.offboard_enabled:
                if self.drone_pos is None:
                    self.get_logger().warn("Waiting for drone pose...", throttle_duration_sec=2.0)
                    return
                self.enable_offboard_mode()
            elif not self.armed:
                self.arm_vehicle()
            else:
                self.get_logger().info("✓ OFFBOARD + ARMED. Starting square pattern...")
                self.init_timer.cancel()
                # Start waypoint navigation at configured rate
                timer_period = 1.0 / self.waypoint_rate
                self.waypoint_timer = self.create_timer(timer_period, self.waypoint_callback)

    def enable_offboard_mode(self):
        req = SetMode.Request()
        req.custom_mode = "GUIDED"
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("GUIDED mode enabled"))

    def arm_vehicle(self):
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("Vehicle armed"))

    # ═══════════════════════════════════════════════════════════════
    # WAYPOINT NAVIGATION
    # ═══════════════════════════════════════════════════════════════
    
    def waypoint_callback(self):
        """Update waypoint at configured frequency"""
        # Skip if in emergency return
        if self.in_emergency_return:
            self.get_logger().warn(
                "Emergency return active - waypoint update paused",
                throttle_duration_sec=2.0
            )
            return
        
        # Check if mission complete
        if self.mission_complete:
            return
        
        # Safety check
        if not self.offboard_enabled or not self.armed:
            self.get_logger().error("Lost OFFBOARD or disarmed!", throttle_duration_sec=2.0)
            return
        
        # Get current waypoint
        wp = self.waypoints[self.waypoint_index]
        
        self.get_logger().info(
            f"→ WP {self.waypoint_index + 1}/{len(self.waypoints)}: "
            f"({wp['x']:.3f}, {wp['y']:.3f}) @ {wp['ori_deg']:.0f}°"
        )
        
        self.publish_waypoint(wp)
        
        # Visualize current waypoint and navigation
        self.visualize_current_waypoint(wp)
        self.visualize_direction_arrow(wp)
        self.visualize_distance_text(wp)
        
        # Move to next waypoint
        self.waypoint_index += 1
        
        # Check if we've completed the square
        if self.waypoint_index >= len(self.waypoints):
            if self.continuous_loop:
                self.waypoint_index = 0
                self.get_logger().info("🔄 Completed square, starting new loop...")
            else:
                self.mission_complete = True
                self.get_logger().info("✓ Square pattern complete!")
                self.waypoint_timer.cancel()
    
    def publish_waypoint(self, waypoint):
        """Publish waypoint to MAVROS"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = waypoint['x']
        msg.pose.position.y = waypoint['y']
        msg.pose.position.z = waypoint['z']
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = euler_to_quaternion(deg_to_rad(waypoint['ori_deg']))[0]
        msg.pose.orientation.w = euler_to_quaternion(deg_to_rad(waypoint['ori_deg']))[1]
        self.position_pub.publish(msg)
    
    # ═══════════════════════════════════════════════════════════════
    # WAYPOINT VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    
    def visualize_current_waypoint(self, waypoint):
        """Visualize the current target waypoint as a red sphere"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_waypoint"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = waypoint['x']
        marker.pose.position.y = waypoint['y']
        marker.pose.position.z = waypoint['z']
        marker.pose.orientation.w = 1.0
        
        # Scale (diameter)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        # Color: RED
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Lifetime (auto-delete if stale)
        marker.lifetime.sec = 3
        
        self.waypoint_marker_pub.publish(marker)
    
    def visualize_direction_arrow(self, waypoint):
        """Visualize arrow from drone to target waypoint"""
        if self.drone_pos is None:
            return
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "direction_arrow"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Arrow from drone to waypoint
        start_point = Point()
        start_point.x = float(self.drone_pos[0])
        start_point.y = float(self.drone_pos[1])
        start_point.z = float(self.drone_pos[2])
        
        end_point = Point()
        end_point.x = waypoint['x']
        end_point.y = waypoint['y']
        end_point.z = waypoint['z']
        
        marker.points = [start_point, end_point]
        
        # Arrow size
        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1   # Head diameter
        marker.scale.z = 0.1   # Head length
        
        # Color: YELLOW
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        # Lifetime
        marker.lifetime.sec = 3
        
        self.waypoint_arrow_pub.publish(marker)
    
    def visualize_distance_text(self, waypoint):
        """Visualize distance to waypoint as text label"""
        if self.drone_pos is None:
            return
        
        # Calculate distance
        dx = waypoint['x'] - self.drone_pos[0]
        dy = waypoint['y'] - self.drone_pos[1]
        dz = waypoint['z'] - self.drone_pos[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "distance_text"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position above waypoint
        marker.pose.position.x = waypoint['x']
        marker.pose.position.y = waypoint['y']
        marker.pose.position.z = waypoint['z'] + 0.3
        marker.pose.orientation.w = 1.0
        
        # Text content
        marker.text = f"{distance:.2f}m"
        
        # Text size
        marker.scale.z = 0.15
        
        # Color: WHITE
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        # Lifetime
        marker.lifetime.sec = 3
        
        self.waypoint_text_pub.publish(marker)
    
    def publish_waypoint_path(self):
        """Publish all waypoints as green spheres for path visualization"""
        marker_array = MarkerArray()
        
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoint_path"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = wp['x']
            marker.pose.position.y = wp['y']
            marker.pose.position.z = wp['z']
            marker.pose.orientation.w = 1.0
            
            # Scale (smaller than current waypoint)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Color: GREEN (semi-transparent)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            # Persistent (lifetime = 0)
            marker.lifetime.sec = 0
            
            marker_array.markers.append(marker)
        
        self.waypoint_path_pub.publish(marker_array)
        self.get_logger().info(f"📍 Published {len(self.waypoints)} waypoint path markers for RViz")


def main(args=None):
    rclpy.init(args=args)
    node = PatternWaypointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop drone
        node.vel_pub.publish(Twist())
        node.get_logger().info("Square pattern navigation stopped.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
