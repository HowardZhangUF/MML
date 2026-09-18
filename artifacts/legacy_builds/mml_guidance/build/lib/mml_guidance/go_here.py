import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
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


class GPOrigin(Node):
    """Waypoint Navigation with Boundary Safety"""

    def __init__(self):
        super().__init__("gp_origin")

        # ═══════════════════════════════════════════════════════════
        # BOUNDARY SAFETY PARAMETERS (from boundary_safety_test.py)
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
        # STATE TRACKING
        # ═══════════════════════════════════════════════════════════
        self.drone_pos = None
        self.current_state = State()
        self.offboard_enabled = False
        self.armed = False
        self.in_emergency_return = False
        self.boundary_violations = 0
        
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
        
        # ═══════════════════════════════════════════════════════════
        # SERVICES
        # ═══════════════════════════════════════════════════════════
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        
        self.get_logger().info("Waiting for MAVROS services...")
        self.arming_client.wait_for_service(timeout_sec=5.0)
        self.set_mode_client.wait_for_service(timeout_sec=5.0)
        
        # ═══════════════════════════════════════════════════════════
        # WAYPOINT SEQUENCE
        # ═══════════════════════════════════════════════════════════
        self.waypoint_values = [-1., -0.5, 0., 0.5, 1.]
        self.ori_values = [-90., 0., 90., 180., 0.]
        self.index = 0
        self.increase = 1
        
        # ═══════════════════════════════════════════════════════════
        # INITIALIZATION
        # ═══════════════════════════════════════════════════════════
        self.offboard_setpoint_counter = 0
        self.init_timer = self.create_timer(0.05, self.initialize_offboard)
        self.waypoint_timer = None
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("WAYPOINT NAVIGATION WITH BOUNDARY SAFETY")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Boundaries: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] "
                              f"Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}] "
                              f"Z[{self.boundary_z_min:.1f}, {self.boundary_z_max:.1f}]")
        self.get_logger().info(f"Safe center: ({self.safe_center_x:.1f}, {self.safe_center_y:.1f}, {self.safe_center_z:.1f})")
        self.get_logger().info(f"Warning distance: {self.warning_distance}m")
        self.get_logger().info(f"Emergency distance: {self.emergency_distance}m")
        self.get_logger().info("=" * 60)

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
    # BOUNDARY SAFETY SYSTEM (from boundary_safety_test.py)
    # ═══════════════════════════════════════════════════════════════
    
    def calculate_boundary_status(self, pos):
        """
        Calculate how close drone is to boundaries
        Returns: (status, direction_to_safe_center, closest_boundary_distance, distance_to_center)
        """
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
        
        distance_to_center = np.linalg.norm(direction[:2])  # XY distance only
        
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
                # Recovered from emergency
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
        # Send initial position
        self.gp_origin_publisher(0.0, 0.0, 1.4, ori_deg=0.0)
        
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
                self.get_logger().info("✓ OFFBOARD + ARMED. Starting waypoint navigation...")
                self.init_timer.cancel()
                # Start waypoint navigation at 1 Hz
                self.waypoint_timer = self.create_timer(1.0, self.gp_timer_callback)

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
    
    def gp_timer_callback(self):
        """Cycle through waypoints"""
        # Skip waypoint update if in emergency return
        if self.in_emergency_return:
            self.get_logger().warn(
                "Emergency return active - waypoint update paused",
                throttle_duration_sec=2.0
            )
            return
        
        # Safety check
        if not self.offboard_enabled or not self.armed:
            self.get_logger().error("Lost OFFBOARD or disarmed!", throttle_duration_sec=2.0)
            return
        
        x_val = self.waypoint_values[self.index]
        y_val = self.waypoint_values[self.index]
        ori_val = self.ori_values[self.index]
        
        self.get_logger().info(
            f"→ Waypoint {self.index + 1}/5: ({x_val:.1f}, {y_val:.1f}) @ {ori_val:.0f}°"
        )
        
        self.gp_origin_publisher(x_val, y_val, self.safe_center_z, ori_deg=ori_val)
        
        # Update index
        if self.index >= len(self.waypoint_values) - 1:
            self.increase = -1
        elif self.index == 0:
            self.increase = 1
        self.index += self.increase
        
    def gp_origin_publisher(self, x, y, z, ori_deg):
        """Publish position setpoint"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = 0.
        msg.pose.orientation.y = 0.
        msg.pose.orientation.z = euler_to_quaternion(deg_to_rad(ori_deg))[0]
        msg.pose.orientation.w = euler_to_quaternion(deg_to_rad(ori_deg))[1]
        self.position_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GPOrigin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop drone
        node.vel_pub.publish(Twist())
        node.get_logger().info("Waypoint navigation stopped.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()