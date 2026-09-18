#!/usr/bin/env python3
"""
Boundary Safety Test Node - Independent boundary enforcement
Tests boundary logic by systematically approaching all 4 corners
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np


class BoundarySafetyTest(Node):
    def __init__(self):
        super().__init__('boundary_safety_test')
        
        # Boundary parameters
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        self.declare_parameter('boundary_z_min', 0.8)
        self.declare_parameter('boundary_z_max', 2.0)
        self.declare_parameter('safe_zone_radius', 0.5)  # Radius of "safe return zone"
        self.declare_parameter('warning_distance', 0.3)   # Start warning this far from boundary
        self.declare_parameter('emergency_distance', 0.15) # Force return this close to boundary
        self.declare_parameter('approach_speed', 0.2)     # Speed when approaching boundary
        
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        self.boundary_z_min = self.get_parameter('boundary_z_min').value
        self.boundary_z_max = self.get_parameter('boundary_z_max').value
        self.safe_zone_radius = self.get_parameter('safe_zone_radius').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.emergency_distance = self.get_parameter('emergency_distance').value
        self.approach_speed = self.get_parameter('approach_speed').value
        
        # Calculate safe zone center (origin)
        self.safe_center_x = (self.boundary_x_min + self.boundary_x_max) / 2.0
        self.safe_center_y = (self.boundary_y_min + self.boundary_y_max) / 2.0
        self.safe_center_z = (self.boundary_z_min + self.boundary_z_max) / 2.0
        
        # Define 4 corner target points (slightly inside boundaries for safety)
        margin = 0.05  # Small margin so we aim just inside boundaries
        self.corner_targets = [
            (self.boundary_x_min + margin, self.boundary_y_min + margin, "Bottom-Left"),
            (self.boundary_x_max - margin, self.boundary_y_min + margin, "Bottom-Right"),
            (self.boundary_x_max - margin, self.boundary_y_max - margin, "Top-Right"),
            (self.boundary_x_min + margin, self.boundary_y_max - margin, "Top-Left"),
        ]
        
        self.current_target_index = 0
        self.test_phase = "RETURN_TO_CENTER"  # States: RETURN_TO_CENTER, APPROACH_CORNER, EMERGENCY_RETURN
        self.corner_tests_completed = 0
        self.total_violations = 0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("BOUNDARY SAFETY TEST NODE - 4 CORNER TEST")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Boundaries: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] "
                              f"Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}] "
                              f"Z[{self.boundary_z_min:.1f}, {self.boundary_z_max:.1f}]")
        self.get_logger().info(f"Safe center: ({self.safe_center_x:.1f}, {self.safe_center_y:.1f}, {self.safe_center_z:.1f})")
        self.get_logger().info(f"Warning distance: {self.warning_distance}m")
        self.get_logger().info(f"Emergency distance: {self.emergency_distance}m")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Test Pattern:")
        for i, (x, y, name) in enumerate(self.corner_targets):
            self.get_logger().info(f"  Corner {i+1}: {name} ({x:.2f}, {y:.2f})")
        self.get_logger().info("=" * 60)
        
        # State
        self.drone_pos = None
        self.current_state = State()
        self.offboard_enabled = False
        self.armed = False
        self.boundary_violations = 0
        self.in_emergency_return = False
        
        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.sub_drone_pose = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.drone_pose_cb, qos_profile)
        self.sub_state = self.create_subscription(
            State, '/mavros/state', self.state_cb, 10)
        
        # Publishers
        self.vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        
        # Services
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        
        self.get_logger().info("Waiting for MAVROS services...")
        self.arming_client.wait_for_service(timeout_sec=5.0)
        self.set_mode_client.wait_for_service(timeout_sec=5.0)
        
        # Timers
        self.offboard_setpoint_counter = 0
        self.init_timer = self.create_timer(0.05, self.initialize_offboard)
        self.control_timer = None
        
        self.get_logger().info("Node ready. Initializing OFFBOARD mode...")

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

    def initialize_offboard(self):
        """Send setpoints before enabling OFFBOARD"""
        cmd = Twist()
        cmd.linear.z = 0.0
        self.vel_pub.publish(cmd)
        
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
                self.get_logger().info("✓ OFFBOARD + ARMED. Starting 4-corner boundary test...")
                self.get_logger().info(f"→ Phase 1: Flying to center first")
                self.init_timer.cancel()
                self.control_timer = self.create_timer(0.05, self.test_boundaries)

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

    def test_boundaries(self):
        """Main boundary test loop - cycles through 4 corners"""
        cmd = Twist()
        
        # Safety check
        if not self.offboard_enabled or not self.armed:
            self.get_logger().error("Lost OFFBOARD or disarmed!", throttle_duration_sec=2.0)
            self.vel_pub.publish(Twist())
            return
        
        if self.drone_pos is None:
            self.get_logger().warn("No drone position", throttle_duration_sec=1.0)
            self.vel_pub.publish(Twist())
            return
        
        # Get current target corner
        target_x, target_y, corner_name = self.corner_targets[self.current_target_index]
        
        # Get boundary status
        status, direction_to_center, min_boundary_dist, dist_to_center = \
            self.calculate_boundary_status(self.drone_pos)
        
        # Calculate direction to target corner
        direction_to_target = np.array([
            target_x - self.drone_pos[0],
            target_y - self.drone_pos[1]
        ])
        dist_to_target = np.linalg.norm(direction_to_target)
        
        if dist_to_target > 0.01:
            direction_to_target_norm = direction_to_target / dist_to_target
        else:
            direction_to_target_norm = np.array([0.0, 0.0])
        
        # Log status periodically
        if not hasattr(self, '_last_status_log') or \
           (self.get_clock().now().nanoseconds / 1e9 - self._last_status_log) > 1.0:
            self._last_status_log = self.get_clock().now().nanoseconds / 1e9
            
            x, y, z = self.drone_pos
            self.get_logger().info(
                f"[{self.test_phase}] [{status}] "
                f"Pos: ({x:.2f}, {y:.2f}) | "
                f"Target: {corner_name} ({target_x:.2f}, {target_y:.2f}) | "
                f"Dist to target: {dist_to_target:.2f}m | "
                f"Boundary dist: {min_boundary_dist:.2f}m | "
                f"Tests: {self.corner_tests_completed}/4 | "
                f"Violations: {self.total_violations}"
            )
        
        # ═══════════════════════════════════════════════════════════
        # STATE MACHINE FOR CORNER TESTING
        # ═══════════════════════════════════════════════════════════
        
        if status == 'VIOLATION' or status == 'EMERGENCY':
            # EMERGENCY: Boundary triggered - activate safety return
            if not self.in_emergency_return:
                self.boundary_violations += 1
                self.total_violations += 1
                self.in_emergency_return = True
                self.test_phase = "EMERGENCY_RETURN"
                
                self.get_logger().warn(
                    f"🚨 BOUNDARY TRIGGERED at {corner_name}! "
                    f"Violation #{self.boundary_violations} - Returning to center",
                    throttle_duration_sec=0.5
                )
            
            # Strong return velocity toward center
            return_speed = 0.4 if status == 'VIOLATION' else 0.3
            cmd.linear.x = float(direction_to_center[0] * return_speed)
            cmd.linear.y = float(direction_to_center[1] * return_speed)
            cmd.linear.z = float(direction_to_center[2] * 0.2)
            cmd.angular.z = 0.0
            
        elif self.test_phase == "EMERGENCY_RETURN":
            # Returning to center after emergency
            if dist_to_center < self.safe_zone_radius:
                # Reached center - move to next corner
                self.in_emergency_return = False
                self.test_phase = "RETURN_TO_CENTER"
                self.corner_tests_completed += 1
                
                self.get_logger().info(
                    f"✓ Returned to safe zone. Corner test {self.corner_tests_completed}/4 complete!"
                )
                
                # Move to next corner
                self.current_target_index = (self.current_target_index + 1) % 4
                next_corner = self.corner_targets[self.current_target_index][2]
                
                if self.corner_tests_completed >= 4:
                    self.get_logger().info("=" * 60)
                    self.get_logger().info(f"🎉 ALL 4 CORNERS TESTED!")
                    self.get_logger().info(f"Total boundary violations: {self.total_violations}")
                    self.get_logger().info("Test complete. Hovering at center.")
                    self.get_logger().info("=" * 60)
                    # Reset to test again
                    self.corner_tests_completed = 0
                    self.current_target_index = 0
                
                self.get_logger().info(f"→ Next target: {next_corner}")
                
            # Continue return
            return_speed = 0.2
            cmd.linear.x = float(direction_to_center[0] * return_speed)
            cmd.linear.y = float(direction_to_center[1] * return_speed)
            cmd.linear.z = float(direction_to_center[2] * 0.1)
            cmd.angular.z = 0.0
            
        elif self.test_phase == "RETURN_TO_CENTER":
            # First ensure we're at center
            if dist_to_center < self.safe_zone_radius:
                self.test_phase = "APPROACH_CORNER"
                self.get_logger().info(f"→ Starting approach to {corner_name}")
            else:
                # Fly to center
                return_speed = 0.2
                cmd.linear.x = float(direction_to_center[0] * return_speed)
                cmd.linear.y = float(direction_to_center[1] * return_speed)
                cmd.linear.z = 0.0
                cmd.angular.z = 0.0
                
        elif self.test_phase == "APPROACH_CORNER":
            # Fly toward target corner until boundary is triggered
            cmd.linear.x = float(direction_to_target_norm[0] * self.approach_speed)
            cmd.linear.y = float(direction_to_target_norm[1] * self.approach_speed)
            cmd.linear.z = 0.0
            cmd.angular.z = 0.0
            
            if status == 'WARNING':
                self.get_logger().warn(
                    f"⚡ Approaching {corner_name} boundary! Dist: {min_boundary_dist:.2f}m",
                    throttle_duration_sec=0.5
                )
        
        # Altitude control (always active)
        if self.drone_pos[2] < self.safe_center_z - 0.2:
            cmd.linear.z = max(cmd.linear.z, 0.2)
        elif self.drone_pos[2] > self.safe_center_z + 0.2:
            cmd.linear.z = min(cmd.linear.z, -0.2)
        
        # Publish
        self.vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = BoundarySafetyTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop drone
        node.vel_pub.publish(Twist())
        node.get_logger().info("Test stopped. Drone commanded to stop.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()