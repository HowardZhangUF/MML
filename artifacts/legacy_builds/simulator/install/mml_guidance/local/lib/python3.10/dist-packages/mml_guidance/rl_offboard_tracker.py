#!/usr/bin/env python3
"""
RL Offboard Tracker with ROBUST Boundary Safety
Priority: Boundary Safety > RL Policy
"""
import sys
sys.path.append('/home/basestation/mml_python_sim')
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np
import numpy
try:
    import numpy._core.numeric
except ImportError:
    print("Applying NumPy 2.0 -> 1.x Compatibility Patch...")
    if hasattr(numpy, 'core'):
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.numeric'] = numpy.core.numeric

from mml_guidance.guidance import Guidance
from mml_guidance.ParticleFilter import ParticleFilter

class RLOffboardTracker(Node):
    """
    Deploys PyGame-trained RL policy with ROBUST boundary safety
    """
    def __init__(self):
        super().__init__('rl_offboard_tracker')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_topic', '/howard/enu/pose')
        self.declare_parameter('drone_height', 1.0)
        self.declare_parameter('num_particles', 200)
        self.declare_parameter('prediction_method', 'Velocity')
        self.declare_parameter('max_vel', 0.5)
        
        # === ENHANCED BOUNDARY PARAMETERS (from boundary_safety_test.py) ===
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        self.declare_parameter('boundary_z_min', 0.8)
        self.declare_parameter('boundary_z_max', 2.0)
        self.declare_parameter('safe_zone_radius', 0.5)
        self.declare_parameter('warning_distance', 0.3)
        self.declare_parameter('emergency_distance', 0.15)
        self.declare_parameter('boundary_return_speed', 0.3)  # Speed when returning to center
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        target_topic = self.get_parameter('target_topic').get_parameter_value().string_value
        self.target_height = self.get_parameter('drone_height').get_parameter_value().double_value
        num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        prediction_method = self.get_parameter('prediction_method').get_parameter_value().string_value
        self.max_vel = self.get_parameter('max_vel').get_parameter_value().double_value
        
        # Boundary parameters
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
        
        # Boundary state tracking
        self.in_emergency_return = False
        self.boundary_violations = 0
        
        self.get_logger().info("=" * 60)


        self.get_logger().info("RL OFFBOARD TRACKER WITH ROBUST BOUNDARY SAFETY")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Target: {target_topic}")
        self.get_logger().info(f"Boundaries: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] "
                              f"Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}] "
                              f"Z[{self.boundary_z_min:.1f}, {self.boundary_z_max:.1f}]")
        self.get_logger().info(f"Safe center: ({self.safe_center_x:.1f}, {self.safe_center_y:.1f}, {self.safe_center_z:.1f})")
        self.get_logger().info(f"Warning distance: {self.warning_distance}m | Emergency distance: {self.emergency_distance}m")
        self.get_logger().info("PRIORITY: Boundary Safety > RL Policy")
        self.get_logger().info("=" * 60)

        # Initialize guidance system
        self.pf = ParticleFilter(
            num_particles=num_particles, 
            prediction_method=prediction_method, 
            drone_height=self.target_height
        )
        
        self.guidance = Guidance(
            guidance_mode="MultiPFInfoPPO_LSTM",
            prediction_method=prediction_method,
            drone_height=self.target_height,
            filter=self.pf,
            ppo_model_path=model_path
        )
        
        self.lstm_states = None
        self.episode_start = np.array([True])

        # MAVROS state
        self.current_state = State()
        self.offboard_enabled = False
        self.armed = False
        
        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_drone_pose = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.drone_pose_cb, qos_profile)
        self.sub_drone_vel = self.create_subscription(
            Twist, '/mavros/local_position/velocity_local', self.drone_vel_cb, qos_profile)
        self.sub_target = self.create_subscription(
            PoseStamped, target_topic, self.target_cb, qos_profile)
        self.sub_state = self.create_subscription(
            State, '/mavros/state', self.state_cb, 10)

        # Service clients
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        
        self.get_logger().info("Waiting for MAVROS services...")
        self.arming_client.wait_for_service(timeout_sec=5.0)
        self.set_mode_client.wait_for_service(timeout_sec=5.0)
        self.get_logger().info("MAVROS services ready")

        # Publisher
        self.vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # State storage
        self.drone_pos = None
        self.drone_orient = None
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.last_target_time = None
        self.raw_target_pos = None
        self.raw_target_vel = np.array([0.0, 0.0])
        
        # Offboard initialization
        self.offboard_setpoint_counter = 0
        self.init_timer = self.create_timer(0.05, self.initialize_offboard)
        self.control_timer = None
        
        self.get_logger().info("Starting OFFBOARD initialization...")

    # ═══════════════════════════════════════════════════════════════════
    # ROBUST BOUNDARY SAFETY SYSTEM (from boundary_safety_test.py)
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_boundary_status(self, pos):
        """
        Calculate boundary status (SAFE/WARNING/EMERGENCY/VIOLATION)
        Returns: (status, direction_to_center, min_boundary_dist, dist_to_center)
        """
        if pos is None:
            return 'UNKNOWN', np.array([0.0, 0.0, 0.0]), 0.0, 0.0
        
        x, y, z = pos
        
        # Distance to each boundary
        dist_x_min = x - self.boundary_x_min
        dist_x_max = self.boundary_x_max - x
        dist_y_min = y - self.boundary_y_min
        dist_y_max = self.boundary_y_max - y
        dist_z_min = z - self.boundary_z_min
        dist_z_max = self.boundary_z_max - z
        
        # Minimum distance to ANY boundary
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
        
        # Direction to safe center
        direction = np.array([
            self.safe_center_x - x,
            self.safe_center_y - y,
            self.safe_center_z - z
        ])
        
        dist_to_center = np.linalg.norm(direction[:2])  # XY distance
        
        if dist_to_center > 0.01:
            direction_norm = direction / np.linalg.norm(direction)
        else:
            direction_norm = np.array([0.0, 0.0, 0.0])
        
        return status, direction_norm, min_dist, dist_to_center

    def apply_boundary_safety(self, rl_cmd):
        """
        PRIORITY 1: Apply boundary safety to RL command
        This OVERRIDES RL policy if drone is near boundaries
        Returns: (safe_cmd, status_message)
        """
        if self.drone_pos is None:
            return rl_cmd, "NO_POSE"
        
        # Get boundary status
        status, dir_to_center, min_dist, dist_to_center = \
            self.calculate_boundary_status(self.drone_pos)
        
        safe_cmd = Twist()
        safe_cmd.linear.x = rl_cmd.linear.x
        safe_cmd.linear.y = rl_cmd.linear.y
        safe_cmd.linear.z = rl_cmd.linear.z
        safe_cmd.angular.z = rl_cmd.angular.z
        
        # ═══════════════════════════════════════════════════════
        # BOUNDARY SAFETY STATE MACHINE
        # ═══════════════════════════════════════════════════════
        
        if status == 'VIOLATION' or status == 'EMERGENCY':
            # CRITICAL: Force return to center
            if not self.in_emergency_return:
                self.boundary_violations += 1
                self.in_emergency_return = True
                self.get_logger().error(
                    f"🚨 BOUNDARY {status}! Violation #{self.boundary_violations} - "
                    f"Overriding RL policy, returning to center",
                    throttle_duration_sec=0.5
                )
            
            # OVERRIDE RL command completely
            return_speed = 0.4 if status == 'VIOLATION' else self.boundary_return_speed
            safe_cmd.linear.x = float(dir_to_center[0] * return_speed)
            safe_cmd.linear.y = float(dir_to_center[1] * return_speed)
            safe_cmd.linear.z = float(dir_to_center[2] * 0.2)
            safe_cmd.angular.z = 0.0
            
            return safe_cmd, f"EMERGENCY_RETURN (dist: {min_dist:.2f}m)"
        
        elif self.in_emergency_return:
            # Continue emergency return until back in safe zone
            if dist_to_center < self.safe_zone_radius:
                self.in_emergency_return = False
                self.get_logger().info("✓ Returned to safe zone - resuming RL control")
                return safe_cmd, f"SAFE (returned, dist: {min_dist:.2f}m)"
            else:
                # Keep returning
                safe_cmd.linear.x = float(dir_to_center[0] * self.boundary_return_speed)
                safe_cmd.linear.y = float(dir_to_center[1] * self.boundary_return_speed)
                safe_cmd.linear.z = float(dir_to_center[2] * 0.1)
                safe_cmd.angular.z = 0.0
                return safe_cmd, f"RETURNING (dist to center: {dist_to_center:.2f}m)"
        
        elif status == 'WARNING':
            # Reduce RL command toward boundary
            self.get_logger().warn(
                f"⚡ WARNING ZONE! Boundary dist: {min_dist:.2f}m - Moderating RL command",
                throttle_duration_sec=1.0
            )
            
            # Scale down velocity components that move toward boundary
            x, y, z = self.drone_pos
            
            # Check each axis
            if x < (self.boundary_x_min + self.warning_distance) and safe_cmd.linear.x < 0:
                scale = max(0.3, (x - self.boundary_x_min) / self.warning_distance)
                safe_cmd.linear.x *= scale
            elif x > (self.boundary_x_max - self.warning_distance) and safe_cmd.linear.x > 0:
                scale = max(0.3, (self.boundary_x_max - x) / self.warning_distance)
                safe_cmd.linear.x *= scale
            
            if y < (self.boundary_y_min + self.warning_distance) and safe_cmd.linear.y < 0:
                scale = max(0.3, (y - self.boundary_y_min) / self.warning_distance)
                safe_cmd.linear.y *= scale
            elif y > (self.boundary_y_max - self.warning_distance) and safe_cmd.linear.y > 0:
                scale = max(0.3, (self.boundary_y_max - y) / self.warning_distance)
                safe_cmd.linear.y *= scale
            
            return safe_cmd, f"WARNING (dist: {min_dist:.2f}m, moderated)"
        
        else:  # SAFE
            # RL has full control
            return safe_cmd, f"SAFE (dist: {min_dist:.2f}m, RL active)"

    # ═══════════════════════════════════════════════════════════════════
    # STANDARD CALLBACKS
    # ═══════════════════════════════════════════════════════════════════
    
    def state_cb(self, msg):
        self.current_state = msg
        self.offboard_enabled = (msg.mode == "GUIDED")
        self.armed = msg.armed

    def drone_pose_cb(self, msg):
        self.drone_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.drone_orient = [msg.pose.orientation.x, msg.pose.orientation.y, 
                            msg.pose.orientation.z, msg.pose.orientation.w]
        self.guidance.update_agent_position(self.drone_pos[:2])

    def drone_vel_cb(self, msg):
        if hasattr(msg, 'twist'):
            v = msg.twist.linear
        else:
            v = msg.linear
        self.drone_vel = np.array([v.x, v.y, v.z])

    def target_cb(self, msg):
        pos = [msg.pose.position.x, msg.pose.position.y]
        self.guidance.update_target_pose(position=pos)
        current_time_sec = self.get_clock().now().nanoseconds / 1e9
        self.guidance.guidance_filter_loop(t=current_time_sec)
        self.raw_target_pos = np.array(pos)
        
        if not hasattr(self, 'last_raw_target'):
            self.last_raw_target = np.array(pos)
            self.last_raw_time = current_time_sec
            self.raw_target_vel = np.array([0.0, 0.0])
        else:
            dt = current_time_sec - self.last_raw_time
            if dt > 0.01:
                self.raw_target_vel = (self.raw_target_pos - self.last_raw_target) / dt
                self.last_raw_target = self.raw_target_pos.copy()
                self.last_raw_time = current_time_sec
        
        self.last_target_time = current_time_sec

    # ═══════════════════════════════════════════════════════════════════
    # OFFBOARD INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    def initialize_offboard(self):
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
                self.get_logger().info("✓ OFFBOARD + ARMED. Starting RL control with boundary safety...")
                self.init_timer.cancel()
                self.control_timer = self.create_timer(0.05, self.control_loop)

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

    # ═══════════════════════════════════════════════════════════════════
    # MAIN CONTROL LOOP
    # ═══════════════════════════════════════════════════════════════════
    
    def control_loop(self):
        """Main RL control loop with integrated boundary safety"""
        
        # Initialize command
        cmd = Twist()
        
        # Safety check: mode/arm status
        if not self.offboard_enabled or not self.armed:
            self.get_logger().error("Lost OFFBOARD/disarmed!", throttle_duration_sec=5.0)
            self.vel_pub.publish(Twist())
            return

        # Altitude control (always active)
        if self.drone_pos is not None:
            err_z = self.target_height - self.drone_pos[2]
            cmd.linear.z = np.clip(err_z * 1.0, -0.5, 0.5)
        else:
            cmd.linear.z = 0.2
            self.vel_pub.publish(cmd)
            return

        # Check prerequisites
        if self.drone_orient is None:
            self.vel_pub.publish(cmd)
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_target_time is None or (current_time - self.last_target_time) > 1.0:
            self.get_logger().warn("Target stale, hovering", throttle_duration_sec=2.0)
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            self.vel_pub.publish(cmd)
            return

        # Get target states
        if hasattr(self, 'raw_target_pos') and self.raw_target_pos is not None:
            tracked_states = [np.concatenate([self.raw_target_pos, self.raw_target_vel])]
        else:
            self.vel_pub.publish(cmd)
            return
        
        # Build agent state
        yaw = self.euler_from_quaternion(self.drone_orient)[2]
        agent_state_vector = np.array([
            self.drone_pos[0], self.drone_pos[1], yaw,
            self.drone_vel[0], self.drone_vel[1], 0.0
        ])

        # ═══════════════════════════════════════════════════════════
        # STEP 1: Get RL policy action
        # ═══════════════════════════════════════════════════════════
        try:
            action, lstm_states = self.guidance.get_ppo_action_all_tracks(
                agent_state=agent_state_vector,
                tracked_states=tracked_states,
                max_targets=10,
                lstm_states=self.lstm_states,
                episode_start=self.episode_start
            )

            self.lstm_states = lstm_states
            self.episode_start = np.array([False])

            if action is not None:
                vx = np.clip(action[0], -self.max_vel, self.max_vel)
                vy = np.clip(action[1], -self.max_vel, self.max_vel)
                omega = np.clip(action[2], -1.0, 1.0)
                cmd.linear.x = float(vx)
                cmd.linear.y = float(vy)
                cmd.angular.z = float(omega)
            else:
                cmd.linear.x = 0.0
                cmd.linear.y = 0.0
                
        except Exception as e:
            self.get_logger().error(f"RL error: {e}", throttle_duration_sec=5.0)
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0

        # ═══════════════════════════════════════════════════════════
        # STEP 2: APPLY BOUNDARY SAFETY (TOP PRIORITY!)
        # ═══════════════════════════════════════════════════════════
        safe_cmd, safety_status = self.apply_boundary_safety(cmd)
        
        # Log
        if not hasattr(self, '_last_log') or (current_time - self._last_log) > 1.0:
            self._last_log = current_time
            self.get_logger().info(
                f"RL→[{safe_cmd.linear.x:.2f}, {safe_cmd.linear.y:.2f}] | {safety_status}"
            )

        # ═══════════════════════════════════════════════════════════
        # STEP 3: PUBLISH (always!)
        # ═══════════════════════════════════════════════════════════
        self.vel_pub.publish(safe_cmd)

    @staticmethod
    def euler_from_quaternion(q):
        x, y, z, w = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        return roll_x, pitch_y, yaw_z


def main(args=None):
    rclpy.init(args=args)
    node = RLOffboardTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
