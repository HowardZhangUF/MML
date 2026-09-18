#!/usr/bin/env python3
"""
Simulated UAV for RViz Testing - DUAL MODE
Mode 1: RL policy waypoints (original)
Mode 2: Direct Howard following at same frequency (for comparison)
"""
import sys
sys.path.append('/home/basestation/mml_python_sim')
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
import numpy as np

from mml_guidance.guidance import Guidance
from mml_guidance.ParticleFilter import ParticleFilter


class SimulatedUAVTracker(Node):
    """
    Virtual UAV with dual control modes:
    1. RL Policy: Waypoints from reinforcement learning
    2. Predefined: Waypoints from Howard's position at same frequency
    """
    def __init__(self):
        super().__init__('simulated_uav_tracker')

        # ═══════════════════════════════════════════════════════════
        # ★ NEW: MODE SELECTION PARAMETER
        # ═══════════════════════════════════════════════════════════
        self.declare_parameter('control_mode', 'rl_policy')  # or 'predefined'
        self.declare_parameter('waypoint_update_rate', 1.0)  # Hz for predefined mode

        # Existing parameters (unchanged)
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_topic', '/howard/enu/pose')
        self.declare_parameter('drone_height', 1.0)
        self.declare_parameter('num_particles', 200)
        self.declare_parameter('prediction_method', 'Velocity')
        self.declare_parameter('max_vel', 0.5)
        
        # Position control parameters
        self.declare_parameter('integration_steps', 5)
        self.declare_parameter('integration_dt', 0.05)
        self.declare_parameter('waypoint_threshold', 0.1)
        
        # Simulation parameters
        self.declare_parameter('sim_dt', 0.05)  # Simulation time step
        self.declare_parameter('sim_velocity', 0.3)  # UAV movement speed
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        
        # Boundary parameters
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        self.declare_parameter('boundary_z_min', 0.8)
        self.declare_parameter('boundary_z_max', 2.0)
        self.declare_parameter('safe_zone_radius', 0.5)
        self.declare_parameter('warning_distance', 0.3)
        self.declare_parameter('emergency_distance', 0.15)

        # ═══════════════════════════════════════════════════════════
        # ★ GET MODE PARAMETER
        # ═══════════════════════════════════════════════════════════
        control_mode_str = self.get_parameter('control_mode').value
        self.control_mode = control_mode_str.lower()  # 'rl_policy' or 'predefined'
        self.waypoint_update_rate = self.get_parameter('waypoint_update_rate').value
        
        # Get existing parameters
        model_path = self.get_parameter('model_path').value
        target_topic = self.get_parameter('target_topic').value
        self.target_height = self.get_parameter('drone_height').value
        num_particles = self.get_parameter('num_particles').value
        prediction_method = self.get_parameter('prediction_method').value
        self.max_vel = self.get_parameter('max_vel').value
        
        self.integration_steps = self.get_parameter('integration_steps').value
        self.integration_dt = self.get_parameter('integration_dt').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        
        self.sim_dt = self.get_parameter('sim_dt').value
        self.sim_velocity = self.get_parameter('sim_velocity').value
        
        # Boundaries
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        self.boundary_z_min = self.get_parameter('boundary_z_min').value
        self.boundary_z_max = self.get_parameter('boundary_z_max').value
        self.safe_zone_radius = self.get_parameter('safe_zone_radius').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.emergency_distance = self.get_parameter('emergency_distance').value
        
        # Safe center
        self.safe_center_x = (self.boundary_x_min + self.boundary_x_max) / 2.0
        self.safe_center_y = (self.boundary_y_min + self.boundary_y_max) / 2.0
        self.safe_center_z = (self.boundary_z_min + self.boundary_z_max) / 2.0
        
        # Simulated UAV state
        self.sim_pos = np.array([
            self.get_parameter('initial_x').value,
            self.get_parameter('initial_y').value,
            self.target_height
        ])
        self.sim_vel = np.array([0.0, 0.0, 0.0])
        self.sim_yaw = 0.0
        
        # Waypoint tracking
        self.current_waypoint = None
        self.velocity_buffer = []  # For RL mode only
        self.step_counter = 0       # For RL mode only
        self.waypoint_history = []
        self.trajectory_points = []
        
        # ★ NEW: Predefined mode timing
        self.last_waypoint_time = 0.0  # Track when last waypoint was generated
        
        # Boundary state
        self.in_emergency_return = False
        self.boundary_violations = 0
        
        # Target tracking
        self.raw_target_pos = None
        self.raw_target_vel = np.array([0.0, 0.0])
        self.last_target_time = None

        # ═══════════════════════════════════════════════════════════
        # ★ LOGGING - Show current mode
        # ═══════════════════════════════════════════════════════════
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("SIMULATED UAV TRACKER - DUAL MODE")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🎮 Control Mode: {self.control_mode}")
        
        if self.control_mode == 'predefined':
            self.get_logger().info(f"📍 Predefined Mode: Follow Howard's position directly")
            self.get_logger().info(f"   Waypoint Update Rate: {self.waypoint_update_rate:.1f} Hz")
            self.get_logger().info(f"   (Matches RL frequency for fair comparison)")
        else:
            self.get_logger().info(f"🤖 RL Policy Mode: Neural network generates waypoints")
            self.get_logger().info(f"   Model: {model_path}")
            self.get_logger().info(f"   Integration: {self.integration_steps} steps × {self.integration_dt}s")
        
        self.get_logger().info(f"🎯 Target: {target_topic}")
        self.get_logger().info(f"📍 Initial: [{self.sim_pos[0]:.2f}, {self.sim_pos[1]:.2f}]")
        self.get_logger().info(f"⚡ Velocity: {self.sim_velocity} m/s")
        self.get_logger().info(f"📦 Boundaries: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] "
                              f"Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}]")
        self.get_logger().info("=" * 60)

        # ═══════════════════════════════════════════════════════════
        # ★ CONDITIONAL GUIDANCE INITIALIZATION
        # ═══════════════════════════════════════════════════════════
        
        if self.control_mode == 'rl_policy':
            # Initialize RL system
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
            
            self.get_logger().info("✓ RL guidance system initialized")
        else:
            # Predefined mode - no RL needed
            self.guidance = None
            self.pf = None
            self.lstm_states = None
            self.episode_start = None
            
            self.get_logger().info("✓ Predefined mode - no RL system loaded (faster startup)")

        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_target = self.create_subscription(
            PoseStamped, target_topic, self.target_cb, qos_profile)
        
        # ★ NEW: Mode switching service (optional)
        self.mode_sub = self.create_subscription(
            String, '/sim_uav/set_mode', self.mode_switch_cb, 10)

        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, '/sim_uav/markers', 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/sim_uav/pose', 10)
        # ★ NEW: Publish current mode
        self.mode_pub = self.create_publisher(
            String, '/sim_uav/current_mode', 10)

        # Timers
        self.control_timer = self.create_timer(self.sim_dt, self.control_loop)
        self.viz_timer = self.create_timer(0.1, self.publish_visualization)
        self.mode_pub_timer = self.create_timer(1.0, self.publish_mode)
        
        self.get_logger().info("✓ Simulated UAV tracker ready - waiting for target...")

    # ═══════════════════════════════════════════════════════════════════
    # ★ NEW: MODE SWITCHING CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def mode_switch_cb(self, msg):
        """Runtime mode switching (optional feature)"""
        new_mode = msg.data.lower()
        if new_mode in ['rl_policy', 'predefined'] and new_mode != self.control_mode:
            self.control_mode = new_mode
            
            # Reset state
            self.current_waypoint = None
            self.velocity_buffer = []
            self.step_counter = 0
            self.last_waypoint_time = 0.0
            
            if new_mode == 'rl_policy' and self.guidance:
                self.lstm_states = None
                self.episode_start = np.array([True])
            
            self.get_logger().info(f"🔄 Switched to mode: {new_mode}")
        else:
            self.get_logger().error(f"Invalid mode: {msg.data}")

    def publish_mode(self):
        """Publish current control mode"""
        msg = String()
        msg.data = self.control_mode
        self.mode_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════════
    # CALLBACKS (mostly unchanged)
    # ═══════════════════════════════════════════════════════════════════

    def target_cb(self, msg):
        """Update target position and velocity"""
        pos = [msg.pose.position.x, msg.pose.position.y]
        
        # ★ CHANGE: Only update guidance if RL mode
        if self.control_mode == 'rl_policy' and self.guidance:
            self.guidance.update_target_pose(position=pos)
            current_time_sec = self.get_clock().now().nanoseconds / 1e9
            self.guidance.guidance_filter_loop(t=current_time_sec)
        
        self.raw_target_pos = np.array(pos)
        
        # Estimate velocity
        if not hasattr(self, 'last_raw_target'):
            self.last_raw_target = np.array(pos)
            self.last_raw_time = self.get_clock().now().nanoseconds / 1e9
            self.raw_target_vel = np.array([0.0, 0.0])
        else:
            current_time_sec = self.get_clock().now().nanoseconds / 1e9
            dt = current_time_sec - self.last_raw_time
            if dt > 0.01:
                self.raw_target_vel = (self.raw_target_pos - self.last_raw_target) / dt
                self.last_raw_target = self.raw_target_pos.copy()
                self.last_raw_time = current_time_sec
        
        self.last_target_time = self.get_clock().now().nanoseconds / 1e9

    # ═══════════════════════════════════════════════════════════════════
    # BOUNDARY SAFETY (unchanged)
    # ═══════════════════════════════════════════════════════════════════

    def calculate_boundary_status(self, pos):
        """Calculate boundary safety status"""
        if pos is None:
            return 'UNKNOWN', np.array([0.0, 0.0, 0.0]), 0.0, 0.0
        
        x, y, z = pos
        
        dist_x_min = x - self.boundary_x_min
        dist_x_max = self.boundary_x_max - x
        dist_y_min = y - self.boundary_y_min
        dist_y_max = self.boundary_y_max - y
        dist_z_min = z - self.boundary_z_min
        dist_z_max = self.boundary_z_max - z
        
        min_dist = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)
        
        if min_dist < 0:
            status = 'VIOLATION'
        elif min_dist < self.emergency_distance:
            status = 'EMERGENCY'
        elif min_dist < self.warning_distance:
            status = 'WARNING'
        else:
            status = 'SAFE'
        
        direction = np.array([
            self.safe_center_x - x,
            self.safe_center_y - y,
            self.safe_center_z - z
        ])
        
        dist_to_center = np.linalg.norm(direction[:2])
        
        if dist_to_center > 0.01:
            direction_norm = direction / np.linalg.norm(direction)
        else:
            direction_norm = np.array([0.0, 0.0, 0.0])
        
        return status, direction_norm, min_dist, dist_to_center

    def apply_boundary_safety(self, rl_cmd):
        """Apply boundary safety to RL velocity command"""
        status, dir_to_center, min_dist, dist_to_center = \
            self.calculate_boundary_status(self.sim_pos)
        
        safe_cmd = Twist()
        safe_cmd.linear.x = rl_cmd.linear.x
        safe_cmd.linear.y = rl_cmd.linear.y
        safe_cmd.angular.z = rl_cmd.angular.z
        
        if status == 'VIOLATION' or status == 'EMERGENCY':
            if not self.in_emergency_return:
                self.boundary_violations += 1
                self.in_emergency_return = True
                self.current_waypoint = None
                self.velocity_buffer = []
                self.step_counter = 0
                self.get_logger().error(
                    f"🚨 BOUNDARY {status}! Returning to center"
                )
            
            return_speed = 0.4 if status == 'VIOLATION' else 0.3
            safe_cmd.linear.x = float(dir_to_center[0] * return_speed)
            safe_cmd.linear.y = float(dir_to_center[1] * return_speed)
            
            return safe_cmd, f"EMERGENCY_RETURN"
        
        elif self.in_emergency_return:
            if dist_to_center < self.safe_zone_radius:
                self.in_emergency_return = False
                self.get_logger().info("✓ Returned to safe zone")
            else:
                safe_cmd.linear.x = float(dir_to_center[0] * 0.3)
                safe_cmd.linear.y = float(dir_to_center[1] * 0.3)
                return safe_cmd, f"RETURNING"
        
        elif status == 'WARNING':
            x, y, z = self.sim_pos
            
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
            
            return safe_cmd, f"WARNING"
        
        return safe_cmd, f"SAFE"

    def clip_waypoint_to_boundaries(self, waypoint):
        """Clip waypoint to safe boundaries"""
        if waypoint is None:
            return None
        
        clipped = waypoint.copy()
        clipped[0] = np.clip(clipped[0],
                            self.boundary_x_min + self.warning_distance,
                            self.boundary_x_max - self.warning_distance)
        clipped[1] = np.clip(clipped[1],
                            self.boundary_y_min + self.warning_distance,
                            self.boundary_y_max - self.warning_distance)
        return clipped

    # ═══════════════════════════════════════════════════════════════════
    # VELOCITY INTEGRATION (for RL mode only)
    # ═══════════════════════════════════════════════════════════════════

    def integrate_velocity_to_waypoint(self, velocity_cmd):
        """Integrate velocity over time steps to generate waypoint"""
        self.velocity_buffer.append([velocity_cmd.linear.x, velocity_cmd.linear.y])
        self.step_counter += 1
        
        if self.step_counter >= self.integration_steps:
            integrated_displacement = np.array([0.0, 0.0])
            for vel in self.velocity_buffer:
                integrated_displacement += np.array(vel) * self.integration_dt
            
            waypoint = self.sim_pos[:2] + integrated_displacement
            
            self.velocity_buffer = []
            self.step_counter = 0
            
            return waypoint
        
        return None

    # ═══════════════════════════════════════════════════════════════════
    # SIMULATION DYNAMICS (unchanged)
    # ═══════════════════════════════════════════════════════════════════

    def simulate_movement(self):
        """Simulate UAV movement toward current waypoint at CONSTANT VELOCITY"""
        if self.current_waypoint is None:
            self.sim_vel[:2] = [0.0, 0.0]
            return
        
        current_pos_2d = self.sim_pos[:2]
        direction = self.current_waypoint - current_pos_2d
        distance = np.linalg.norm(direction)
        
        if distance < 0.05:
            self.sim_pos[:2] = self.current_waypoint
            self.sim_vel[:2] = [0.0, 0.0]
            if not hasattr(self, '_waypoint_reached') or not self._waypoint_reached:
                self._waypoint_reached = True
                self.get_logger().info(
                    f"✓ Holding at waypoint [{self.current_waypoint[0]:.2f}, {self.current_waypoint[1]:.2f}]"
                )
            return
        
        self._waypoint_reached = False
        direction_norm = direction / distance
        step_size = min(self.sim_velocity * self.sim_dt, distance)
        
        self.sim_pos[:2] += direction_norm * step_size
        self.sim_vel[:2] = direction_norm * self.sim_velocity
        
        if len(self.trajectory_points) == 0 or \
           np.linalg.norm(self.sim_pos - self.trajectory_points[-1]) > 0.05:
            self.trajectory_points.append(self.sim_pos.copy())
            if len(self.trajectory_points) > 1000:
                self.trajectory_points.pop(0)

    # ═══════════════════════════════════════════════════════════════════
    # ★ MAIN CONTROL LOOP - MODE BRANCHING HAPPENS HERE
    # ═══════════════════════════════════════════════════════════════════

    def control_loop(self):
        """
        Main control loop - runs at sim_dt rate (20 Hz)
        Branches based on control_mode
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check if target is available
        if self.last_target_time is None or \
           (current_time - self.last_target_time) > 1.0:
            return
        
        if self.raw_target_pos is None:
            return
        
        # ═════════════════════════════════════════════════════════
        # ★ MODE BRANCHING - THIS IS THE KEY CHANGE
        # ═════════════════════════════════════════════════════════
        
        new_waypoint = None
        
        if self.control_mode == 'rl_policy':
            # ──────────────────────────────────────────────────────
            # MODE 1: RL Policy (ORIGINAL)
            # ──────────────────────────────────────────────────────
            new_waypoint = self.generate_rl_waypoint()
        
        elif self.control_mode == 'predefined':
            # ──────────────────────────────────────────────────────
            # MODE 2: Predefined (Direct Howard Following)
            # ──────────────────────────────────────────────────────
            new_waypoint = self.generate_predefined_waypoint(current_time)
        
        # ═════════════════════════════════════════════════════════
        # COMMON: Both modes use same waypoint tracking
        # ═════════════════════════════════════════════════════════
        
        if new_waypoint is not None:
            self.current_waypoint = new_waypoint
            
            # Store waypoint for visualization
            self.waypoint_history.append({
                'pos': new_waypoint.copy(),
                'time': current_time
            })
            if len(self.waypoint_history) > 50:
                self.waypoint_history.pop(0)
        
        # Simulate movement toward current waypoint (same for both modes)
        self.simulate_movement()
        
        # Publish pose
        self.publish_pose()

    # ═══════════════════════════════════════════════════════════════════
    # ★ NEW: PREDEFINED MODE WAYPOINT GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def generate_predefined_waypoint(self, current_time):
        """
        Generate waypoint from Howard's position at same frequency as RL
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            np.array([x, y]) or None if not time yet
        """
        # ──────────────────────────────────────────────────────
        # Rate limiting: Only generate at waypoint_update_rate
        # ──────────────────────────────────────────────────────
        dt_since_last = current_time - self.last_waypoint_time
        
        if dt_since_last < (1.0 / self.waypoint_update_rate):
            return None  # Not time yet, skip this cycle
        
        # Time to generate new waypoint
        self.last_waypoint_time = current_time
        
        # ──────────────────────────────────────────────────────
        # Waypoint = Howard's current position (simple!)
        # ──────────────────────────────────────────────────────
        waypoint = self.raw_target_pos.copy()
        
        # ──────────────────────────────────────────────────────
        # Safety: Clip to boundaries
        # ──────────────────────────────────────────────────────
        waypoint = self.clip_waypoint_to_boundaries(waypoint)
        
        self.get_logger().info(
            f"📍 Predefined waypoint: [{waypoint[0]:.2f}, {waypoint[1]:.2f}] "
            f"(Howard's position)"
        )
        
        return waypoint

    # ═══════════════════════════════════════════════════════════════════
    # RL POLICY WAYPOINT GENERATION (unchanged)
    # ═══════════════════════════════════════════════════════════════════

    def generate_rl_waypoint(self):
        """Generate waypoint using RL policy (original method)"""
        if self.guidance is None:
            return None
        
        # Update guidance with current position
        self.guidance.update_agent_position(self.sim_pos[:2])
        
        # Build agent state
        agent_state = np.array([
            self.sim_pos[0], self.sim_pos[1], self.sim_yaw,
            self.sim_vel[0], self.sim_vel[1], 0.0
        ])
        
        # Get tracked states
        tracked_states = [np.concatenate([self.raw_target_pos, self.raw_target_vel])]
        
        # Get RL action
        rl_velocity_cmd = Twist()
        try:
            action, lstm_states = self.guidance.get_ppo_action_all_tracks(
                agent_state=agent_state,
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
                rl_velocity_cmd.linear.x = float(vx)
                rl_velocity_cmd.linear.y = float(vy)
                rl_velocity_cmd.angular.z = float(omega)
        except Exception as e:
            self.get_logger().error(f"RL error: {e}")
            return None
        
        # Apply boundary safety
        safe_velocity_cmd, safety_status = self.apply_boundary_safety(rl_velocity_cmd)
        
        # Integrate velocity to waypoint
        new_waypoint = self.integrate_velocity_to_waypoint(safe_velocity_cmd)
        
        if new_waypoint is not None:
            new_waypoint = self.clip_waypoint_to_boundaries(new_waypoint)
            self.get_logger().info(
                f"🤖 RL waypoint: [{new_waypoint[0]:.2f}, {new_waypoint[1]:.2f}] | {safety_status}"
            )
            return new_waypoint
        
        return None

    def publish_pose(self):
        """Publish simulated UAV pose"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(self.sim_pos[0])
        msg.pose.position.y = float(self.sim_pos[1])
        msg.pose.position.z = float(self.sim_pos[2])
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZATION (unchanged - works for both modes)
    # ═══════════════════════════════════════════════════════════════════

    def publish_visualization(self):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # 1. Simulated UAV position (large sphere)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sim_uav"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.sim_pos[0])
        marker.pose.position.y = float(self.sim_pos[1])
        marker.pose.position.z = float(self.sim_pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        
        # ★ Color based on mode
        status, _, min_dist, _ = self.calculate_boundary_status(self.sim_pos)
        if status == 'VIOLATION':
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        elif status == 'EMERGENCY':
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        elif status == 'WARNING':
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        else:
            # Purple for RL, Blue for predefined
            if self.control_mode == 'rl_policy':
                marker.color.r, marker.color.g, marker.color.b = 0.6, 0.0, 1.0  # Purple
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.5, 1.0  # Blue
        marker.color.a = 1.0
        
        marker_array.markers.append(marker)
        marker_id += 1
        
        # 2. Simulated UAV trajectory (line)
        if len(self.trajectory_points) > 1:
            traj_marker = Marker()
            traj_marker.header.frame_id = "map"
            traj_marker.header.stamp = self.get_clock().now().to_msg()
            traj_marker.ns = "sim_trajectory"
            traj_marker.id = marker_id
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD
            traj_marker.scale.x = 0.04
            
            # Purple for RL, Blue for predefined
            if self.control_mode == 'rl_policy':
                traj_marker.color.r, traj_marker.color.g, traj_marker.color.b = 0.6, 0.0, 1.0
            else:
                traj_marker.color.r, traj_marker.color.g, traj_marker.color.b = 0.0, 0.5, 1.0
            traj_marker.color.a = 0.8
            
            from geometry_msgs.msg import Point
            for point in self.trajectory_points:
                p = Point()
                p.x, p.y, p.z = point
                traj_marker.points.append(p)
            
            marker_array.markers.append(traj_marker)
            marker_id += 1
        
        # 3. Current waypoint (large green sphere)
        if self.current_waypoint is not None:
            wp_marker = Marker()
            wp_marker.header.frame_id = "map"
            wp_marker.header.stamp = self.get_clock().now().to_msg()
            wp_marker.ns = "current_waypoint"
            wp_marker.id = marker_id
            wp_marker.type = Marker.SPHERE
            wp_marker.action = Marker.ADD
            wp_marker.pose.position.x = float(self.current_waypoint[0])
            wp_marker.pose.position.y = float(self.current_waypoint[1])
            wp_marker.pose.position.z = float(self.target_height)
            wp_marker.pose.orientation.w = 1.0
            wp_marker.scale.x = 0.2
            wp_marker.scale.y = 0.2
            wp_marker.scale.z = 0.2
            wp_marker.color.r = 0.0
            wp_marker.color.g = 1.0
            wp_marker.color.b = 0.0
            wp_marker.color.a = 1.0
            
            marker_array.markers.append(wp_marker)
            marker_id += 1
        
        # 4. Waypoint history (small spheres)
        for i, wp_data in enumerate(self.waypoint_history):
            wp = wp_data['pos']
            hist_marker = Marker()
            hist_marker.header.frame_id = "map"
            hist_marker.header.stamp = self.get_clock().now().to_msg()
            hist_marker.ns = "waypoint_history"
            hist_marker.id = marker_id
            hist_marker.type = Marker.SPHERE
            hist_marker.action = Marker.ADD
            hist_marker.pose.position.x = float(wp[0])
            hist_marker.pose.position.y = float(wp[1])
            hist_marker.pose.position.z = float(self.target_height)
            hist_marker.pose.orientation.w = 1.0
            hist_marker.scale.x = 0.08
            hist_marker.scale.y = 0.08
            hist_marker.scale.z = 0.08
            
            # Fade older waypoints
            alpha = 0.3 + 0.7 * (i / max(len(self.waypoint_history), 1))
            hist_marker.color.r = 0.0
            hist_marker.color.g = 1.0
            hist_marker.color.b = 1.0
            hist_marker.color.a = alpha
            
            marker_array.markers.append(hist_marker)
            marker_id += 1
        
        # 5. Velocity arrow (if moving)
        if np.linalg.norm(self.sim_vel[:2]) > 0.05:
            vel_marker = Marker()
            vel_marker.header.frame_id = "map"
            vel_marker.header.stamp = self.get_clock().now().to_msg()
            vel_marker.ns = "sim_velocity"
            vel_marker.id = marker_id
            vel_marker.type = Marker.ARROW
            vel_marker.action = Marker.ADD
            
            from geometry_msgs.msg import Point
            start = Point()
            start.x, start.y, start.z = self.sim_pos
            
            end = Point()
            end.x = self.sim_pos[0] + self.sim_vel[0] * 2.0
            end.y = self.sim_pos[1] + self.sim_vel[1] * 2.0
            end.z = self.sim_pos[2]
            
            vel_marker.points.append(start)
            vel_marker.points.append(end)
            
            vel_marker.scale.x = 0.05
            vel_marker.scale.y = 0.1
            vel_marker.scale.z = 0.15
            
            vel_marker.color.r = 0.0
            vel_marker.color.g = 1.0
            vel_marker.color.b = 1.0
            vel_marker.color.a = 1.0
            
            marker_array.markers.append(vel_marker)
            marker_id += 1
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedUAVTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()