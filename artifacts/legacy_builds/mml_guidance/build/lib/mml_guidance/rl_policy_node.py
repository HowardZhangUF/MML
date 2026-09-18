#!/usr/bin/env python3
"""
RL Policy Node (NODE 1)
=======================
Pure RL inference node that:
  - Reads UAV and target poses
  - Builds observations for RL model
  - Inferences RecurrentPPO policy
  - MODE 1 (velocity): Publishes velocity commands directly
  - MODE 2 (position): Integrates velocity to position waypoints
  - Does NOT communicate with MAVROS directly
  
Published Topics:
  - /rl_policy/command (RLCommand): Commands for drone_commander
  - /rl_policy/markers (MarkerArray): Visualization
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    PPO_AVAILABLE = True
except ImportError as e:
    PPO_AVAILABLE = False
    print(f"WARNING: stable-baselines3 or sb3-contrib import failed: {e}")
    print("Run: pip install stable-baselines3 sb3-contrib")
except Exception as e:
    PPO_AVAILABLE = False
    print(f"ERROR importing RL libraries: {type(e).__name__}: {e}")

# Import custom message (will be generated after colcon build)
try:
    from mml_guidance.msg import RLCommand
except ImportError:
    print("WARNING: RLCommand message not built yet. Run: colcon build --packages-select mml_guidance")
    RLCommand = None


class RLPolicyNode(Node):
    """RL Policy Inference Node with dual control modes"""
    
    def __init__(self):
        super().__init__('rl_policy_node')
        
        # ═══════════════════════════════════════════════════════════════
        # DECLARE PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        self.declare_parameter('control_mode', 'velocity')  # 'velocity' or 'position'
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('model_path', '')
        self.declare_parameter('fixed_altitude', 1.4)  # meters
        self.declare_parameter('max_tracked_targets', 10)
        
        # Integration parameters (for position mode)
        self.declare_parameter('max_position_jump', 0.5)  # meters per cycle
        self.declare_parameter('boundary_x_min', -2.0)
        self.declare_parameter('boundary_x_max', 2.0)
        self.declare_parameter('boundary_y_min', -1.5)
        self.declare_parameter('boundary_y_max', 1.5)
        
        # Get parameters
        self.control_mode = self.get_parameter('control_mode').value
        self.control_rate = self.get_parameter('control_rate').value
        self.model_path = self.get_parameter('model_path').value
        self.fixed_altitude = self.get_parameter('fixed_altitude').value
        self.max_tracked_targets = self.get_parameter('max_tracked_targets').value
        self.max_position_jump = self.get_parameter('max_position_jump').value
        
        # Boundaries
        self.boundary_x_min = self.get_parameter('boundary_x_min').value
        self.boundary_x_max = self.get_parameter('boundary_x_max').value
        self.boundary_y_min = self.get_parameter('boundary_y_min').value
        self.boundary_y_max = self.get_parameter('boundary_y_max').value
        
        self.get_logger().info("="*80)
        self.get_logger().info("🎮 RL POLICY NODE INITIALIZED")
        self.get_logger().info("="*80)
        self.get_logger().info(f"Control Mode: {self.control_mode.upper()}")
        self.get_logger().info(f"Control Rate: {self.control_rate} Hz")
        self.get_logger().info(f"Model Path: {self.model_path}")
        self.get_logger().info("="*80)
        
        # ═══════════════════════════════════════════════════════════════
        # STATE VARIABLES
        # ═══════════════════════════════════════════════════════════════
        self.uav_pos = None
        self.uav_vel = None
        self.prev_uav_pos = None
        self.prev_uav_time = None
        
        self.target_pos = None
        self.target_vel = None
        self.prev_target_pos = None
        self.prev_target_time = None
        
        self.uav_received = False
        self.target_received = False
        
        # RL model state
        self.model = None
        self.lstm_states = None
        self.episode_start = True
        
        # Position integration state (MODE 2)
        self.integrated_position = None
        self.last_integration_time = None
        self.integration_initialized = False
        
        # ═══════════════════════════════════════════════════════════════
        # LOAD RL MODEL
        # ═══════════════════════════════════════════════════════════════
        self.load_rl_model()
        
        # ═══════════════════════════════════════════════════════════════
        # PUBLISHERS
        # ═══════════════════════════════════════════════════════════════
        if RLCommand is not None:
            self.cmd_publisher = self.create_publisher(
                RLCommand, '/rl_policy/command', 10)
        else:
            self.get_logger().error("RLCommand message not available. Cannot create publisher.")
            self.cmd_publisher = None
        
        # Visualization publisher (optional - may fail in some ROS environments)
        try:
            self.marker_publisher = self.create_publisher(
                MarkerArray, '/rl_policy/markers', 10)
        except Exception as e:
            self.get_logger().warn(f"Could not create visualization publisher: {e}")
            self.get_logger().warn("Continuing without visualization markers")
            self.marker_publisher = None
        
        # ═══════════════════════════════════════════════════════════════
        # SUBSCRIBERS
        # ═══════════════════════════════════════════════════════════════
        self.uav_sub = self.create_subscription(
            PoseStamped, '/pop/enu/pose', self.uav_pose_callback, 10)
        
        self.target_sub = self.create_subscription(
            PoseStamped, '/howard/enu/pose', self.target_pose_callback, 10)
        
        # ═══════════════════════════════════════════════════════════════
        # CONTROL TIMER
        # ═══════════════════════════════════════════════════════════════
        timer_period = 1.0 / self.control_rate  # seconds
        self.control_timer = self.create_timer(timer_period, self.control_loop_callback)
        
        self.get_logger().info(f"✅ RL Policy Node ready! Control loop @ {self.control_rate} Hz")
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═══════════════════════════════════════════════════════════════════
    
    def load_rl_model(self):
        """Load the trained RecurrentPPO model"""
        if not self.model_path or not os.path.exists(self.model_path):
            self.get_logger().error(f"❌ Model not found: {self.model_path}")
            self.get_logger().error("Please provide valid model_path parameter")
            return
        
        try:
            self.get_logger().info(f"Loading RL model from: {self.model_path}")
            
            if not PPO_AVAILABLE:
                self.get_logger().error("❌ RL libraries not installed!")
                return
            
            # Add workspace root to path for custom modules (attention_policy, etc.)
            # The model file is in models_recurrent/ which is at workspace root
            workspace_root = str(Path(self.model_path).parent.parent)
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
                self.get_logger().info(f"Added to Python path: {workspace_root}")
            
            self.model = RecurrentPPO.load(self.model_path)
            self.get_logger().info("✅ RL model loaded successfully!")
            
            # Initialize LSTM states
            self.lstm_states = None
            self.episode_start = True
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")
            self.model = None
    
    # ═══════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════
    
    def uav_pose_callback(self, msg):
        """Callback for UAV pose"""
        current_time = self.get_clock().now()
        
        # Extract position
        self.uav_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Compute velocity (numerical derivative)
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
        self.uav_received = True
    
    def target_pose_callback(self, msg):
        """Callback for target (Howard) pose"""
        current_time = self.get_clock().now()
        
        # Extract position
        self.target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Compute velocity
        if self.prev_target_pos is not None and self.prev_target_time is not None:
            dt = (current_time - self.prev_target_time).nanoseconds / 1e9
            if dt > 0:
                self.target_vel = (self.target_pos - self.prev_target_pos) / dt
            else:
                self.target_vel = np.zeros(3)
        else:
            self.target_vel = np.zeros(3)
        
        self.prev_target_pos = self.target_pos.copy()
        self.prev_target_time = current_time
        self.target_received = True
    
    # ═══════════════════════════════════════════════════════════════════
    # CONTROL LOOP
    # ═══════════════════════════════════════════════════════════════════
    
    def control_loop_callback(self):
        """Main control loop - runs at control_rate Hz"""
        
        # Check if we have data
        if not self.uav_received or not self.target_received:
            self.get_logger().warn("⏳ Waiting for pose data...", throttle_duration_sec=2.0)
            return
        
        if self.model is None:
            self.get_logger().error("❌ RL model not loaded!", throttle_duration_sec=5.0)
            return
        
        if self.cmd_publisher is None:
            self.get_logger().error("❌ Command publisher not available!", throttle_duration_sec=5.0)
            return
        
        # ───────────────────────────────────────────────────────────────
        # 1. Build Observation
        # ───────────────────────────────────────────────────────────────
        obs = self.build_observation()
        
        # ───────────────────────────────────────────────────────────────
        # 2. RL Model Inference
        # ───────────────────────────────────────────────────────────────
        try:
            action, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=True
            )
            self.episode_start = False
            
            # Extract action components
            vx_rl = float(action[0])
            vy_rl = float(action[1])
            omega_rl = float(action[2]) if len(action) > 2 else 0.0
            
            # Clip to safety limits
            vx_rl = np.clip(vx_rl, -0.6, 0.6)
            vy_rl = np.clip(vy_rl, -0.6, 0.6)
            
        except Exception as e:
            self.get_logger().error(f"❌ RL inference failed: {e}")
            return
        
        # ───────────────────────────────────────────────────────────────
        # 3. Mode-Specific Processing
        # ───────────────────────────────────────────────────────────────
        current_time = self.get_clock().now()
        
        if self.control_mode == "position":
            # MODE 2: Integrate velocity to position
            target_x, target_y, dt_integration = self.integrate_to_position(
                vx_rl, vy_rl, current_time)
        else:
            # MODE 1: Direct velocity
            target_x = float('nan')
            target_y = float('nan')
            dt_integration = 0.0
        
        # ───────────────────────────────────────────────────────────────
        # 4. Compute Tracking Error
        # ───────────────────────────────────────────────────────────────
        rel_x = self.target_pos[0] - self.uav_pos[0]
        rel_y = self.target_pos[1] - self.uav_pos[1]
        tracking_error = np.sqrt(rel_x**2 + rel_y**2)
        
        # ───────────────────────────────────────────────────────────────
        # 5. Publish RLCommand
        # ───────────────────────────────────────────────────────────────
        cmd_msg = RLCommand()
        cmd_msg.header.stamp = current_time.to_msg()
        cmd_msg.header.frame_id = 'map'
        
        cmd_msg.control_mode = self.control_mode
        cmd_msg.vx = vx_rl
        cmd_msg.vy = vy_rl
        cmd_msg.vz = 0.0
        cmd_msg.omega = omega_rl
        
        cmd_msg.target_x = target_x
        cmd_msg.target_y = target_y
        cmd_msg.target_z = self.fixed_altitude
        
        cmd_msg.tracking_error = tracking_error
        cmd_msg.dt = dt_integration
        
        cmd_msg.agent_state = [
            self.uav_pos[0], self.uav_pos[1],
            self.uav_vel[0], self.uav_vel[1]
        ]
        cmd_msg.target_state = [
            rel_x, rel_y,
            self.target_vel[0] - self.uav_vel[0],
            self.target_vel[1] - self.uav_vel[1]
        ]
        
        self.cmd_publisher.publish(cmd_msg)
        
        # ───────────────────────────────────────────────────────────────
        # 6. Log Output
        # ───────────────────────────────────────────────────────────────
        self.log_rl_output(vx_rl, vy_rl, target_x, target_y, 
                          tracking_error, dt_integration)
        
        # ───────────────────────────────────────────────────────────────
        # 7. Publish Visualization
        # ───────────────────────────────────────────────────────────────
        self.publish_markers(vx_rl, vy_rl)
    
    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION BUILDING
    # ═══════════════════════════════════════════════════════════════════
    
    def build_observation(self):
        """Build dict observation for attention-based RL model"""
        
        # Agent state (2D)
        agent_x = self.uav_pos[0]
        agent_y = self.uav_pos[1]
        agent_vx = self.uav_vel[0]
        agent_vy = self.uav_vel[1]
        
        # Target state (relative)
        rel_x = self.target_pos[0] - agent_x
        rel_y = self.target_pos[1] - agent_y
        rel_vx = self.target_vel[0] - agent_vx
        rel_vy = self.target_vel[1] - agent_vy
        
        # Build dict observation (matches training format)
        obs = {
            'agent': np.array([agent_x, agent_y, agent_vx, agent_vy], dtype=np.float32),
            
            'targets': np.zeros((self.max_tracked_targets, 4), dtype=np.float32),
            
            'mask': np.zeros(self.max_tracked_targets, dtype=np.float32)
        }
        
        # Fill in first target (Howard)
        obs['targets'][0] = [rel_x, rel_y, rel_vx, rel_vy]
        obs['mask'][0] = 1.0
        
        return obs
    
    # ═══════════════════════════════════════════════════════════════════
    # POSITION INTEGRATION (MODE 2)
    # ═══════════════════════════════════════════════════════════════════
    
    def integrate_to_position(self, vx, vy, current_time):
        """Integrate velocity to position using Euler method"""
        
        # Initialize integrated position on first call
        if not self.integration_initialized:
            self.integrated_position = np.array([
                self.uav_pos[0],
                self.uav_pos[1]
            ])
            self.last_integration_time = current_time
            self.integration_initialized = True
            self.get_logger().info(f"🧮 Integration initialized at: {self.integrated_position}")
            return self.integrated_position[0], self.integrated_position[1], 0.0
        
        # Compute time step
        dt = (current_time - self.last_integration_time).nanoseconds / 1e9
        
        if dt <= 0 or dt > 1.0:  # Sanity check
            self.get_logger().warn(f"⚠️ Invalid dt: {dt}s, skipping integration")
            return self.integrated_position[0], self.integrated_position[1], 0.0
        
        # Euler integration: x_new = x_old + v * dt
        delta_x = vx * dt
        delta_y = vy * dt
        
        # Check for unreasonable jumps
        jump_distance = np.sqrt(delta_x**2 + delta_y**2)
        if jump_distance > self.max_position_jump:
            self.get_logger().warn(
                f"⚠️ Position jump too large: {jump_distance:.3f}m, "
                f"clamping to {self.max_position_jump}m"
            )
            scale = self.max_position_jump / jump_distance
            delta_x *= scale
            delta_y *= scale
        
        # Update integrated position
        self.integrated_position[0] += delta_x
        self.integrated_position[1] += delta_y
        
        # Clamp to boundaries (prevent drift)
        self.integrated_position[0] = np.clip(
            self.integrated_position[0],
            self.boundary_x_min,
            self.boundary_x_max
        )
        self.integrated_position[1] = np.clip(
            self.integrated_position[1],
            self.boundary_y_min,
            self.boundary_y_max
        )
        
        self.last_integration_time = current_time
        
        return self.integrated_position[0], self.integrated_position[1], dt
    
    # ═══════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════
    
    def log_rl_output(self, vx, vy, target_x, target_y, error, dt):
        """Log RL policy output to terminal"""
        
        timestamp = self.get_clock().now().nanoseconds
        
        self.get_logger().info("┌─────────────────────────────────────────────────────────┐")
        self.get_logger().info("│  🎮 RL POLICY OUTPUT                                    │")
        self.get_logger().info("├─────────────────────────────────────────────────────────┤")
        self.get_logger().info(f"│  Timestamp: {timestamp}")
        self.get_logger().info(f"│  Control Mode: {self.control_mode.upper()}")
        self.get_logger().info("│")
        self.get_logger().info(f"│  📍 Current State:")
        self.get_logger().info(f"│     UAV: x={self.uav_pos[0]:.3f}, y={self.uav_pos[1]:.3f}, z={self.uav_pos[2]:.3f}")
        self.get_logger().info(f"│     Target: x={self.target_pos[0]:.3f}, y={self.target_pos[1]:.3f}")
        self.get_logger().info(f"│     Tracking Error: {error:.3f}m")
        self.get_logger().info("│")
        self.get_logger().info(f"│  🤖 RL Decision:")
        self.get_logger().info(f"│     Velocity: vx={vx:.3f}, vy={vy:.3f} m/s")
        
        if self.control_mode == "position":
            self.get_logger().info("│")
            self.get_logger().info(f"│  🧮 Integration (dt={dt:.3f}s):")
            self.get_logger().info(f"│     Integrated Position: x={target_x:.3f}, y={target_y:.3f} m")
        
        self.get_logger().info("└─────────────────────────────────────────────────────────┘")
    
    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    def publish_markers(self, vx, vy):
        """Publish visualization markers"""
        markers = MarkerArray()
        
        # Target marker (orange cube)
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.header.stamp = self.get_clock().now().to_msg()
        target_marker.ns = "rl_policy"
        target_marker.id = 0
        target_marker.type = Marker.CUBE
        target_marker.action = Marker.ADD
        
        target_marker.pose.position.x = self.target_pos[0]
        target_marker.pose.position.y = self.target_pos[1]
        target_marker.pose.position.z = self.target_pos[2]
        target_marker.pose.orientation.w = 1.0
        
        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        
        target_marker.color.r = 1.0
        target_marker.color.g = 0.5
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        
        markers.markers.append(target_marker)
        
        # Action arrow (red)
        action_marker = Marker()
        action_marker.header.frame_id = "map"
        action_marker.header.stamp = self.get_clock().now().to_msg()
        action_marker.ns = "rl_policy"
        action_marker.id = 1
        action_marker.type = Marker.ARROW
        action_marker.action = Marker.ADD
        
        start = Point()
        start.x = self.uav_pos[0]
        start.y = self.uav_pos[1]
        start.z = self.uav_pos[2]
        
        scale = 2.0
        end = Point()
        end.x = start.x + vx * scale
        end.y = start.y + vy * scale
        end.z = start.z
        
        action_marker.points = [start, end]
        action_marker.scale.x = 0.05
        action_marker.scale.y = 0.1
        
        action_marker.color.r = 1.0
        action_marker.color.g = 0.0
        action_marker.color.b = 0.0
        action_marker.color.a = 1.0
        
        markers.markers.append(action_marker)
        
        if self.marker_publisher is not None:
            self.marker_publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = RLPolicyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
