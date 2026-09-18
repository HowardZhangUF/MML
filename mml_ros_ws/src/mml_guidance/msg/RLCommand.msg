# Custom message for RL policy commands
# Used to communicate between rl_policy_node and drone_commander

std_msgs/Header header

# Control mode selection
string control_mode           # "velocity" or "position"

# RL velocity output (always present from model)
float32 vx                    # Velocity X (m/s)
float32 vy                    # Velocity Y (m/s)
float32 vz                    # Velocity Z (m/s) - usually 0.0 from RL
float32 omega                 # Yaw rate (rad/s) - ignored in current implementation

# Integrated position (only valid when control_mode="position")
float32 target_x              # meters (NaN if velocity mode)
float32 target_y              # meters (NaN if velocity mode)
float32 target_z              # meters (altitude)

# Metadata for debugging and monitoring
float32 tracking_error        # Distance to target (m)
float32 dt                    # Integration time step (s) - only for position mode
float32[4] agent_state        # [x, y, vx, vy] - agent state
float32[4] target_state       # [rel_x, rel_y, rel_vx, rel_vy] - relative target state
