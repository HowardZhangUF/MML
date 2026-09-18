# Pure RL Tracking System - Setup Guide

## 🎯 Overview

This two-node architecture separates RL inference from drone control, enabling clear verification of the data flow from RL policy → MAVROS → actual UAV commands.

### System Components

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────┐
│  rl_policy_node │────────>│ drone_commander  │────────>│  MAVROS  │
│  (RL inference) │         │ (MAVROS wrapper) │         │   PX4    │
└─────────────────┘         └──────────────────┘         └──────────┘
        │                            │
        │ /rl_policy/command         │ /mavros/setpoint_*
        └────────────────────────────┘
           (Observable in terminal)
```

### Control Modes

1. **MODE 1: Velocity Control** (Direct)
   - RL outputs velocity → Sent directly to `/mavros/setpoint_velocity/cmd_vel_unstamped`
   - Fast response, immediate control

2. **MODE 2: Position Control** (Integrated)
   - RL outputs velocity → Integrated to position → Sent to `/mavros/setpoint_position/local`
   - Smoother tracking, waypoint-based control

---

## 🏗️ Build Instructions

### Step 1: Clean Previous Build (if needed)

```bash
cd ~/mml_python_sim
rm -rf build/ install/ log/
```

### Step 2: Build the Package

```bash
cd ~/mml_python_sim
colcon build --packages-select mml_guidance --symlink-install
```

**Note:** The `--symlink-install` flag allows you to edit Python files without rebuilding.

### Step 3: Source the Workspace

```bash
source ~/mml_python_sim/install/setup.bash
```

**Add to ~/.bashrc for automatic sourcing:**
```bash
echo "source ~/mml_python_sim/install/setup.bash" >> ~/.bashrc
```

### Step 4: Verify Installation

```bash
# Check if message is built
ros2 interface show mml_guidance/msg/RLCommand

# Check if nodes are available
ros2 pkg executables mml_guidance | grep -E "(rl_policy|drone_commander|monitor)"
```

Expected output:
```
mml_guidance rl_policy_node
mml_guidance drone_commander
mml_guidance monitor_tracking
```

---

## 🚀 Usage

### Quick Start: Position Control Mode (Recommended)

```bash
source ~/mml_python_sim/install/setup.bash

ros2 launch mml_guidance pure_rl_tracking.launch.py \
    control_mode:=position \
    control_rate:=10.0
```

### Velocity Control Mode

```bash
ros2 launch mml_guidance pure_rl_tracking.launch.py \
    control_mode:=velocity \
    control_rate:=10.0
```

### Custom Model Path

```bash
ros2 launch mml_guidance pure_rl_tracking.launch.py \
    control_mode:=position \
    model_path:=/path/to/your/model.zip
```

---

## 🔍 Monitoring & Verification

### Method 1: Check Node Logs (Built-in)

The launch file automatically displays logs from both nodes in the main terminal.

**Terminal Output Structure:**
```
[rl_policy]: ┌─ RL POLICY OUTPUT ─┐
[rl_policy]: │  Control Mode: POSITION
[rl_policy]: │  RL Velocity: vx=0.34, vy=0.21
[rl_policy]: │  Integrated Position: x=1.234, y=0.456
[rl_policy]: └─────────────────────┘

[drone_commander]: ┌─ SENT TO MAVROS ─┐
[drone_commander]: │  Target Waypoint: x=1.234, y=0.456, z=1.400
[drone_commander]: │  Latency: 0.77ms
[drone_commander]: └───────────────────┘
```

### Method 2: Topic Echo (Separate Terminals)

**Terminal 1: RL Policy Output**
```bash
ros2 topic echo /rl_policy/command
```

**Terminal 2: MAVROS Position Setpoint (Position Mode)**
```bash
ros2 topic echo /mavros/setpoint_position/local
```

**Terminal 2: MAVROS Velocity Setpoint (Velocity Mode)**
```bash
ros2 topic echo /mavros/setpoint_velocity/cmd_vel_unstamped
```

**Terminal 3: Actual UAV Pose**
```bash
ros2 topic echo /pop/enu/pose
```

### Method 3: Real-Time Monitor Tool

Launch the custom monitoring tool:

```bash
# For position mode
ros2 run mml_guidance monitor_tracking --mode position

# For velocity mode
ros2 run mml_guidance monitor_tracking --mode velocity
```

This displays a live comparison of:
- RL policy output
- MAVROS setpoint
- Actual UAV position
- Latency measurements
- Mismatch warnings

---

## 📊 Verification Checklist

### ✅ Pre-Flight Checks

1. **ROS2 topics active:**
   ```bash
   ros2 topic list | grep -E "(rl_policy|mavros|howard)"
   ```
   Should see:
   - `/rl_policy/command`
   - `/rl_policy/markers`
   - `/mavros/setpoint_position/local` or `/mavros/setpoint_velocity/cmd_vel_unstamped`
   - `/howard/enu/pose`

2. **Node status:**
   ```bash
   ros2 node list
   ```
   Should see:
   - `/rl_policy`
   - `/drone_commander`
   - `/howard_sim`

3. **Message flow:**
   ```bash
   ros2 topic hz /rl_policy/command
   ```
   Should show ~10 Hz (or your configured control_rate)

### ✅ In-Flight Verification

1. **Check latency in logs:**
   - Look for "Latency: X.XXms" in drone_commander output
   - Should be < 5ms typically

2. **Verify position/velocity match:**
   - **Position mode:** RL integrated position should match MAVROS waypoint
   - **Velocity mode:** RL velocity should match MAVROS velocity

3. **Monitor tracking error:**
   - Check "Tracking Error" in RL policy output
   - Should decrease as UAV approaches target

---

## 🐛 Troubleshooting

### Issue: "RLCommand message not built"

**Solution:**
```bash
cd ~/mml_python_sim
colcon build --packages-select mml_guidance
source install/setup.bash
```

### Issue: "Model not found"

**Solution:** Verify model path exists:
```bash
ls -lh ~/mml_python_sim/models_recurrent/*.zip
```

Update launch file or pass correct path:
```bash
ros2 launch mml_guidance pure_rl_tracking.launch.py \
    model_path:=/full/path/to/model.zip
```

### Issue: "stable-baselines3 not installed"

**Solution:**
```bash
pip3 install stable-baselines3
```

### Issue: No output from nodes

**Solution:** Check if MAVROS is running:
```bash
ros2 topic list | grep mavros
```

If not, start MAVROS or run in simulation mode.

### Issue: High latency (>10ms)

**Possible causes:**
1. System load too high
2. Control rate too fast for hardware
3. Model inference too slow

**Solution:**
- Reduce control_rate: `control_rate:=5.0`
- Check CPU usage: `htop`
- Use GPU for inference if available

---

## 📁 File Structure

```
src/mml_guidance/
├── msg/
│   └── RLCommand.msg                  # Custom message definition
├── mml_guidance/
│   ├── rl_policy_node.py             # NODE 1: RL inference
│   ├── drone_commander.py            # NODE 2: MAVROS interface
│   ├── monitor_tracking.py           # Monitoring tool
│   └── Sim_turtle.py                 # Howard (target robot)
├── launch/
│   └── pure_rl_tracking.launch.py   # Main launch file
├── CMakeLists.txt                    # Build configuration
├── package.xml                       # Package metadata
└── setup.py                          # Python package setup
```

---

## 🔧 Parameters

### rl_policy_node Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_mode` | `'velocity'` | `'velocity'` or `'position'` |
| `control_rate` | `10.0` | Hz - Policy update frequency |
| `model_path` | Required | Path to trained model .zip file |
| `fixed_altitude` | `1.4` | meters - Target flight altitude |
| `max_tracked_targets` | `10` | Maximum targets in observation |
| `max_position_jump` | `0.5` | meters - Max integration jump per cycle |
| `boundary_x_min/max` | `-2.0/2.0` | meters - Safety boundaries |
| `boundary_y_min/max` | `-1.5/1.5` | meters - Safety boundaries |

### drone_commander Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_mode` | `'velocity'` | Must match rl_policy_node |
| `fixed_altitude` | `1.4` | meters - Altitude setpoint |
| `altitude_kp` | `0.5` | PD controller proportional gain |
| `altitude_kd` | `0.2` | PD controller derivative gain |
| `max_climb_rate` | `0.3` | m/s - Maximum vertical speed |
| `warning_distance` | `0.3` | meters - Boundary warning threshold |
| `emergency_distance` | `0.15` | meters - Emergency override threshold |
| `max_waypoint_distance` | `0.5` | meters - Max waypoint jump (position mode) |

---

## 📝 Important Notes

1. **Control Mode Consistency:** Both nodes must use the same `control_mode` parameter.

2. **Model Compatibility:** Ensure the loaded model expects observations in the format:
   ```python
   {
       'agent': [x, y, vx, vy],          # Shape: (4,)
       'targets': [[rel_x, rel_y, ...]], # Shape: (N, 4)
       'mask': [1.0, 0.0, ...]           # Shape: (N,)
   }
   ```

3. **Integration Drift (Position Mode):** The integrated position may drift over time. Monitor the tracking error.

4. **Safety Boundaries:** Always configure boundaries appropriate for your flight space.

---

## 🎓 Next Steps

1. **Test in Simulation First:** Use Gazebo/PX4 SITL before real hardware
2. **Monitor Terminals:** Always watch both node outputs during operation
3. **Record Data:** Use `ros2 bag record` for post-flight analysis
4. **Tune Parameters:** Adjust control_rate, altitude gains as needed

---

## 📞 Support

For issues or questions:
1. Check logs in both node terminals
2. Run monitor_tracking tool for detailed diagnostics
3. Verify all prerequisites are installed
4. Check ROS2 topic connections with `ros2 topic list`

---

**🎉 You're ready to track! Launch the system and monitor the terminal outputs to verify the RL policy → MAVROS data flow.**
