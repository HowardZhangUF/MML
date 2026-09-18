// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice

#ifndef MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__STRUCT_H_
#define MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'control_mode'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/RLCommand in the package mml_guidance.
/**
  * Custom message for RL policy commands
  * Used to communicate between rl_policy_node and drone_commander
 */
typedef struct mml_guidance__msg__RLCommand
{
  std_msgs__msg__Header header;
  /// Control mode selection
  /// "velocity" or "position"
  rosidl_runtime_c__String control_mode;
  /// RL velocity output (always present from model)
  /// Velocity X (m/s)
  float vx;
  /// Velocity Y (m/s)
  float vy;
  /// Velocity Z (m/s) - usually 0.0 from RL
  float vz;
  /// Yaw rate (rad/s) - ignored in current implementation
  float omega;
  /// Integrated position (only valid when control_mode="position")
  /// meters (NaN if velocity mode)
  float target_x;
  /// meters (NaN if velocity mode)
  float target_y;
  /// meters (altitude)
  float target_z;
  /// Metadata for debugging and monitoring
  /// Distance to target (m)
  float tracking_error;
  /// Integration time step (s) - only for position mode
  float dt;
  /// [x, y, vx, vy] - agent state
  float agent_state[4];
  /// [rel_x, rel_y, rel_vx, rel_vy] - relative target state
  float target_state[4];
} mml_guidance__msg__RLCommand;

// Struct for a sequence of mml_guidance__msg__RLCommand.
typedef struct mml_guidance__msg__RLCommand__Sequence
{
  mml_guidance__msg__RLCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mml_guidance__msg__RLCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__STRUCT_H_
