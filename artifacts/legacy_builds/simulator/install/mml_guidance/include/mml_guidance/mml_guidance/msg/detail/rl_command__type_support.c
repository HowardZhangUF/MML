// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mml_guidance/msg/detail/rl_command__rosidl_typesupport_introspection_c.h"
#include "mml_guidance/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mml_guidance/msg/detail/rl_command__functions.h"
#include "mml_guidance/msg/detail/rl_command__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `control_mode`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mml_guidance__msg__RLCommand__init(message_memory);
}

void mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_fini_function(void * message_memory)
{
  mml_guidance__msg__RLCommand__fini(message_memory);
}

size_t mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__size_function__RLCommand__agent_state(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_const_function__RLCommand__agent_state(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_function__RLCommand__agent_state(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__fetch_function__RLCommand__agent_state(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_const_function__RLCommand__agent_state(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__assign_function__RLCommand__agent_state(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_function__RLCommand__agent_state(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

size_t mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__size_function__RLCommand__target_state(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_const_function__RLCommand__target_state(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_function__RLCommand__target_state(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__fetch_function__RLCommand__target_state(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_const_function__RLCommand__target_state(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__assign_function__RLCommand__target_state(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_function__RLCommand__target_state(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_member_array[13] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "control_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, control_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, vx),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vy",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, vy),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vz",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, vz),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "omega",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, omega),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, target_x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, target_y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_z",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, target_z),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "tracking_error",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, tracking_error),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "dt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, dt),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "agent_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, agent_state),  // bytes offset in struct
    NULL,  // default value
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__size_function__RLCommand__agent_state,  // size() function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_const_function__RLCommand__agent_state,  // get_const(index) function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_function__RLCommand__agent_state,  // get(index) function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__fetch_function__RLCommand__agent_state,  // fetch(index, &value) function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__assign_function__RLCommand__agent_state,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(mml_guidance__msg__RLCommand, target_state),  // bytes offset in struct
    NULL,  // default value
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__size_function__RLCommand__target_state,  // size() function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_const_function__RLCommand__target_state,  // get_const(index) function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__get_function__RLCommand__target_state,  // get(index) function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__fetch_function__RLCommand__target_state,  // fetch(index, &value) function pointer
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__assign_function__RLCommand__target_state,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_members = {
  "mml_guidance__msg",  // message namespace
  "RLCommand",  // message name
  13,  // number of fields
  sizeof(mml_guidance__msg__RLCommand),
  mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_member_array,  // message members
  mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_type_support_handle = {
  0,
  &mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mml_guidance
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mml_guidance, msg, RLCommand)() {
  mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_type_support_handle.typesupport_identifier) {
    mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mml_guidance__msg__RLCommand__rosidl_typesupport_introspection_c__RLCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
