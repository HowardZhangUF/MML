// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice
#include "mml_guidance/msg/detail/rl_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `control_mode`
#include "rosidl_runtime_c/string_functions.h"

bool
mml_guidance__msg__RLCommand__init(mml_guidance__msg__RLCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    mml_guidance__msg__RLCommand__fini(msg);
    return false;
  }
  // control_mode
  if (!rosidl_runtime_c__String__init(&msg->control_mode)) {
    mml_guidance__msg__RLCommand__fini(msg);
    return false;
  }
  // vx
  // vy
  // vz
  // omega
  // target_x
  // target_y
  // target_z
  // tracking_error
  // dt
  // agent_state
  // target_state
  return true;
}

void
mml_guidance__msg__RLCommand__fini(mml_guidance__msg__RLCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // control_mode
  rosidl_runtime_c__String__fini(&msg->control_mode);
  // vx
  // vy
  // vz
  // omega
  // target_x
  // target_y
  // target_z
  // tracking_error
  // dt
  // agent_state
  // target_state
}

bool
mml_guidance__msg__RLCommand__are_equal(const mml_guidance__msg__RLCommand * lhs, const mml_guidance__msg__RLCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // control_mode
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->control_mode), &(rhs->control_mode)))
  {
    return false;
  }
  // vx
  if (lhs->vx != rhs->vx) {
    return false;
  }
  // vy
  if (lhs->vy != rhs->vy) {
    return false;
  }
  // vz
  if (lhs->vz != rhs->vz) {
    return false;
  }
  // omega
  if (lhs->omega != rhs->omega) {
    return false;
  }
  // target_x
  if (lhs->target_x != rhs->target_x) {
    return false;
  }
  // target_y
  if (lhs->target_y != rhs->target_y) {
    return false;
  }
  // target_z
  if (lhs->target_z != rhs->target_z) {
    return false;
  }
  // tracking_error
  if (lhs->tracking_error != rhs->tracking_error) {
    return false;
  }
  // dt
  if (lhs->dt != rhs->dt) {
    return false;
  }
  // agent_state
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->agent_state[i] != rhs->agent_state[i]) {
      return false;
    }
  }
  // target_state
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->target_state[i] != rhs->target_state[i]) {
      return false;
    }
  }
  return true;
}

bool
mml_guidance__msg__RLCommand__copy(
  const mml_guidance__msg__RLCommand * input,
  mml_guidance__msg__RLCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // control_mode
  if (!rosidl_runtime_c__String__copy(
      &(input->control_mode), &(output->control_mode)))
  {
    return false;
  }
  // vx
  output->vx = input->vx;
  // vy
  output->vy = input->vy;
  // vz
  output->vz = input->vz;
  // omega
  output->omega = input->omega;
  // target_x
  output->target_x = input->target_x;
  // target_y
  output->target_y = input->target_y;
  // target_z
  output->target_z = input->target_z;
  // tracking_error
  output->tracking_error = input->tracking_error;
  // dt
  output->dt = input->dt;
  // agent_state
  for (size_t i = 0; i < 4; ++i) {
    output->agent_state[i] = input->agent_state[i];
  }
  // target_state
  for (size_t i = 0; i < 4; ++i) {
    output->target_state[i] = input->target_state[i];
  }
  return true;
}

mml_guidance__msg__RLCommand *
mml_guidance__msg__RLCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mml_guidance__msg__RLCommand * msg = (mml_guidance__msg__RLCommand *)allocator.allocate(sizeof(mml_guidance__msg__RLCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mml_guidance__msg__RLCommand));
  bool success = mml_guidance__msg__RLCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mml_guidance__msg__RLCommand__destroy(mml_guidance__msg__RLCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mml_guidance__msg__RLCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mml_guidance__msg__RLCommand__Sequence__init(mml_guidance__msg__RLCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mml_guidance__msg__RLCommand * data = NULL;

  if (size) {
    data = (mml_guidance__msg__RLCommand *)allocator.zero_allocate(size, sizeof(mml_guidance__msg__RLCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mml_guidance__msg__RLCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mml_guidance__msg__RLCommand__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
mml_guidance__msg__RLCommand__Sequence__fini(mml_guidance__msg__RLCommand__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      mml_guidance__msg__RLCommand__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

mml_guidance__msg__RLCommand__Sequence *
mml_guidance__msg__RLCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mml_guidance__msg__RLCommand__Sequence * array = (mml_guidance__msg__RLCommand__Sequence *)allocator.allocate(sizeof(mml_guidance__msg__RLCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mml_guidance__msg__RLCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mml_guidance__msg__RLCommand__Sequence__destroy(mml_guidance__msg__RLCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mml_guidance__msg__RLCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mml_guidance__msg__RLCommand__Sequence__are_equal(const mml_guidance__msg__RLCommand__Sequence * lhs, const mml_guidance__msg__RLCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mml_guidance__msg__RLCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mml_guidance__msg__RLCommand__Sequence__copy(
  const mml_guidance__msg__RLCommand__Sequence * input,
  mml_guidance__msg__RLCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mml_guidance__msg__RLCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mml_guidance__msg__RLCommand * data =
      (mml_guidance__msg__RLCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mml_guidance__msg__RLCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mml_guidance__msg__RLCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mml_guidance__msg__RLCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
