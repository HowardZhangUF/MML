// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice

#ifndef MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__FUNCTIONS_H_
#define MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "mml_guidance/msg/rosidl_generator_c__visibility_control.h"

#include "mml_guidance/msg/detail/rl_command__struct.h"

/// Initialize msg/RLCommand message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mml_guidance__msg__RLCommand
 * )) before or use
 * mml_guidance__msg__RLCommand__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
bool
mml_guidance__msg__RLCommand__init(mml_guidance__msg__RLCommand * msg);

/// Finalize msg/RLCommand message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
void
mml_guidance__msg__RLCommand__fini(mml_guidance__msg__RLCommand * msg);

/// Create msg/RLCommand message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mml_guidance__msg__RLCommand__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
mml_guidance__msg__RLCommand *
mml_guidance__msg__RLCommand__create();

/// Destroy msg/RLCommand message.
/**
 * It calls
 * mml_guidance__msg__RLCommand__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
void
mml_guidance__msg__RLCommand__destroy(mml_guidance__msg__RLCommand * msg);

/// Check for msg/RLCommand message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
bool
mml_guidance__msg__RLCommand__are_equal(const mml_guidance__msg__RLCommand * lhs, const mml_guidance__msg__RLCommand * rhs);

/// Copy a msg/RLCommand message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
bool
mml_guidance__msg__RLCommand__copy(
  const mml_guidance__msg__RLCommand * input,
  mml_guidance__msg__RLCommand * output);

/// Initialize array of msg/RLCommand messages.
/**
 * It allocates the memory for the number of elements and calls
 * mml_guidance__msg__RLCommand__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
bool
mml_guidance__msg__RLCommand__Sequence__init(mml_guidance__msg__RLCommand__Sequence * array, size_t size);

/// Finalize array of msg/RLCommand messages.
/**
 * It calls
 * mml_guidance__msg__RLCommand__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
void
mml_guidance__msg__RLCommand__Sequence__fini(mml_guidance__msg__RLCommand__Sequence * array);

/// Create array of msg/RLCommand messages.
/**
 * It allocates the memory for the array and calls
 * mml_guidance__msg__RLCommand__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
mml_guidance__msg__RLCommand__Sequence *
mml_guidance__msg__RLCommand__Sequence__create(size_t size);

/// Destroy array of msg/RLCommand messages.
/**
 * It calls
 * mml_guidance__msg__RLCommand__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
void
mml_guidance__msg__RLCommand__Sequence__destroy(mml_guidance__msg__RLCommand__Sequence * array);

/// Check for msg/RLCommand message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
bool
mml_guidance__msg__RLCommand__Sequence__are_equal(const mml_guidance__msg__RLCommand__Sequence * lhs, const mml_guidance__msg__RLCommand__Sequence * rhs);

/// Copy an array of msg/RLCommand messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_mml_guidance
bool
mml_guidance__msg__RLCommand__Sequence__copy(
  const mml_guidance__msg__RLCommand__Sequence * input,
  mml_guidance__msg__RLCommand__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__FUNCTIONS_H_
