// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice

#ifndef MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__STRUCT_HPP_
#define MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__mml_guidance__msg__RLCommand __attribute__((deprecated))
#else
# define DEPRECATED__mml_guidance__msg__RLCommand __declspec(deprecated)
#endif

namespace mml_guidance
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RLCommand_
{
  using Type = RLCommand_<ContainerAllocator>;

  explicit RLCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->control_mode = "";
      this->vx = 0.0f;
      this->vy = 0.0f;
      this->vz = 0.0f;
      this->omega = 0.0f;
      this->target_x = 0.0f;
      this->target_y = 0.0f;
      this->target_z = 0.0f;
      this->tracking_error = 0.0f;
      this->dt = 0.0f;
      std::fill<typename std::array<float, 4>::iterator, float>(this->agent_state.begin(), this->agent_state.end(), 0.0f);
      std::fill<typename std::array<float, 4>::iterator, float>(this->target_state.begin(), this->target_state.end(), 0.0f);
    }
  }

  explicit RLCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    control_mode(_alloc),
    agent_state(_alloc),
    target_state(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->control_mode = "";
      this->vx = 0.0f;
      this->vy = 0.0f;
      this->vz = 0.0f;
      this->omega = 0.0f;
      this->target_x = 0.0f;
      this->target_y = 0.0f;
      this->target_z = 0.0f;
      this->tracking_error = 0.0f;
      this->dt = 0.0f;
      std::fill<typename std::array<float, 4>::iterator, float>(this->agent_state.begin(), this->agent_state.end(), 0.0f);
      std::fill<typename std::array<float, 4>::iterator, float>(this->target_state.begin(), this->target_state.end(), 0.0f);
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _control_mode_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _control_mode_type control_mode;
  using _vx_type =
    float;
  _vx_type vx;
  using _vy_type =
    float;
  _vy_type vy;
  using _vz_type =
    float;
  _vz_type vz;
  using _omega_type =
    float;
  _omega_type omega;
  using _target_x_type =
    float;
  _target_x_type target_x;
  using _target_y_type =
    float;
  _target_y_type target_y;
  using _target_z_type =
    float;
  _target_z_type target_z;
  using _tracking_error_type =
    float;
  _tracking_error_type tracking_error;
  using _dt_type =
    float;
  _dt_type dt;
  using _agent_state_type =
    std::array<float, 4>;
  _agent_state_type agent_state;
  using _target_state_type =
    std::array<float, 4>;
  _target_state_type target_state;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__control_mode(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->control_mode = _arg;
    return *this;
  }
  Type & set__vx(
    const float & _arg)
  {
    this->vx = _arg;
    return *this;
  }
  Type & set__vy(
    const float & _arg)
  {
    this->vy = _arg;
    return *this;
  }
  Type & set__vz(
    const float & _arg)
  {
    this->vz = _arg;
    return *this;
  }
  Type & set__omega(
    const float & _arg)
  {
    this->omega = _arg;
    return *this;
  }
  Type & set__target_x(
    const float & _arg)
  {
    this->target_x = _arg;
    return *this;
  }
  Type & set__target_y(
    const float & _arg)
  {
    this->target_y = _arg;
    return *this;
  }
  Type & set__target_z(
    const float & _arg)
  {
    this->target_z = _arg;
    return *this;
  }
  Type & set__tracking_error(
    const float & _arg)
  {
    this->tracking_error = _arg;
    return *this;
  }
  Type & set__dt(
    const float & _arg)
  {
    this->dt = _arg;
    return *this;
  }
  Type & set__agent_state(
    const std::array<float, 4> & _arg)
  {
    this->agent_state = _arg;
    return *this;
  }
  Type & set__target_state(
    const std::array<float, 4> & _arg)
  {
    this->target_state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mml_guidance::msg::RLCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const mml_guidance::msg::RLCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mml_guidance::msg::RLCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mml_guidance::msg::RLCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mml_guidance__msg__RLCommand
    std::shared_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mml_guidance__msg__RLCommand
    std::shared_ptr<mml_guidance::msg::RLCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RLCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->control_mode != other.control_mode) {
      return false;
    }
    if (this->vx != other.vx) {
      return false;
    }
    if (this->vy != other.vy) {
      return false;
    }
    if (this->vz != other.vz) {
      return false;
    }
    if (this->omega != other.omega) {
      return false;
    }
    if (this->target_x != other.target_x) {
      return false;
    }
    if (this->target_y != other.target_y) {
      return false;
    }
    if (this->target_z != other.target_z) {
      return false;
    }
    if (this->tracking_error != other.tracking_error) {
      return false;
    }
    if (this->dt != other.dt) {
      return false;
    }
    if (this->agent_state != other.agent_state) {
      return false;
    }
    if (this->target_state != other.target_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const RLCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RLCommand_

// alias to use template instance with default allocator
using RLCommand =
  mml_guidance::msg::RLCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mml_guidance

#endif  // MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__STRUCT_HPP_
