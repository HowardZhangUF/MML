// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice

#ifndef MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__BUILDER_HPP_
#define MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mml_guidance/msg/detail/rl_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mml_guidance
{

namespace msg
{

namespace builder
{

class Init_RLCommand_target_state
{
public:
  explicit Init_RLCommand_target_state(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  ::mml_guidance::msg::RLCommand target_state(::mml_guidance::msg::RLCommand::_target_state_type arg)
  {
    msg_.target_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_agent_state
{
public:
  explicit Init_RLCommand_agent_state(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_target_state agent_state(::mml_guidance::msg::RLCommand::_agent_state_type arg)
  {
    msg_.agent_state = std::move(arg);
    return Init_RLCommand_target_state(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_dt
{
public:
  explicit Init_RLCommand_dt(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_agent_state dt(::mml_guidance::msg::RLCommand::_dt_type arg)
  {
    msg_.dt = std::move(arg);
    return Init_RLCommand_agent_state(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_tracking_error
{
public:
  explicit Init_RLCommand_tracking_error(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_dt tracking_error(::mml_guidance::msg::RLCommand::_tracking_error_type arg)
  {
    msg_.tracking_error = std::move(arg);
    return Init_RLCommand_dt(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_target_z
{
public:
  explicit Init_RLCommand_target_z(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_tracking_error target_z(::mml_guidance::msg::RLCommand::_target_z_type arg)
  {
    msg_.target_z = std::move(arg);
    return Init_RLCommand_tracking_error(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_target_y
{
public:
  explicit Init_RLCommand_target_y(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_target_z target_y(::mml_guidance::msg::RLCommand::_target_y_type arg)
  {
    msg_.target_y = std::move(arg);
    return Init_RLCommand_target_z(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_target_x
{
public:
  explicit Init_RLCommand_target_x(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_target_y target_x(::mml_guidance::msg::RLCommand::_target_x_type arg)
  {
    msg_.target_x = std::move(arg);
    return Init_RLCommand_target_y(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_omega
{
public:
  explicit Init_RLCommand_omega(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_target_x omega(::mml_guidance::msg::RLCommand::_omega_type arg)
  {
    msg_.omega = std::move(arg);
    return Init_RLCommand_target_x(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_vz
{
public:
  explicit Init_RLCommand_vz(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_omega vz(::mml_guidance::msg::RLCommand::_vz_type arg)
  {
    msg_.vz = std::move(arg);
    return Init_RLCommand_omega(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_vy
{
public:
  explicit Init_RLCommand_vy(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_vz vy(::mml_guidance::msg::RLCommand::_vy_type arg)
  {
    msg_.vy = std::move(arg);
    return Init_RLCommand_vz(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_vx
{
public:
  explicit Init_RLCommand_vx(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_vy vx(::mml_guidance::msg::RLCommand::_vx_type arg)
  {
    msg_.vx = std::move(arg);
    return Init_RLCommand_vy(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_control_mode
{
public:
  explicit Init_RLCommand_control_mode(::mml_guidance::msg::RLCommand & msg)
  : msg_(msg)
  {}
  Init_RLCommand_vx control_mode(::mml_guidance::msg::RLCommand::_control_mode_type arg)
  {
    msg_.control_mode = std::move(arg);
    return Init_RLCommand_vx(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

class Init_RLCommand_header
{
public:
  Init_RLCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RLCommand_control_mode header(::mml_guidance::msg::RLCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RLCommand_control_mode(msg_);
  }

private:
  ::mml_guidance::msg::RLCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mml_guidance::msg::RLCommand>()
{
  return mml_guidance::msg::builder::Init_RLCommand_header();
}

}  // namespace mml_guidance

#endif  // MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__BUILDER_HPP_
