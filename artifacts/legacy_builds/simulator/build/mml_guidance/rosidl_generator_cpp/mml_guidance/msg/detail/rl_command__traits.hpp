// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mml_guidance:msg/RLCommand.idl
// generated code does not contain a copyright notice

#ifndef MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__TRAITS_HPP_
#define MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mml_guidance/msg/detail/rl_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace mml_guidance
{

namespace msg
{

inline void to_flow_style_yaml(
  const RLCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: control_mode
  {
    out << "control_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.control_mode, out);
    out << ", ";
  }

  // member: vx
  {
    out << "vx: ";
    rosidl_generator_traits::value_to_yaml(msg.vx, out);
    out << ", ";
  }

  // member: vy
  {
    out << "vy: ";
    rosidl_generator_traits::value_to_yaml(msg.vy, out);
    out << ", ";
  }

  // member: vz
  {
    out << "vz: ";
    rosidl_generator_traits::value_to_yaml(msg.vz, out);
    out << ", ";
  }

  // member: omega
  {
    out << "omega: ";
    rosidl_generator_traits::value_to_yaml(msg.omega, out);
    out << ", ";
  }

  // member: target_x
  {
    out << "target_x: ";
    rosidl_generator_traits::value_to_yaml(msg.target_x, out);
    out << ", ";
  }

  // member: target_y
  {
    out << "target_y: ";
    rosidl_generator_traits::value_to_yaml(msg.target_y, out);
    out << ", ";
  }

  // member: target_z
  {
    out << "target_z: ";
    rosidl_generator_traits::value_to_yaml(msg.target_z, out);
    out << ", ";
  }

  // member: tracking_error
  {
    out << "tracking_error: ";
    rosidl_generator_traits::value_to_yaml(msg.tracking_error, out);
    out << ", ";
  }

  // member: dt
  {
    out << "dt: ";
    rosidl_generator_traits::value_to_yaml(msg.dt, out);
    out << ", ";
  }

  // member: agent_state
  {
    if (msg.agent_state.size() == 0) {
      out << "agent_state: []";
    } else {
      out << "agent_state: [";
      size_t pending_items = msg.agent_state.size();
      for (auto item : msg.agent_state) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_state
  {
    if (msg.target_state.size() == 0) {
      out << "target_state: []";
    } else {
      out << "target_state: [";
      size_t pending_items = msg.target_state.size();
      for (auto item : msg.target_state) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RLCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: control_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "control_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.control_mode, out);
    out << "\n";
  }

  // member: vx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vx: ";
    rosidl_generator_traits::value_to_yaml(msg.vx, out);
    out << "\n";
  }

  // member: vy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vy: ";
    rosidl_generator_traits::value_to_yaml(msg.vy, out);
    out << "\n";
  }

  // member: vz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vz: ";
    rosidl_generator_traits::value_to_yaml(msg.vz, out);
    out << "\n";
  }

  // member: omega
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "omega: ";
    rosidl_generator_traits::value_to_yaml(msg.omega, out);
    out << "\n";
  }

  // member: target_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_x: ";
    rosidl_generator_traits::value_to_yaml(msg.target_x, out);
    out << "\n";
  }

  // member: target_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_y: ";
    rosidl_generator_traits::value_to_yaml(msg.target_y, out);
    out << "\n";
  }

  // member: target_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_z: ";
    rosidl_generator_traits::value_to_yaml(msg.target_z, out);
    out << "\n";
  }

  // member: tracking_error
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tracking_error: ";
    rosidl_generator_traits::value_to_yaml(msg.tracking_error, out);
    out << "\n";
  }

  // member: dt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dt: ";
    rosidl_generator_traits::value_to_yaml(msg.dt, out);
    out << "\n";
  }

  // member: agent_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.agent_state.size() == 0) {
      out << "agent_state: []\n";
    } else {
      out << "agent_state:\n";
      for (auto item : msg.agent_state) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_state.size() == 0) {
      out << "target_state: []\n";
    } else {
      out << "target_state:\n";
      for (auto item : msg.target_state) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RLCommand & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace mml_guidance

namespace rosidl_generator_traits
{

[[deprecated("use mml_guidance::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mml_guidance::msg::RLCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  mml_guidance::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mml_guidance::msg::to_yaml() instead")]]
inline std::string to_yaml(const mml_guidance::msg::RLCommand & msg)
{
  return mml_guidance::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mml_guidance::msg::RLCommand>()
{
  return "mml_guidance::msg::RLCommand";
}

template<>
inline const char * name<mml_guidance::msg::RLCommand>()
{
  return "mml_guidance/msg/RLCommand";
}

template<>
struct has_fixed_size<mml_guidance::msg::RLCommand>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mml_guidance::msg::RLCommand>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mml_guidance::msg::RLCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MML_GUIDANCE__MSG__DETAIL__RL_COMMAND__TRAITS_HPP_
