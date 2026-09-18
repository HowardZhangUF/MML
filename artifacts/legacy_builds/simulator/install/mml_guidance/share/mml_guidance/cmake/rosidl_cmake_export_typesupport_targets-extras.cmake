# generated from
# rosidl_cmake/cmake/template/rosidl_cmake_export_typesupport_targets.cmake.in

set(_exported_typesupport_targets
  "__rosidl_generator_c:mml_guidance__rosidl_generator_c;__rosidl_typesupport_fastrtps_c:mml_guidance__rosidl_typesupport_fastrtps_c;__rosidl_typesupport_introspection_c:mml_guidance__rosidl_typesupport_introspection_c;__rosidl_typesupport_c:mml_guidance__rosidl_typesupport_c;__rosidl_generator_cpp:mml_guidance__rosidl_generator_cpp;__rosidl_typesupport_fastrtps_cpp:mml_guidance__rosidl_typesupport_fastrtps_cpp;__rosidl_typesupport_introspection_cpp:mml_guidance__rosidl_typesupport_introspection_cpp;__rosidl_typesupport_cpp:mml_guidance__rosidl_typesupport_cpp;__rosidl_generator_py:mml_guidance__rosidl_generator_py")

# populate mml_guidance_TARGETS_<suffix>
if(NOT _exported_typesupport_targets STREQUAL "")
  # loop over typesupport targets
  foreach(_tuple ${_exported_typesupport_targets})
    string(REPLACE ":" ";" _tuple "${_tuple}")
    list(GET _tuple 0 _suffix)
    list(GET _tuple 1 _target)

    set(_target "mml_guidance::${_target}")
    if(NOT TARGET "${_target}")
      # the exported target must exist
      message(WARNING "Package 'mml_guidance' exports the typesupport target '${_target}' which doesn't exist")
    else()
      list(APPEND mml_guidance_TARGETS${_suffix} "${_target}")
    endif()
  endforeach()
endif()
