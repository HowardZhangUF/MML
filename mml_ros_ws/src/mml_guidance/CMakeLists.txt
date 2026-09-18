cmake_minimum_required(VERSION 3.10)
project(mml_guidance)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RLCommand.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python package and executables using custom command
install(CODE "
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}:\$ENV{PYTHONPATH}
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install
        --prefix=${CMAKE_INSTALL_PREFIX}
        --install-lib=${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}
        --install-data=${CMAKE_INSTALL_PREFIX}
        --single-version-externally-managed
        --record=${CMAKE_CURRENT_BINARY_DIR}/python-install-record.txt
        --install-scripts=${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE python_install_status
  )
  if(NOT python_install_status EQUAL 0)
    message(FATAL_ERROR \"Python package installation failed\")
  endif()
")

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()