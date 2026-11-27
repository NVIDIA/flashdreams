# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

cmake_minimum_required(VERSION 3.8)
project(flashdreams_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Control.msg"
  "msg/VehicleState.msg"
  "msg/CameraModel.msg"
  "srv/Reset.srv"
  "srv/Step.srv"
  DEPENDENCIES builtin_interfaces std_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
