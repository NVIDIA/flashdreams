# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

cmake_minimum_required(VERSION 3.8)
project(flashdreams_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_module(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../apps/interactive_drive/interactive_drive/ros_endpoint.py"
  DESTINATION_SUFFIX flashdreams_bridge)
install(FILES camera_calibration.py cockpit_joy.py drive_diagnostics.py ego_visualization.py joy_driver.py
  model_logs.py scene_visualization.py DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
install(PROGRAMS bridge.py DESTINATION lib/${PROJECT_NAME} RENAME bridge)
install(PROGRAMS camera_preview.py DESTINATION lib/${PROJECT_NAME} RENAME camera_preview)
install(FILES launch/remote_joy.launch.py DESTINATION share/${PROJECT_NAME}/launch)
ament_package()
