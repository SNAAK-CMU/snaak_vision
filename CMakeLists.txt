cmake_minimum_required(VERSION 3.5)
project(snaak_vision)

# Find dependencies
find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Add service files to be generated
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetDepthAtPoint.srv"
  "srv/GetXYZFromImage.srv"
  DEPENDENCIES std_msgs sensor_msgs
)

# Install Python scripts (nodes)
install(
  DIRECTORY src/
  USE_SOURCE_PERMISSIONS
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install other files like package.xml and srv files
install(
  DIRECTORY srv/
  DESTINATION share/${PROJECT_NAME}/srv
)

ament_package()
