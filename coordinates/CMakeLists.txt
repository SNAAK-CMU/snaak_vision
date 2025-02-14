cmake_minimum_required(VERSION 3.5)
project(coordinates)

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
  DEPENDENCIES std_msgs sensor_msgs
)

# Install Python scripts (nodes)
install(
  PROGRAMS src/depth_service.py
  DESTINATION lib/${PROJECT_NAME}
)



# Install other files like package.xml and srv files
install(
  DIRECTORY srv/
  DESTINATION share/${PROJECT_NAME}/srv
)

ament_package()
