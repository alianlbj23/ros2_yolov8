cmake_minimum_required(VERSION 3.5)
project(yolo_pkg)

# 查找所需的 ROS 2 包
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

# 安装 Python 脚本
install(PROGRAMS
  scripts/face_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装资源文件
install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
  FILES_MATCHING PATTERN "*.xml"
)

ament_package()
