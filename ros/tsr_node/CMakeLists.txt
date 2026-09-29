cmake_minimum_required(VERSION 3.8)
project(tsr_node)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-unresolved-symbols=ignore-in-shared-libs")
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)

# ---- messages ----
set(msg_files
  msg/Nv12Image.msg
  msg/SpeedLimitSign.msg
  msg/SpeedLimitSignArray.msg
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs builtin_interfaces
)

# ---- BPU dnn + headers ----
include_directories(/usr/include/dnn)
link_directories(/usr/lib)
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# ---- core library (detector + classifier + tracker + nv12) ----
add_library(tsr_core
  src/detector.cpp
  src/classifier.cpp
  src/tracker.cpp
)
target_include_directories(tsr_core PUBLIC include)
target_link_libraries(tsr_core ${OpenCV_LIBS} dnn pthread rt dl)

# ---- ROS node (executable named tsr_detect to avoid colliding with the
#      package/interface target "tsr_node") ----
add_executable(tsr_detect src/tsr_node.cpp)
target_link_libraries(tsr_detect tsr_core ${OpenCV_LIBS} dnn pthread rt dl)
ament_target_dependencies(tsr_detect rclcpp)
rosidl_target_interfaces(tsr_detect ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ---- replay node (test) ----
add_executable(tsr_replay src/tsr_replay.cpp)
target_link_libraries(tsr_replay ${OpenCV_LIBS})
ament_target_dependencies(tsr_replay rclcpp)
rosidl_target_interfaces(tsr_replay ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ---- Level 1 standalone test (no ROS) ----
add_executable(tsr_test test/test_tsr.cpp)
target_link_libraries(tsr_test tsr_core ${OpenCV_LIBS} dnn pthread rt dl)

install(TARGETS tsr_detect tsr_replay tsr_test
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
