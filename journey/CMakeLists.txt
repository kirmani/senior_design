cmake_minimum_required(VERSION 2.8.3)
project(journey)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  ros_numpy
  std_msgs
  gazebo_dev
)

add_message_files(
  FILES
  CollisionState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime gazebo_dev
)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${catkin_INCLUDE_DIRS} ${gazebo_INCLUDE_DIRS})

add_library(randomize_material src/randomize_material.cpp)
target_link_libraries(randomize_material ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(randomize_light src/randomize_light.cpp)
target_link_libraries(randomize_light ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_executable(randomizer_node src/randomizer_node.cpp)
target_link_libraries(randomizer_node ${catkin_LIBRARIES})