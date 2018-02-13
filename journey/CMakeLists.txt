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

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${gazebo_INCLUDE_DIRS}
)

add_executable(randomizer_tools src/randomizer_tools.cpp)
target_link_libraries(randomizer_tools ${catkin_LIBRARIES})
