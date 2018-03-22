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
include_directories(include ${catkin_INCLUDE_DIRS} ${gazebo_INCLUDE_DIRS})

add_library(randomize_material src/randomize_material.cpp)

add_executable(randomizer_node src/randomizer_node.cpp)
add_dependencies(randomizer_node ${catkin_EXPORTED_TARGETS})

target_link_libraries(randomizer_node ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h")

foreach(dir launch)
  install(DIRECTORY ${dir}/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach()

install(TARGETS
        randomizer_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
