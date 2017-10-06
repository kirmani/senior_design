#! /bin/sh
#
# setup.sh
# Copyright (C) 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
#

PROJECT_DIR=`pwd -P`

# Set-up githooks.
git config core.hooksPath .githooks

# Initialize and update submodules.
git submodule update --init --recursive

# Link catkin workspace.
unlink ~/catkin_ws
ln -s $PROJECT_DIR/../catkin_ws ~/catkin_ws

# Install ROS.
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
sudo apt-get update
sudo apt-get -y install ros-indigo-desktop-full
sudo rosdep init
rosdep update
echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get -y install python-rosinstall
sudo apt-get -y freeglut3-dev

# Build ardrone_autonomy.
cd ~/catkin_ws/src
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:`pwd`/ardrone_autonomy
cd ~/catkin_ws/src/ardrone_autonomy
./build_sdk.sh
rosmake

# Build tum_ardrone
cd ~/catkin_ws/src
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:`pwd`/tum_ardrone
cd ~/catkin_ws/src/tum_ardrone
rosmake tum_ardrone

# Build catkin workspace.
cd ~/catkin_ws
catkin_make
