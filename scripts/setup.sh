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

# Install ardrone_autonomy.
sudo apt-get -y install ros-indigo-ardrone-autonomy
sudo apt-get -y install freeglut3-dev
sudo apt-get -y install liblapack-dev
sudo apt-get -y install libopenblas-dev

# Upgrade to Gazebo 7.
sudo apt-get -y purge gazebo*
sudo apt-get -y install ros-indigo-gazebo7-ros-pkgs ros-indigo-gazebo7-ros-control

# Build catkin workspace.
cd ~/catkin_ws
source /opt/ros/indigo/setup.bash
catkin_make
