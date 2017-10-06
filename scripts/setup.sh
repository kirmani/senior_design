#! /bin/sh
#
# setup.sh
# Copyright (C) 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
#

PROJECT_DIR=`pwd -P`/..

# Set-up githooks.
git config core.hooksPath .githooks

# Initialize and update submodules.
git submodule update --init --recursive

# Link catkin workspace.
ln -s $PROJECT_DIR/catkin_ws ~/catkin_ws

# Install ROS.
echo "TODO(kirmani): Add script to set-up ROS"

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
