#!/bin/sh
#
# setup.sh
# Copyright (C) 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
#
# This script sets up the build environment.
PROJECT_DIR=`pwd -P`

# Set-up githooks.
git config core.hooksPath .githooks

# Initialize and update submodules.
git submodule update --init --recursive

# Create a journey workspace.
mkdir -p ~/journey_ws/src

# Install ROS.
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  # Setup your computer to accept software from packages.ros.org.
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

  # Set up your keys.
  sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

  # Make sure your Debian package index is up-to-date.
  sudo apt-get update

  # Desktop-Full Install: (Recommended) : ROS, rqt, rviz, robot-generic
  # libraries, 2D/3D simulators, navigation and 2D/3D perception
  sudo apt-get -y install ros-kinetic-desktop-full

  # In order to use rosdep, we have to initialize it.
  sudo rosdep init
  rosdep update

  # Up to now you have installed what you need to run the core ROS packages. To
  # create and manage your own ROS workspaces, there are various tools and
  # requirements that are distributed separately. For example, rosinstall is a
  # frequently used command-line tool that enables you to easily download many
  # source trees for ROS packages with one command.

  # To install this tool and other dependencies for building ROS packages, run:
  sudo apt-get -y install python-rosinstall python-rosinstall-generator python-wstool build-essential python-catkin-tools

  # Install ardrone_autonomy.
  sudo apt-get -y install ros-kinetic-ardrone-autonomy freeglut3-dev liblapack-dev libopenblas-dev

  # Link this project to your journey workspace.
  unlink ~/journey_ws/src/journey
  ln -s $PROJECT_DIR ~/journey_ws/src/journey

  # # Add some convenient bash commands.
  echo "source ~/journey_ws/src/journey/journey.bash" >> ~/.bashrc
  bash ~/.bashrc

  # Build journey workspace.
  cd ~/journey_ws
  catkin init
  catkin clean --yes
  cd src
  wstool update
  rosdep install tum_ardrone
  cd ..
  catkin build
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # Use homebrew to install additional software.
  brew update
  brew install cmake

  # Add our ROS Indigo tap and the Homebrew science tap so you can get some
  # non-standard formulae.
  brew tap ros/deps
  brew tap osrf/simulation  # Gazebo, sdformat, and ogre
  brew tap homebrew/versions # VTK5
  brew tap homebrew/science  # others

  # Export environment to path.
  export PATH=/usr/local/bin:$PATH

  # Tell python about modules installed with homebrew.
  mkdir -p ~/Library/Python/2.7/lib/python/site-packages
  echo "$(brew --prefix)/lib/python2.7/site-packages" >> ~/Library/Python/2.7/lib/python/site-packages/homebrew.pth

  # If you don't already have pip, install it with:
  brew install python
  sudo -H python2 -m pip install -U pip  # Update pip

  # Install ROS using pip.
  sudo -H python2 -m pip install -U wstool rosdep rosinstall rosinstall_generator rospkg catkin-pkg sphinx

  # In order to use rosdep, we have to initialize it.
  sudo rosdep init
  rosdep update

  # Go to workspace.
  cd ~/journey_ws

  # Fetch core packages so we can build them. We will use wstool for this.
  # Desktop Install (recommended): ROS, rqt, rviz, and robot-generic libraries
  rosinstall_generator desktop --rosdistro kinetic --deps --wet-only --tar > kinetic-desktop-wet.rosinstall

  # This will add all of the catkin or wet packages in the given variant and
  # then fetch the sources into the ~/ros_catkin_ws/src directory. The command
  # will take a few minutes to download all of the core ROS packages into the
  # src folder. The -j8 option downloads 8 packages in parallel.
  wstool init -j8 src kinetic-desktop-wet.rosinstall

  # Before you can build your catkin workspace you need to make sure that you
  # have all the required dependencies. We use the rosdep tool for this:
  rosdep install --from-paths src --ignore-src --rosdistro kinetic -y
fi
exit 0
