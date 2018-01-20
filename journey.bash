### Journey drone helper script.

if [ -f /opt/ros/indigo/setup.bash ]; then
  source /opt/ros/indigo/setup.bash
fi

# Source the default catkin workspace if one exists.
if [ -f ~/journey_ws/devel/setup.bash ]; then
  source ~/journey_ws/devel/setup.bash
fi

# Ubuntu Gazebo fix on for VMWare.
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export SVGA_VGPU10=0
fi

# The functions below make use of this variable.
# Make sure it points to wherever you created your workspace!
export JOURNEY_WS=${HOME}/journey_ws

# Quickly go to the Catkin workspace and configure the build environment.
function journey() {
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    source /opt/ros/kinetic/setup.bash
    source ${JOURNEY_WS}/devel/setup.bash
    cd ${JOURNEY_WS}
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "OS X catkin() not yet implemented."
  fi
}

function ardrone_connect() {
  journey
  roslaunch journey journey.launch
}

function ardrone_takeoff() {
  rostopic pub --once /ardrone/takeoff std_msgs/Empty "{}"
}

function ardrone_land() {
  rostopic pub --once /ardrone/land std_msgs/Empty "{}"
}

function ardrone_reset() {
  rostopic pub --once /ardrone/reset std_msgs/Empty "{}"
}

function ardrone_goal() {
  rostopic pub --once /journey/set_nav_goal geometry_msgs/Point "{x: $1, y: $2, z: $3}"
}
