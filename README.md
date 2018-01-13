# Autonomous Navigation and Obstacle Avoidance for Drones

### See [Wiki](https://github.com/kirmani/ee364d/wiki) for documentation.

## Team

- Armand Behroozi <<armandb@utexas.edu>>
- Ali de Jong <<ali.dejong@utexas.edu>>
- Sean Kirmani <<sean@kirmani.io>>
- Yuriy Minin <<me@yuriy.io>>
- Josh Minor <<joshminor@utexas.edu>>
- Taylor Zhao <<taylorzhao@utexas.edu>>

## Requirements

For stability and compatibility with open-source ROS packages, our system is
optimized to work for [ROS Kinetic](http://wiki.ros.org/kinetic) on
[Ubuntu 16.04 LTS (Xenial Xerus)](http://releases.ubuntu.com/16.04/).

It may be possible to build on other versions of ROS for different
distributions, but do so at your own risk.

## Installation

On Ubuntu 16.04:

```
git clone https://github.com/kirmani/ee364d
cd ee364d
bash setup.sh
```

This setup script will install ROS Indigo on your machine, upgrade to Gazebo 7
install the dependencies for the ROS Journey package, and build your
environment.

## Usage

### Train Collision Avoidance Model

To run the simulation, open a terminal, and run the following commands:

```
catkin
roslaunch journey simulation_randomization.launch
```

To start training, open terminal, and run the following:

```
catkin
roslaunch journey train_obstacle_avoidance.py
```

### Sending navigation goals in simulation.

To run the simulation, open a terminal, and run the following commands:

```
catkin
roslaunch journey journey_simulator.launch
```

To set a navigation goal with some delta x, delta y, at a fixed altitude z,
just open a new terminal window and run the follow:

```
catkin
ardrone_goal x y z
```

where x, y, and z are in meters.

### Sending navigation goals on a real drone.

To run the simulation, open a terminal, and run the following commands:

```
catkin
roslaunch journey journey.launch
```

To set a navigation goal with some delta x, delta y, at a fixed altitude z,
just open a new terminal window and run the follow:

```
catkin
ardrone_goal x y z
```

where x, y, and z are in meters.
