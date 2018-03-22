/*
 * randomize_material.h
 * Copyright (C) 2018 Sean Kirmani <sean@kirmani.io>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef RANDOMIZE_MATERIAL_H
#define RANDOMIZE_MATERIAL_H

#include "ros/ros.h"

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/math/gzmath.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Pose3.hh>

// #include <stdio.h>
// #include <string.h>
// #include <random>

class RandomizeMaterial : public gazebo::WorldPlugin {
 public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf);
  void Call(ConstVector3dPtr &msg);

 private:
  gazebo::transport::NodePtr gzNode_;
  gazebo::transport::PublisherPtr visPub_;
  gazebo::transport::SubscriberPtr sub_;
  gazebo::physics::WorldPtr world_;
  gazebo::physics::Model_V models_;
  std::vector<std::string> model_names_;
};
#endif /* !RANDOMIZE_MATERIAL_H */
