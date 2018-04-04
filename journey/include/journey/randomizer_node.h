/*
 * randomizer_node.h
 * Copyright (C) 2018 Sean Kirmani <sean@kirmani.io>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef RANDOMIZER_NODE_H
#define RANDOMIZER_NODE_H

#include <gazebo/gazebo.hh>
#include <ignition/math/Pose3.hh>
#include "journey/randomize_material.h"
#include "std_msgs/Empty.h"

class RandomizerTools {
 public:
  RandomizerTools(const gazebo::transport::NodePtr &node);
  void OnRandomize(const std_msgs::Empty::ConstPtr &msg);
  ignition::math::Pose3d GetRandomPose();

 private:
  gazebo::transport::PublisherPtr lightPub_;
  gazebo::transport::PublisherPtr materialPub_;
  gazebo::msgs::Vector3d material_;
  gazebo::msgs::Light light_;
};

#endif /* !RANDOMIZER_NODE_H */
