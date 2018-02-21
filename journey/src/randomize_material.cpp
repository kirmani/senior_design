#include "ros/ros.h"

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/common/common.hh>
#include <gazebo/math/gzmath.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Pose3.hh>

#include <random>
#include <stdio.h>

using namespace gazebo;

// TODO armand find some way to change the intensity of the light

class RandomizeMaterial : public VisualPlugin {
  public:
  RandomizeMaterial() {
    printf("======Randomizer tool initialized.=======\n\n");
  }

  public:
  void Load(rendering::VisualPtr _visual, sdf::ElementPtr _sdf) {
    this->visual = _visual;

    // Create the node
    this->node = transport::NodePtr(new transport::Node());
    this->node->Init("default");


    // Create a topic name
    std::string topicName = "~/modifymaterial";

    // Subscribe to the topic, and register a callback
    this->sub = this->node->Subscribe(topicName,
       &RandomizeMaterial::OnMsg, this);
  }

  private: void OnMsg(ConstVector3dPtr &_msg)
  {
    this->colorA.Set(0.3,1,1,1);
    this->visual->SetDiffuse(colorA);
    this->visual->SetAmbient(colorA);
    printf("Recieved material message\n\n");
  }

  private:
  rendering::VisualPtr visual;
  common::Color colorA;
  private: transport::NodePtr node;
  private: transport::SubscriberPtr sub;
};

GZ_REGISTER_VISUAL_PLUGIN(RandomizeMaterial)

