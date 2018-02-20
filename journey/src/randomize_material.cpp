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

class RandomizeMaterial : public ModelPlugin {
  public:
  RandomizeMaterial() {
    std::cout << "Randomizer tool initialized." << std::endl;
  }

  public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) {
    if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
        << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
      return;
    }

    this->parent_ = _parent;
    this->node = transport::NodePtr(new transport::Node());
    this->node->Init(_parent->GetWorld()->GetName());
    this->visPub = this->node->Advertise<msgs::Visual>("~/visual", 10);
  }

  public:
  void SendVisualMessage(void) {
    msgs::Visual visualMsg;
    // Set the visual's name. This should be unique.
    visualMsg.set_name("visual_boi");

    // Set the visual's parent. This visual will be attached to the parent
    printf("Attaching to: %s\n\n", this->parent_->GetScopedName().c_str());
    visualMsg.set_parent_name(this->parent_->GetScopedName());

    printf("Link %s %d children\n", this->parent_->GetLinks().at(0)->GetName().c_str(), this->parent_->GetLinks().at(0)->GetChildCount());

    // Set the material to be bright red
    visualMsg.mutable_material()->mutable_script()->set_name("Gazebo/WoodFloor");

    visPub->Publish(visualMsg);

    printf("Sent Material Message\n\n");
  }

  private:
  transport::NodePtr node;
  transport::PublisherPtr visPub;
  physics::ModelPtr parent_;
};

GZ_REGISTER_MODEL_PLUGIN(RandomizeMaterial)

