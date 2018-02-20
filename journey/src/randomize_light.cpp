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

class RandomizeLight : public ModelPlugin {
  public:
  RandomizeLight() {
    std::cout << "Randomize Light tool initialized." << std::endl;
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
    this->lightPub = this->node->Advertise<msgs::Light>("~/light/modify", 10);
  }
  
  public:
  void SendLightMessage(void) {
    msgs::Light lightMsg;
    // Set the visual's name. This should be unique.
    lightMsg.set_name("sun");

    lightPub->WaitForConnection();

    ignition::math::Pose3d pose_sun = GetRandomPose();
    gazebo::msgs::Set(lightMsg.mutable_pose(), pose_sun);

    lightPub->Publish(lightMsg);

    printf("Sent Light Message\n\n");
  }

  /*randomly chooses pose (we're just gonna do roll pitch and yaw*/
  // pose in meters and radians
  static ignition::math::Pose3d GetRandomPose() {
    std::random_device rd_;
    std::mt19937 gen(rd_());
    std::uniform_real_distribution<float> distribution(0.0, 0.5);
    return ignition::math::Pose3d(0, 0, 10, distribution(gen),
                                  distribution(gen),
                                  distribution(gen));
  }

 private:
  transport::NodePtr node;
  transport::PublisherPtr lightPub;
  physics::ModelPtr parent_;
};

GZ_REGISTER_MODEL_PLUGIN(RandomizeLight)