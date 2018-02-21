#include "ros/ros.h"
#include "std_msgs/Empty.h"

#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/math/gzmath.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Pose3.hh>

#include <iostream>
#include <stdio.h>
#include <random>

using namespace gazebo;

class RandomizerTools {
 public:
  RandomizerTools(const transport::NodePtr &node) {
    // Publish to a Gazebo topic
    //lightPub_ = node->Advertise<msgs::Light>("~/light/modify");
    materialPub_ = node->Advertise<msgs::Vector3d>("~/modifymaterial");

    //light_.set_name("sun");

  }

  void OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
    printf("OnRandomize received\n");

//    // Wait for a subscriber to connect
//    lightPub_->WaitForConnection();
//
//    ignition::math::Pose3d pose_sun = GetRandomPose();
//    msgs::Set(light_.mutable_pose(), pose_sun);
//    std::cout << pose_sun << std::endl; // for debugging
//
//    lightPub_->Publish(light_);

    //publish to update materials
    materialPub_->WaitForConnection();
    msgs::Set(&material_, ignition::math::Vector3d(2, 0, 0));
    materialPub_->Publish(material_);
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
  transport::PublisherPtr lightPub_;
  transport::PublisherPtr materialPub_;
  msgs::Vector3d material_;
  msgs::Light light_;
};

int main(int argc, char **argv) {
  // Setup ROS
  ros::init(argc, argv, "randomizer_tools");
  ros::NodeHandle ros_node;

  // Setup gazebo.
  client::setup(argc, argv);
  transport::NodePtr gazebo_node(new transport::Node());
  gazebo_node->Init();
  transport::run();

  // Create our randomizer tools.
  RandomizerTools randomizer_tools(gazebo_node);

  // Subscribe to /journey/randomize.
  ros::Subscriber sub =
      ros_node.subscribe("/journey/randomize", 1000,
                         &RandomizerTools::OnRandomize, &randomizer_tools);

  ros::spin();
  return 0;
}