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
#include <random>

// TODO armand find some way to change the intensity of the light

class RandomizerTools {
 public:
  RandomizerTools(const gazebo::transport::NodePtr &node) {
    // Publish to a Gazebo topic
    pub_ = node->Advertise<gazebo::msgs::Light>("~/light/modify");

    light_.set_name("sun");

    std::cout << "Randomizer tool initialized." << std::endl;
  }

  void OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
    std::cout << "OnRandomize received." << std::endl;

    // Wait for a subscriber to connect
    pub_->WaitForConnection();

    ignition::math::Pose3d pose_sun = GetRandomPose();
    gazebo::msgs::Set(light_.mutable_pose(), pose_sun);
    std::cout << pose_sun << std::endl; // for debugging

    pub_->Publish(light_);
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
  gazebo::transport::PublisherPtr pub_;
  gazebo::msgs::Light light_;
};

int main(int argc, char **argv) {
  // Setup ROS>
  ros::init(argc, argv, "randomizer_tools");
  ros::NodeHandle ros_node;

  // Setup gazebo.
  gazebo::client::setup(argc, argv);
  gazebo::transport::NodePtr gazebo_node(new gazebo::transport::Node());
  gazebo_node->Init();
  gazebo::transport::run();

  // Create our randomizer tools.
  RandomizerTools randomizer_tools(gazebo_node);

  // Subscribe to /journey/randomize.
  ros::Subscriber sub =
      ros_node.subscribe("/journey/randomize", 1000,
                         &RandomizerTools::OnRandomize, &randomizer_tools);

  ros::spin();
  return 0;
}
