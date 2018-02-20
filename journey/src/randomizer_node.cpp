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

#include <stdio.h>
#include <random>

// TODO armand find some way to change the intensity of the light

class RandomizerTools {
  public:
  RandomizerTools(const gazebo::transport::NodePtr &node) {
    // Publish to a Gazebo topic
    pub_ = node->Advertise<gazebo::msgs::Status>("~/log/status");
    printf("Randomizer tool initialized.\n");
  }

  void OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
    printf("OnRandomize received.\n");
  }

  private:
  gazebo::transport::PublisherPtr pub_;
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
