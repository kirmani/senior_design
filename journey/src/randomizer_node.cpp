#include "journey/randomizer_node.h"

#include <gazebo/gazebo_client.hh>
#include "ros/ros.h"

RandomizerTools::RandomizerTools(const gazebo::transport::NodePtr &node) {
  // Publish to a Gazebo topic
  lightPub_ = node->Advertise<gazebo::msgs::Light>("~/light/modify");
  materialPub_ = node->Advertise<gazebo::msgs::Vector3d>("~/modifymaterial");
  light_.set_name("sun");
}

void RandomizerTools::OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
  // Wait for a subscriber to connect
  lightPub_->WaitForConnection();

  ignition::math::Pose3d pose_sun = GetRandomPose();
  gazebo::msgs::Set(light_.mutable_pose(), pose_sun);

  lightPub_->Publish(light_);

  // publish to update materials
  materialPub_->WaitForConnection();
  gazebo::msgs::Set(&material_, ignition::math::Vector3d(2, 0, 0));
  materialPub_->Publish(material_);
}

/*randomly chooses pose (we're just gonna do roll pitch and yaw*/
// pose in meters and radians
ignition::math::Pose3d RandomizerTools::GetRandomPose() {
  std::random_device rd_;
  std::mt19937 gen(rd_());
  std::uniform_real_distribution<float> distribution(0.0, 0.5);
  return ignition::math::Pose3d(0, 0, 10, distribution(gen), distribution(gen),
                                distribution(gen));
}

int main(int argc, char **argv) {
  // Setup ROS
  ros::init(argc, argv, "randomizer_tools");
  ros::NodeHandle ros_node;

  // Setup gazebo with light and material transport nodes
  gazebo::client::setup(argc, argv);
  gazebo::transport::NodePtr gazebo_node(new gazebo::transport::Node());
  gazebo_node->Init();
  gazebo::transport::run();

  // Create our randomizer tools
  RandomizerTools randomizer_tools(gazebo_node);

  // Subscribe to /journey/randomize
  ros::Subscriber sub =
      ros_node.subscribe("/journey/randomize", 1000,
                         &RandomizerTools::OnRandomize, &randomizer_tools);

  ros::spin();
  return 0;
}
