#include "ros/ros.h"
#include "std_msgs/Empty.h"

#include <random>

#include <gazebo/gazebo.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <math/gzmath.hh>

#include <iostream>

#include <transport/transport.hh>
#include <rendering/rendering.hh>


//TODO armand find some way to change the intensity of the light
class RandomizerTools {
 public:

  struct pose_t {
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;
} ;

  RandomizerTools(ros::NodeHandle node) {
    std::cout << "Randomizer tool initialized." << std::endl;
    
    /*initialization code*/
    // Create our node for communication
    node_ = new gazebo::transport::Node();
    node_->Init();

    // Start transport
    gazebo::transport::run();
  }

  void OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
    std::cout << "OnRandomize received." << std::endl;
    //when called, this is what needs to be done

    // Publish to a Gazebo topic
    gazebo::transport::PublisherPtr pub =
    node_->Advertise<gazebo::msgs::Light>("~/light/modify");

    // Wait for a subscriber to connect
    pub->WaitForConnection();

    // Throttle Publication
    gazebo::common::Time::MSleep(100);

    msgs::Light msg;
    msg.set_name("sun2");

    msgs::Set(msg.mutable_pose(), <0 5 5 0 0 0>);

    pub->Publish(msg);
  }

  /*randomly chooses pose (we're just gonna do roll pitch and yaw*/
  void pose_randomization(pose_t* pose) {
    std::random_device rd;
    std::mt19937 gen(rd())
    std:uniform_real_distribution<0>

    
}    

 private:
  gazebo::transport::NodePtr node_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "randomizer_tools");
  ros::NodeHandle node;

  RandomizerTools randomizer_tools(node);

  ros::Subscriber sub =
      node.subscribe("/journey/randomize", 1000, &RandomizerTools::OnRandomize,
                     &randomizer_tools);

  ros::spin();
  return 0;
}


