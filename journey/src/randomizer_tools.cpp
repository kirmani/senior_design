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
    node_->Init(

    // Start transport
    gazebo::transport::run();
  }

  void OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
    std::cout << "OnRandomize received." << std::endl;

    // Publish to a Gazebo topic
    gazebo::transport::PublisherPtr pub = node_->Advertise<gazebo::msgs::Light>("~/light/modify");

    // Wait for a subscriber to connect
    pub->WaitForConnection();

    // Throttle Publication (make sure gazebo and ros are like updating at the same time ish)
    gazebo::common::Time::MSleep(100); //TODO what does this do?

    msgs::Light msg;
    msg.set_name("sun");

    pose_t pose_sun;
    pose_randomization(&pose_sun);
    msgs::Set(msg.mutable_pose(), <0 5 5 pose_sun.roll pose_sun.pitch pose_sun.yaw>); 
    //might have to use quaterniond and stuff
    //type in gazebo world plugin tutorial to get to

    pub->Publish(msg);
  }

  /*randomly chooses pose (we're just gonna do roll pitch and yaw*/
  //TODO gazbo pose in radians or degrees? in tutorial it's in radians, so let's do radians
  void pose_randomization(pose_t* pose) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0); 
    pose->roll = distribution(generator) * 0.1;
    pose->pitch = distribution(generator) * 0.1;
    pose->yaw = distribution(generator) * 0.1;    
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


