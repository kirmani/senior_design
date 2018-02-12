#include "ros/ros.h"
#include "std_msgs/Empty.h"

class RandomizerTools {
 public:
  RandomizerTools(ros::NodeHandle node) {
    std::cout << "Randomizer tool initialized." << std::endl;
  }

  void OnRandomize(const std_msgs::Empty::ConstPtr &msg) {
    std::cout << "OnRandomize received." << std::endl;
  }
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
