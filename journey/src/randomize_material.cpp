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
#include <string.h>

using namespace gazebo;

// TODO armand find some way to change the intensity of the light

class RandomizeMaterial : public WorldPlugin {
  public:
  RandomizeMaterial() {}

  public:
  void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) {
    this->world = _world;
    this->models = world->GetModels();

    // add model names to vector
    for(int i = 0; i < models.size(); ++i) {
      model_names.push_back(models.at(i)->GetName());
      printf("%s\n", model_names.at(i).c_str());
    }

    // Create the nodes
    this->recieve_node = transport::NodePtr(new transport::Node());
    this->recieve_node->Init("default");

    this->send_node = transport::NodePtr(new transport::Node());
    this->send_node->Init("default");

    // Create a topic name
    std::string topicName = "~/modifymaterial";

    // Subscribe to the topic, and register a callback
    this->sub = this->recieve_node->Subscribe(topicName,
       &RandomizeMaterial::OnMsg, this);

    // Publish to the visual topic
    pub = send_node->Advertise<msgs::Visual>("~/visual");
  }

  private:
  void OnMsg(ConstVector3dPtr &_msg)
  {
    physics::ModelPtr mptr;
    sdf::ElementPtr sdf;
    msgs::Visual visualMsg;
    common::Color colorA;
    colorA.Set(1,0,0.3,1);
    for(int i = 0; i < model_names.size(); ++i) {
      sdf = models.at(i)->GetSDF();
      if (sdf->HasElement("link")) {
         physics::LinkPtr lptr = models.at(i)->GetChildLink("link");
         rendering::VisualPtr vptr = (rendering::VisualPtr) lptr->GetByName("visual");
//        sdf = sdf->GetElement("link");
//        if (sdf->HasElement("visual")){
//          sdf = sdf->GetElement("visual");
//          visualMsg = msgs::VisualFromSDF(sdf);
//          visualMsg.set_parent_name(model_names.at(i).c_str());
//          msgs::Set(visualMsg.mutable_material()->mutable_ambient(), colorA);
//          if(visualMsg.mutable_material()->mutable_script()->has_name()){
//            printf("old script name: %s\n", visualMsg.mutable_material()->mutable_script()->name().c_str());
//          }
//          visualMsg.mutable_material()->mutable_script()->set_name("Gazebo/Bricks");
//          msgs::VisualToSDF(visualMsg, sdf);
//          //printf("modified supposedly\n");
//        }
      }




//      // Set the visual's name. This should be unique.
//      visualMsg.set_name("visual");
//      visualMsg.set_parent_name(model_names.at(i));
//      //visualMsg.mutable_material()->mutable_script()->set_name("Gazebo/Bricks");
//      msgs::Set(visualMsg.mutable_material()->mutable_ambient(), colorA);
//      // Create a cylinder
////      msgs::Geometry *geomMsg = visualMsg.mutable_geometry();
////      geomMsg->set_type(msgs::Geometry::CYLINDER);
////      geomMsg->mutable_cylinder()->set_radius(1);
////      geomMsg->mutable_cylinder()->set_length(.1);
//      printf("modified %s supposedly\n", model_names.at(i).c_str());
      pub->WaitForConnection();
      pub->Publish(visualMsg);
    }
  }

  private:
  physics::WorldPtr world;
  transport::NodePtr recieve_node;
  transport::NodePtr send_node;
  transport::SubscriberPtr sub;
  transport::PublisherPtr pub;
  physics::Model_V models;
  std::vector<std::string> model_names;
};

GZ_REGISTER_WORLD_PLUGIN(RandomizeMaterial)

