#include "journey/randomize_material.h"

void RandomizeMaterial::Load(gazebo::physics::WorldPtr world,
                             sdf::ElementPtr sdf) {
  std::cout << "Loading material randomizer." << std::endl;
  world_ = world;
  models_ = world->GetModels();

  gzNode_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
  gzNode_->Init(world_->GetName());
  visPub_ = gzNode_->Advertise<gazebo::msgs::Visual>("~/visual");

  // add model names to vector
  for (const auto& model : models_) {
    std::string model_name = model->GetName();
    model_names_.push_back(model_name);
    std::cout << model_name << std::endl;
  }

  // Subscribe to the topic, and register a callback
  sub_ = gzNode_->Subscribe("~/modifymaterial", &RandomizeMaterial::Call, this);
}

void RandomizeMaterial::Call(ConstVector3dPtr& msg) {
  std::cout << "Randomizing material." << std::endl;

  gazebo::common::Color newColor(1.0, 1.0, 1.0, 0.0);
  gazebo::msgs::Color* colorMsg =
      new gazebo::msgs::Color(gazebo::msgs::Convert(newColor));
  gazebo::msgs::Color* diffuseMsg = new gazebo::msgs::Color(*colorMsg);

  for (const auto& model : models_) {
    for (auto link : model->GetLinks()) {
      // Get all the visuals
      sdf::ElementPtr linkSDF = link->GetSDF();

      if (!linkSDF) {
        std::cout << "Link had NULL SDF" << std::endl;
        return;
      }
      if (linkSDF->HasElement("visual")) {
        for (sdf::ElementPtr visualSDF = linkSDF->GetElement("visual");
             visualSDF; visualSDF = linkSDF->GetNextElement("visual")) {
          GZ_ASSERT(visualSDF->HasAttribute("name"),
                    "Malformed visual element!");
          std::string visualName = visualSDF->Get<std::string>("name");
          gazebo::msgs::Visual visMsg;
          visMsg = link->GetVisualMessage(visualName);
          if ((!visMsg.has_material()) || visMsg.mutable_material() == NULL) {
            gazebo::msgs::Material* materialMsg = new gazebo::msgs::Material;
            visMsg.set_allocated_material(materialMsg);
          }
          // gazebo::msgs::Material* materialMsg = visMsg.mutable_material();
          // if (materialMsg->has_ambient()) {
          //   materialMsg->clear_ambient();
          // }
          // materialMsg->set_allocated_ambient(colorMsg);
          // if (materialMsg->has_diffuse()) {
          //   materialMsg->clear_diffuse();
          // }
          visMsg.set_name(link->GetScopedName());
          visMsg.set_parent_name(model->GetScopedName());
          // materialMsg->set_allocated_diffuse(diffuseMsg);
          visPub_->Publish(visMsg);
        }
      }
    }
  }
}

GZ_REGISTER_WORLD_PLUGIN(RandomizeMaterial)
