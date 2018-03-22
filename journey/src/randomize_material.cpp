#include "journey/randomize_material.h"

namespace {
static const std::string kMaterialUri =
    "file://media/materials/scripts/gazebo.material";

gazebo::common::Color GetRandomColor() {
  std::random_device rd_;
  std::mt19937 gen(rd_());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  return gazebo::common::Color(distribution(gen), distribution(gen),
                               distribution(gen), 1.0);
}

std::string GetRandomMaterial() { return "Gazebo/Wood"; }
}  // namespace

void RandomizeMaterial::Load(gazebo::physics::WorldPtr world,
                             sdf::ElementPtr sdf) {
  std::cout << "Loading material randomizer." << std::endl;
  world_ = world;

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

  for (auto model : world_->GetModels()) {
    if (model->GetName().find("sean") != 0 &&
        model->GetName().find("floor") != 0 &&
        model->GetName().find("ceiling") != 0 &&
        model->GetName().find("counter") != 0) {
      continue;
    }

    std::cout << "Model: " << model->GetName() << std::endl;
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
          std::cout << "Visual: " << visualName << std::endl;
          gazebo::msgs::Visual visMsg;
          visMsg = link->GetVisualMessage(visualName);
          if ((!visMsg.has_material()) || visMsg.mutable_material() == NULL) {
            gazebo::msgs::Material* materialMsg = new gazebo::msgs::Material;
            visMsg.set_allocated_material(materialMsg);
          }

          gazebo::common::Color newColor = GetRandomColor();
          gazebo::msgs::Color* colorMsg =
              new gazebo::msgs::Color(gazebo::msgs::Convert(newColor));
          gazebo::msgs::Color* diffuseMsg = new gazebo::msgs::Color(*colorMsg);

          gazebo::msgs::Material* materialMsg = visMsg.mutable_material();
          if (materialMsg->has_ambient()) {
            materialMsg->clear_ambient();
          }
          materialMsg->set_allocated_ambient(colorMsg);

          if (materialMsg->has_diffuse()) {
            materialMsg->clear_diffuse();
          }
          materialMsg->set_allocated_diffuse(diffuseMsg);

          if (!materialMsg->has_script() ||
              materialMsg->mutable_script() == NULL) {
            gazebo::msgs::Material_Script* scriptMsg =
                new gazebo::msgs::Material_Script;
            materialMsg->set_allocated_script(scriptMsg);
          }
          gazebo::msgs::Material_Script* scriptMsg =
              materialMsg->mutable_script();

          scriptMsg->clear_uri();
          scriptMsg->add_uri(kMaterialUri);

          if (scriptMsg->has_name()) {
            scriptMsg->clear_name();
          }
          scriptMsg->set_name(GetRandomMaterial());

          visMsg.set_name(link->GetScopedName());
          visMsg.set_parent_name(model->GetScopedName());
          visPub_->Publish(visMsg);
        }
      }
    }
  }
}

GZ_REGISTER_WORLD_PLUGIN(RandomizeMaterial)
