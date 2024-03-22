#include "flightlib/objects/unity_object.hpp"

namespace flightlib {

UnityObject::UnityObject(std::string id, std::string prefab_id)
  : id_(id), prefab_id_(prefab_id), sign_(1.0) {
  state_.setZero();
}


bool UnityObject::isStatic(void) {
  if (traj_.size() > 1) {
    return false;
  }
  return true;
}

void UnityObject::run(const Scalar dt) {
  if (traj_.size() <= 1) return;

  int idx = int(state_.t / dt);

  if (idx <= 0) {
    idx = 0;
    sign_ = 1.0;
  } else if (idx >= int(traj_.size() - 1)) {
    idx = int(traj_.size() - 1);
    sign_ = -1.0;
  }

  state_.t += sign_ * dt;
  state_.x = traj_[idx].x;
}

bool UnityObject::loadTrajectory(const std::string traj_csv) {
  std::ifstream infile(traj_csv);
  // iterate through all rows
  bool skip_header = true;

  // DEBUG
  bool prev_flag = true;
  RigidState prev_state;

  for (auto& row : CSVRange(infile)) {
    if (skip_header) {
      skip_header = false;
      continue;
    }
    // Read column 0 for time
    RigidState state_i;
    state_i.setZero();
    state_i.t = std::stod((std::string)row[0]);

    //
    state_i.x[RS::POSX] = std::stod((std::string)row[1]);
    state_i.x[RS::POSY] = std::stod((std::string)row[2]);
    state_i.x[RS::POSZ] = std::stod((std::string)row[3]);

    //
    state_i.x[RS::ATTW] = std::stod((std::string)row[3]);
    state_i.x[RS::ATTX] = std::stod((std::string)row[4]);
    state_i.x[RS::ATTY] = std::stod((std::string)row[5]);
    state_i.x[RS::ATTZ] = std::stod((std::string)row[6]);


    // ENGEL HIZ BILGISI DEBUG Calculate speed
    if (!prev_flag) {
      double dt = state_i.t - prev_state.t;
      double dx = state_i.x[RS::POSX] - prev_state.x[RS::POSX];
      double dy = state_i.x[RS::POSY] - prev_state.x[RS::POSY];
      double dz = state_i.x[RS::POSZ] - prev_state.x[RS::POSZ];
      state_i.x[RS::VELX] = dx / dt;
      state_i.x[RS::VELY] = dy / dt;
      state_i.x[RS::VELZ] = dz / dt;
      
    }
    traj_.push_back(state_i);

    // DEBUG
    prev_state = state_i;
    prev_flag = false;
  }

  // static object
  if (traj_.size() == 1) {
    state_ = traj_[0];
  }

  return true;
}


}  // namespace flightlib
