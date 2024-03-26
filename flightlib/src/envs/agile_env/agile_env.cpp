
#include "flightlib/envs/agile_env/agile_env.hpp"

namespace flightlib {

AgileEnv::AgileEnv()
  : AgileEnv(getenv("FLIGHTMARE_PATH") +
                std::string("/flightpy/configs/vision/config.yaml"),
              0) {}

AgileEnv::AgileEnv(const std::string &cfg_path, const int env_id)
  : EnvBase() {
  // check if configuration file exist
  if (!(file_exists(cfg_path))) {
    logger_.error("Configuration file %s does not exists.", cfg_path);
  }
  // load configuration file
  cfg_ = YAML::LoadFile(cfg_path);
  //
  init();
  env_id_ = env_id;
}

AgileEnv::AgileEnv(const YAML::Node &cfg_node, const int env_id) : EnvBase() {
  cfg_ = cfg_node;
  //
  init();
  env_id_ = env_id;
}


void AgileEnv::init() {
  //
  
  is_collision_ = false;
  unity_render_offset_ << 0.0, 0.0, 0.0;
  goal_linear_vel_ << 0.0, 0.0, 0.0;
  linear_vel_mask_ << 1.0, 0.005, 0.005;
  cmd_.setZeros();

  // create quadrotors
  quad_ptr_ = std::make_shared<Quadrotor>();

  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr_->updateDynamics(dynamics);

  // obs_dim_ = agileenv::kNObs;
  act_dim_ = agileenv::kNAct;
  rew_dim_ = 0;
  num_detected_obstacles_ = agileenv::kNObstacles;

  // load parameters
  loadParam(cfg_);

  // define input and output dimension for the environment
  if (STUDENT_FLAG_){
    obs_dim_ = 65; //agileenv::kNObs - (agileenv::kNObstacles * agileenv::kNObstaclesVel);
  }
  else{
    obs_dim_ = agileenv::kNObs;
  }
  
  // add camera
  if (!configCamera(cfg_)) {
    logger_.error(
      "Cannot config RGB Camera. Something wrong with the config file");
  }

  obstacle_cfg_path_ = getenv("FLIGHTMARE_PATH") +
                       std::string("/flightpy/configs/vision/") +
                       difficulty_level_ + std::string("/") + env_folder_;

  // add dynamic objects
  std::string dynamic_object_yaml =
    obstacle_cfg_path_ + std::string("/dynamic_obstacles.yaml");
  if (!configDynamicObjects(dynamic_object_yaml)) {
    logger_.error(
      "Cannot config Dynamic Object Yaml. Something wrong with the config "
      "file");
  }
  
  // add static objects
  static_object_csv_ =
    obstacle_cfg_path_ + std::string("/static_obstacles.csv");
  if (!configStaticObjects(static_object_csv_)) {
    logger_.error(
      "Cannot config Static Object. Something wrong with the config file");
  }
  
  // use single rotor control or bodyrate control
  Scalar max_force = quad_ptr_->getDynamics().getForceMax();
  Vector<3> max_omega = quad_ptr_->getDynamics().getOmegaMax();
  //
  act_mean_ << (max_force / quad_ptr_->getMass()) / 2, 0.0, 0.0, 0.0;
  act_std_ << (max_force / quad_ptr_->getMass()) / 2, max_omega.x(),
    max_omega.y(), max_omega.z();
}

AgileEnv::~AgileEnv() {}

bool AgileEnv::reset(Ref<Vector<>> obs) {
  quad_state_.setZero();
  pi_act_.setZero();
  old_pi_act_.setZero();
  is_collision_ = false;

  // randomly reset the quadrotor state
  // reset position
  // DEFAULT
  // quad_state_.x(QS::POSX) = norm_dist_(random_gen_) * 20 + 1;
  quad_state_.x(QS::POSX) = uniform_dist_(random_gen_) + 10;
  quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) * 9.0;         //3.0;
  quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) * 4 + 5.0;     //3.0;
  
  // quad_state_.x(QS::POSX) = 10.0;
  // quad_state_.x(QS::POSY) = 3.0;
  // quad_state_.x(QS::POSZ) = 3.0;

  // reset quadrotor with random states
  quad_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  // use collective thrust and bodyrate control mode
  cmd_.setCmdMode(quadcmd::THRUSTRATE);
  cmd_.collective_thrust = 0;
  cmd_.omega.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool AgileEnv::reset(Ref<Vector<>> obs, bool random) { return reset(obs); }

bool AgileEnv::getObs(Ref<Vector<>> obs) {
  if (obs.size() != obs_dim_) {
    logger_.error("Observation dimension mismatch. %d != %d", obs.size(),
                  obs_dim_);
    return false;
  }
  // compute rotation matrix
  Vector<9> ori = Map<Vector<>>(quad_state_.R().data(), quad_state_.R().size());

  // get N most closest obstacles as the observation
  Vector<agileenv::kNObstacles * agileenv::kNObstaclesState> obstacle_obs;
  Vector<agileenv::kNObstacles * agileenv::kNObstaclesVel> obstacle_vel_obs;
  
  
  getObstacleState(obstacle_obs, obstacle_vel_obs);

  /*
  goal_linear_vel_ : 3
  ori              : 9
  quad_state_.p    : 3
  quad_state_.v    : 3
  quad_state_.w    : 3
  obstacle_obs     : 70 (10 * 7) - 10 obstacles, each with 7 states ((x,y,z) + size(r) + (Vx,Vy,Vz)), (x: ileri/geri, y: sag/sol, z: yukseklik)
  world_box sinir  : 4
  TOPLAM           : 91 + 4 
  */
  if (STUDENT_FLAG_){
    obs << goal_linear_vel_, ori, quad_state_.p, quad_state_.v, quad_state_.w, obstacle_obs,
                                                                      world_box_[2] - quad_state_.x(QS::POSY), 
                                                                      world_box_[3] - quad_state_.x(QS::POSY),
                                                                      world_box_[4] - quad_state_.x(QS::POSZ),
                                                                      world_box_[5] - quad_state_.x(QS::POSZ);

  }else{
    obs << goal_linear_vel_, ori, quad_state_.p, quad_state_.v, quad_state_.w, obstacle_obs, 
                                                                      world_box_[2] - quad_state_.x(QS::POSY), 
                                                                      world_box_[3] - quad_state_.x(QS::POSY),
                                                                      world_box_[4] - quad_state_.x(QS::POSZ),
                                                                      world_box_[5] - quad_state_.x(QS::POSZ),
                                                                      obstacle_vel_obs;

  }
  return true;
}

bool AgileEnv::getObstacleState(Ref<Vector<>> obs_state, Ref<Vector<>> obstacle_vel_obs) {
  
  if (dynamic_objects_.size() <= 0 || static_objects_.size() <= 0) {
    logger_.error("No dynamic or static obstacles.");
    return false;
  }

  // make sure to reset the collision penalty
  relative_pos_norm_.clear();
  relative_pos_vector_.clear();
  obstacle_radius_.clear();

  //
  quad_ptr_->getState(&quad_state_);

  // compute relative distance to dynamic obstacles
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> relative_pos;
  // Obs velocity information / allocator for vectorized environment
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> obstacle_vel_list;
  
  //  ----------------------------------------------- Dynamic -----------------------------------------------
  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    
    // Velocity calculation
    Vector<3> obs_velocity = dynamic_objects_[i]->getSpeed();
    float magnitude = std::sqrt(obs_velocity[0] * obs_velocity[0] + obs_velocity[1] * obs_velocity[1] + obs_velocity[2] * obs_velocity[2]);
    if (magnitude == 0){
      Vector<3>normalized_v{0.0, 0.0, 0.0};
      obstacle_vel_list.push_back(normalized_v);     
    }
    else{
      Vector<3> normalized_v;
      normalized_v[0] = std::max(0.0,std::min(100.0,(obs_velocity[0] / magnitude)));
      normalized_v[1] = std::max(0.0,std::min(100.0,(obs_velocity[1] / magnitude)));
      normalized_v[2] = std::max(0.0,std::min(100.0,(obs_velocity[2] / magnitude)));
      obstacle_vel_list.push_back(normalized_v);
    }
      
    // compute relative position vector
    Vector<3> delta_pos = dynamic_objects_[i]->getPos() - quad_state_.p;
    relative_pos.push_back(delta_pos);

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();
    if (obstacle_dist > max_detection_range_){
      obstacle_dist = max_detection_range_;
    }
    relative_pos_norm_.push_back(obstacle_dist);

    // store the obstacle radius
    Scalar obs_radius = dynamic_objects_[i]->getScale()[0] / 2;
    obstacle_radius_.push_back(obs_radius);

    // Check collision 
    if (obstacle_dist < obs_radius) {
      is_collision_ = true;
    }
  }
  
  //  ----------------------------------------------- Static -----------------------------------------------
  // compute relatiev distance to static obstacles
  for (int i = 0; i < (int)static_objects_.size(); i++) {
    // compute relative position vector
    Vector<3> delta_pos = static_objects_[i]->getPos() - quad_state_.p;

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();
    // Scalar obstacle_dist = (quad_state_.v.cwiseProduct(delta_pos)).norm();

    // Dot Product
    Scalar obstacle_dist_dot = (quad_state_.v.dot(delta_pos));

    // Angle Between Velocity and RelativePosition Vector
    double magnitude_v = sqrt(pow(quad_state_.v[0],2) + pow(quad_state_.v[1],2) + pow(quad_state_.v[2],2));
    double magnitude_p = sqrt(pow(delta_pos[0],2) + pow(delta_pos[1],2) + pow(delta_pos[2],2));
    Scalar magnitudes = magnitude_v * magnitude_p;
    Scalar direction_angle = std::acos(obstacle_dist_dot / magnitudes);
    

    // store the obstacle radius
    Scalar obs_radius = static_objects_[i]->getScale()[0] / 2;

    if (obstacle_dist > max_detection_range_) {
      obstacle_dist = max_detection_range_;
    }
    
    if (obstacle_dist_dot>0 && direction_angle<35.0){
    // if (true){
      relative_pos.push_back(delta_pos);

      relative_pos_vector_.push_back(delta_pos);
      relative_pos_norm_.push_back(obstacle_dist);
      //relative_pos_dot_.push_back(obstacle_dist_dot);

      obstacle_radius_.push_back(obs_radius);

      Vector<3>normalized_v{0.0, 0.0, 0.0};
      obstacle_vel_list.push_back(normalized_v);
    }

    if (obstacle_dist < obs_radius) {
      is_collision_ = true;
    }
  }
  
  //  ----------------------------------------------- Segment -----------------------------------------------
  
  size_t idx = 0;
  for (size_t sort_idx : sort_indexes(relative_pos_norm_)) {
    if (idx >= agileenv::kNObstacles) break;

    if (idx < relative_pos.size()) {
      // if enough obstacles in the environment
      if (relative_pos_norm_[sort_idx] <= max_detection_range_) {
        // if obstacles are within detection range
        obs_state.segment<agileenv::kNObstaclesState>(
          idx * agileenv::kNObstaclesState)
          << relative_pos[sort_idx], obstacle_radius_[sort_idx];
          // << relative_pos[sort_idx], obstacle_radius_[sort_idx], obstacle_vel_list[sort_idx];
        
        // Velocity
        obstacle_vel_obs.segment<agileenv::kNObstaclesVel>(idx * agileenv::kNObstaclesVel) << obstacle_vel_list[sort_idx];

      } else {
        // if obstacles are beyong detection range
        obs_state.segment<agileenv::kNObstaclesState>(
          idx * agileenv::kNObstaclesState) =
           Vector<4>(max_detection_range_, max_detection_range_,
                     max_detection_range_, obstacle_radius_[sort_idx]);
          // To add velocity information, we can also add velocity as another element of obs_state
          // obs_state.segment<agileenv::kNObstaclesState>(
          // idx * agileenv::kNObstaclesState) << max_detection_range_, max_detection_range_,
          //                                      max_detection_range_, obstacle_radius_[sort_idx],
          //                                      0.01, 0.01, 0.01;
        
        // Velocity
        obstacle_vel_obs.segment<agileenv::kNObstaclesVel>(idx * agileenv::kNObstaclesVel) << 0.0, 0.0, 0.0;

      }

    } else {
      // DEFAULT
      // if not enough obstacles in the environment
      obs_state.segment<agileenv::kNObstaclesState>(
        idx * agileenv::kNObstaclesState) =
        Vector<agileenv::kNObstaclesState>(max_detection_range_,
                                            max_detection_range_,
                                            max_detection_range_, 0.0);
        // To add velocity information, we can also add velocity as another element of obs_state
        // obs_state.segment<agileenv::kNObstaclesState>(
        //   idx * agileenv::kNObstaclesState) << max_detection_range_,
        //                                        max_detection_range_,
        //                                        max_detection_range_, 0.0, 0.01, 0.01, 0.01;
      
      // Velocity
      obstacle_vel_obs.segment<agileenv::kNObstaclesVel>(idx * agileenv::kNObstaclesVel) << 0.0, 0.0, 0.0;                               
    }
    idx += 1;
  }

  return true;
}

bool AgileEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs,
                     Ref<Vector<>> reward) {
  if (!act.allFinite() || act.rows() != act_dim_ || rew_dim_ != reward.rows()) {
    return false;
    logger_.error(
      "Cannot run environment simulation. dimension mismatch or invalid "
      "actions.");
  }

  //
  old_pi_act_ = pi_act_;

  // @TODO: BUNU KONTROL ET
  // compute actual control actions
  // act has range between [-1, 1] due to Tanh layer of the NN policy
  pi_act_ = act.cwiseProduct(act_std_) + act_mean_;

  cmd_.t += sim_dt_;
  quad_state_.t += sim_dt_;

  // apply old actions to simulate delay
  cmd_.collective_thrust = old_pi_act_(0);
  cmd_.omega = old_pi_act_.segment<3>(1);

  // simulate quadrotor
  quad_ptr_->run(cmd_, sim_dt_);

  // update quadrotor state and old quad_state
  quad_old_state_ = quad_state_;
  quad_ptr_->getState(&quad_state_);

  // simulate dynamic obstacles
  simDynamicObstacles(sim_dt_);

  // update observations
  getObs(obs);

  return computeReward(reward);
}

bool AgileEnv::simDynamicObstacles(const Scalar dt) {
  if (dynamic_objects_.size() <= 0) {
    logger_.warn(
      "No Dynamic Obstacles defined. Skipping dynamic obstacles simulation.");
    return false;
  }
  
  for (int i = 0; i < int(dynamic_objects_.size()); i++) {
    dynamic_objects_[i]->run(sim_dt_);
  }


  return true;
}

bool AgileEnv::computeReward(Ref<Vector<>> reward) {
  // ---------------------- reward function design
  // - compute collision penalty
  Scalar collision_penalty = 0.0;
  // Scalar move_reward = 0.0;
  Scalar relative_dist = 0.0;
  size_t idx = 0;
  for (size_t sort_idx : sort_indexes(relative_pos_norm_)) {
    if (idx >= agileenv::kNObstacles) break;

    // EKSI DEGERLERI SIFIRA CEKIYOR ???
    // Scalar relative_dist =
    //   (relative_pos_norm_[sort_idx] > 0) &&
    //       (relative_pos_norm_[sort_idx] < max_detection_range_)
    //     ? relative_pos_norm_[sort_idx]
    //     : max_detection_range_;
    
    if (DEBUG_FLAG_){
    std::cout << idx 
              << " - obstacle : " << relative_pos_vector_[sort_idx][0]
              << " "<< relative_pos_vector_[sort_idx][1]
              << " " << relative_pos_vector_[sort_idx][2]
              << "  -  r: " << obstacle_radius_[sort_idx]
              << " " << relative_pos_norm_[sort_idx]
              << std::endl;
    }
    relative_dist = 0.0;
    
    // 4 DEGISIKLIK
    if (relative_pos_norm_[sort_idx] > 0){
      if (relative_pos_norm_[sort_idx] < max_detection_range_){
        relative_dist = relative_pos_norm_[sort_idx];
      }
      else{
        relative_dist = max_detection_range_;
      }
    }
    else{
      relative_dist = std::max(relative_pos_norm_[sort_idx], 0.0);
      // std::cout  << "RELATIVE SET TO ZERO  :  " << relative_dist<<std::endl;
    }


    // dist_margin_ = 1.0;
    // 5 DEGISIKLIK
    if (relative_pos_norm_[sort_idx] > 0.0f &&  relative_pos_norm_[sort_idx] <= obstacle_radius_[sort_idx] + dist_margin_) {
      // compute distance penalty
      collision_penalty += collision_coeff_ * std::exp(-1.0 * relative_dist);
      // move_reward = -0.001 * (quad_state_.p(QS::POSX) - quad_old_state_.p(QS::POSX));
      if (DEBUG_FLAG_){
      std::cout << " relative_dist: " << relative_dist << std::endl;
      }
    }
    else{
      // collision_penalty -= collision_coeff_ * std::exp(-1.0 * relative_dist);
      // move_reward = 0.01 * (quad_state_.p(QS::POSX) - quad_old_state_.p(QS::POSX));
    }

    idx += 1;
  }

  // - tracking a constant linear velocity
  // Scalar lin_vel_reward =
  //   vel_coeff_ * (quad_state_.v - goal_linear_vel_).norm();

  // SADECE X EKSENI
  Vector<3> velocity_error = quad_state_.v - goal_linear_vel_;
  if (abs(velocity_error[0])<1.0){
    velocity_error[0] *= 0.5;
  }
  Scalar lin_vel_reward =
    vel_coeff_ * ((velocity_error).cwiseProduct(linear_vel_mask_)).norm();

  // DEFAULT
  // Scalar lin_vel_reward =
  //   vel_coeff_ * ((quad_state_.v - goal_linear_vel_).cwiseProduct(linear_vel_mask_)).norm();

  // - angular velocity penalty, to avoid oscillations
  const Scalar ang_vel_penalty = angular_vel_coeff_ * quad_state_.w.norm();

  // Scalar survive_rew_temp = survive_rew_;
  // if (is_collision_){
  //   survive_rew_temp = 0.0;
  // }
  
  //  change progress reward as survive reward
  const Scalar total_reward =
    lin_vel_reward + collision_penalty + ang_vel_penalty + survive_rew_;// + move_reward;
    // lin_vel_reward + collision_penalty + ang_vel_penalty + survive_rew_;// + distance_penalty;

  if (DEBUG_FLAG_){

    std::cout << "lin: " << lin_vel_reward
              << " coll: " << collision_penalty
              << " ang: " << ang_vel_penalty
              << " total: " << total_reward
              << " x: " << quad_state_.v[0]
              << " y: " << quad_state_.v[1]
              << " z: " << quad_state_.v[2]
              << std::endl; 
  }
  // return all reward components for debug purposes
  // only the total reward is used by the RL algorithm
  reward << lin_vel_reward, collision_penalty, ang_vel_penalty, survive_rew_,
    total_reward;
  return true;
}

bool AgileEnv::isTerminalState(Scalar &reward) {
  // if (is_collision_ && num_env_==1) {
  //   std::cout << " COLLISION " << std::endl;
  //   reward = -1.0;
  //   // return false;
  //   return true;
  // }

  // simulation time out
  if (cmd_.t >= max_t_ - sim_dt_) {
    reward = 0.0;
    return true;
  }

  // world boundling box check
  // - x, y, and z
  const Scalar safty_threshold = 0.1;
  bool x_valid = quad_state_.p(QS::POSX) >= world_box_[0] + safty_threshold &&
                 quad_state_.p(QS::POSX) <= world_box_[1] - safty_threshold;
  bool y_valid = quad_state_.p(QS::POSY) >= world_box_[2] + safty_threshold &&
                 quad_state_.p(QS::POSY) <= world_box_[3] - safty_threshold;
  bool z_valid = quad_state_.x(QS::POSZ) >= world_box_[4] + safty_threshold &&
                 quad_state_.x(QS::POSZ) <= world_box_[5] - safty_threshold;
  if (!x_valid || !y_valid || !z_valid) {
    reward = -1.0;
    return true;
  }
  return false;
}


bool AgileEnv::getQuadAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && pi_act_.allFinite() && (act.size() == pi_act_.size())) {
    act = pi_act_;
    return true;
  }
  return false;
}

bool AgileEnv::getQuadState(Ref<Vector<>> obs) const {
  if (quad_state_.t >= 0.0 && (obs.rows() == agileenv::kNQuadState)) {
    obs << quad_state_.t, quad_state_.p, quad_state_.qx, quad_state_.v,
      quad_state_.w, quad_state_.a, quad_ptr_->getMotorOmega(),
      quad_ptr_->getMotorThrusts();
    return true;
  }
  logger_.error("Get Quadrotor state failed.");
  return false;
}

bool AgileEnv::getDepthImage(Ref<DepthImgVector<>> depth_img) {
  if (!rgb_camera_ || !rgb_camera_->getEnabledLayers()[0]) {
    logger_.error(
      "No RGB Camera or depth map is not enabled. Cannot retrieve depth "
      "images.");
    return false;
  }
  rgb_camera_->getDepthMap(depth_img_);

  depth_img = Map<DepthImgVector<>>((float_t *)depth_img_.data,
                                    depth_img_.rows * depth_img_.cols);
  return true;
}


bool AgileEnv::getImage(Ref<ImgVector<>> img, const bool rgb) {
  if (!rgb_camera_) {
    logger_.error("No Camera! Cannot retrieve Images.");
    return false;
  }

  rgb_camera_->getRGBImage(rgb_img_);

  if (rgb_img_.rows != img_height_ || rgb_img_.cols != img_width_) {
    logger_.error(
      "Image resolution mismatch. Aborting.. Image rows %d != %d, Image cols "
      "%d != %d",
      rgb_img_.rows, img_height_, rgb_img_.cols, img_width_);
    return false;
  }

  if (!rgb) {
    // converting rgb image to gray image
    cvtColor(rgb_img_, gray_img_, CV_RGB2GRAY);
    // map cv::Mat data to Eiegn::Vector
    img = Map<ImgVector<>>(gray_img_.data, gray_img_.rows * gray_img_.cols);
  } else {
    img = Map<ImgVector<>>(rgb_img_.data, rgb_img_.rows * rgb_img_.cols *
                                            rgb_camera_->getChannels());
  }
  return true;
}


bool AgileEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["environment"]) {
    difficulty_level_ = cfg["environment"]["level"].as<std::string>();
    env_folder_ = cfg["environment"]["env_folder"].as<std::string>();
    world_box_ = cfg["environment"]["world_box"].as<std::vector<Scalar>>();
    std::vector<Scalar> goal_vel_vec =
      cfg["environment"]["goal_vel"].as<std::vector<Scalar>>();
    goal_linear_vel_ = Vector<3>(goal_vel_vec.data());
    max_detection_range_ =
      cfg["environment"]["max_detection_range"].as<Scalar>();

    // DEBUG
    std::vector<Scalar> goal_position_vec = cfg["environment"]["goal_position"].as<std::vector<Scalar>>();
    goal_position_ = Vector<3>(goal_position_vec.data());
    DEBUG_FLAG_ = cfg["environment"]["debug_flag"].as<bool>();
    
    STUDENT_FLAG_ = cfg["environment"]["student_flag"].as<bool>();
  }

  if (cfg["simulation"]) {
    sim_dt_ = cfg["simulation"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["simulation"]["max_t"].as<Scalar>();
    num_env_ = cfg["simulation"]["num_envs"].as<Scalar>();
  } else {
    logger_.error("Cannot load [quadrotor_env] parameters");
    return false;
  }

  if (cfg["rewards"]) {
    // load reward coefficients for reinforcement learning
    vel_coeff_ = cfg["rewards"]["vel_coeff"].as<Scalar>();
    collision_coeff_ = cfg["rewards"]["collision_coeff"].as<Scalar>();
    angular_vel_coeff_ = cfg["rewards"]["angular_vel_coeff"].as<Scalar>();
    survive_rew_ = cfg["rewards"]["survive_rew"].as<Scalar>();

    // load reward settings
    reward_names_ = cfg["rewards"]["names"].as<std::vector<std::string>>();
    rew_dim_ = reward_names_.size();

    // DEBUG
    goal_position_coeff_ = cfg["rewards"]["goal_position_coeff_"].as<Scalar>();
    dist_margin_ = cfg["rewards"]["dist_margin"].as<Scalar>();
  } else {
    logger_.error("Cannot load [rewards] parameters");
    return false;
  }

  // environment
  if (cfg["unity"]) {
    unity_render_ = cfg["unity"]["render"].as<bool>();
    scene_id_ = cfg["unity"]["scene_id"].as<SceneID>();
  }

  //
  std::string scene_file =
    getenv("FLIGHTMARE_PATH") + std::string("/flightpy/configs/scene.yaml");
  // check if configuration file exist
  if (!(file_exists(scene_file))) {
    logger_.error("Unity scene configuration file %s does not exists.",
                  scene_file);
  }
  // load configuration file
  YAML::Node scene_cfg_node = YAML::LoadFile(scene_file);
  std::string scene_idx = "scene_" + std::to_string(scene_id_);

  std::vector<Scalar> render_offset =
    scene_cfg_node[scene_idx]["render_offset"].as<std::vector<Scalar>>();
  unity_render_offset_ = Vector<3>(render_offset.data());
  return true;
}

bool AgileEnv::configDynamicObjects(const std::string &yaml_file) {
  //
  if (!(file_exists(yaml_file))) {
    logger_.error("Configuration file %s does not exists.", yaml_file);
    return false;
  }
  YAML::Node cfg_node = YAML::LoadFile(yaml_file);

  // logger_.info("Configuring dynamic objects");

  int num_objects = cfg_node["N"].as<int>();
  // create static objects
  for (int i = 0; i < num_objects; i++) {
    std::string object_id = "Object" + std::to_string(i + 1);
    std::string prefab_id = cfg_node[object_id]["prefab"].as<std::string>();
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    // load location, rotation and size
    std::vector<Scalar> posvec =
      (cfg_node[object_id]["position"]).as<std::vector<Scalar>>();
    std::vector<Scalar> rotvec =
      (cfg_node[object_id]["rotation"]).as<std::vector<Scalar>>();
    std::vector<Scalar> scalevec =
      (cfg_node[object_id]["scale"]).as<std::vector<Scalar>>();

    obj->setPosition(Vector<3>(posvec.data()));
    obj->setRotation(Quaternion(rotvec.data()));
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(Vector<3>(scalevec.data()));

    std::string csv_name = cfg_node[object_id]["csvtraj"].as<std::string>();
    std::string csv_file = obstacle_cfg_path_ + std::string("/csvtrajs/") +
                           csv_name + std::string(".csv");
    if (!(file_exists(csv_file))) {
      logger_.error("Configuration file %s does not exists.", csv_file);
      return false;
    }
    obj->loadTrajectory(csv_file);

    dynamic_objects_.push_back(obj);
  }
  num_dynamic_objects_ = dynamic_objects_.size();
    
  return true;
}

bool AgileEnv::configStaticObjects(const std::string &csv_file) {
  //
  if (!(file_exists(csv_file))) {
    logger_.error("Configuration file %s does not exists.", csv_file);
    return false;
  }
  std::ifstream infile(csv_file);
  int i = 0;
  for (auto &row : CSVRange(infile)) {

    // Read column 0 for time
    std::string object_id = "StaticObject" + std::to_string(i + 1);
    std::string prefab_id = (std::string)row[0];

    //
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    //
    Vector<3> pos;
    pos << std::stod((std::string)row[1]), std::stod((std::string)row[2]),
      std::stod((std::string)row[3]);

    Quaternion quat;
    quat.w() = std::stod((std::string)row[4]);
    quat.x() = std::stod((std::string)row[5]);
    quat.y() = std::stod((std::string)row[6]);
    quat.z() = std::stod((std::string)row[7]);

    Vector<3> scale;
    scale << std::stod((std::string)row[8]), std::stod((std::string)row[9]),
      std::stod((std::string)row[10]);

    //
    obj->setPosition(pos);
    obj->setRotation(quat);
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(scale);
    static_objects_.push_back(obj);
  }
  num_static_objects_ = static_objects_.size();

  return true;
}

bool AgileEnv::configCamera(const YAML::Node &cfg) {
  if (!cfg["rgb_camera"]) {
    logger_.error("Cannot config RGB Camera");
    return false;
  }

  if (!cfg["rgb_camera"]["on"].as<bool>()) {
    logger_.warn("Camera is off. Please turn it on.");
    return false;
  }

  if (quad_ptr_->getNumCamera() >= 1) {
    logger_.warn("Camera has been added. Skipping the camera configuration.");
    return false;
  }

  // create camera
  rgb_camera_ = std::make_shared<RGBCamera>();

  // load camera settings
  std::vector<Scalar> t_BC_vec =
    cfg["rgb_camera"]["t_BC"].as<std::vector<Scalar>>();
  std::vector<Scalar> r_BC_vec =
    cfg["rgb_camera"]["r_BC"].as<std::vector<Scalar>>();

  //
  Vector<3> t_BC(t_BC_vec.data());
  Matrix<3, 3> r_BC =
    (AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ()) *
     AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
     AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()))
      .toRotationMatrix();
  std::vector<bool> post_processing = {false, false, false};
  post_processing[0] = cfg["rgb_camera"]["enable_depth"].as<bool>();
  post_processing[1] = cfg["rgb_camera"]["enable_segmentation"].as<bool>();
  post_processing[2] = cfg["rgb_camera"]["enable_opticalflow"].as<bool>();

  //
  rgb_camera_->setFOV(cfg["rgb_camera"]["fov"].as<Scalar>());
  rgb_camera_->setWidth(cfg["rgb_camera"]["width"].as<int>());
  rgb_camera_->setChannels(cfg["rgb_camera"]["channels"].as<int>());
  rgb_camera_->setHeight(cfg["rgb_camera"]["height"].as<int>());
  rgb_camera_->setRelPose(t_BC, r_BC);
  rgb_camera_->setPostProcessing(post_processing);


  // add camera to the quadrotor
  quad_ptr_->addRGBCamera(rgb_camera_);

  // adapt parameters
  img_width_ = rgb_camera_->getWidth();
  img_height_ = rgb_camera_->getHeight();
  rgb_img_ = cv::Mat::zeros(img_height_, img_width_,
                            CV_MAKETYPE(CV_8U, rgb_camera_->getChannels()));
  depth_img_ = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
  return true;
}

bool AgileEnv::addQuadrotorToUnity(const std::shared_ptr<UnityBridge> bridge) {
  if (!quad_ptr_) return false;
  bridge->addQuadrotor(quad_ptr_);

  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    bridge->addDynamicObject(dynamic_objects_[i]);
  }

  //
  bridge->setRenderOffset(unity_render_offset_);
  bridge->setObjectCSV(static_object_csv_);
  return true;
}

bool AgileEnv::setUnity(bool render) {
  unity_render_ = render;
  if (!unity_render_ || unity_bridge_ptr_ != nullptr) {
    logger_.warn(
      "Unity render is False or Flightmare Bridge has been already created. "
      "Cannot set Unity.");
    return false;
  }
  // create unity bridge
  unity_bridge_ptr_ = UnityBridge::getInstance();
  // add objects to Unity

  addQuadrotorToUnity(unity_bridge_ptr_);

  logger_.info("Flightmare Bridge created.");
  return true;
}


bool AgileEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}


FrameID AgileEnv::updateUnity(const FrameID frame_id) {
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(frame_id);
    return unity_bridge_ptr_->handleOutput(frame_id);
  } else {
    return 0;
  }
}


void AgileEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

int AgileEnv::getNumDetectedObstacles(void) { return num_detected_obstacles_; }

std::ostream &operator<<(std::ostream &os, const AgileEnv &agile_env) {
  os.precision(3);
  os << "Agile Environment:\n"
     << "obs dim =            [" << agile_env.obs_dim_ << "]\n"
     << "act dim =            [" << agile_env.act_dim_ << "]\n"
     << "#dynamic objects=    [" << agile_env.num_dynamic_objects_ << "]\n"
     << "#static objects=     [" << agile_env.num_static_objects_ << "]\n"
     << "obstacle dim =       [" << agile_env.num_detected_obstacles_ << "]\n"
     << "sim dt =             [" << agile_env.sim_dt_ << "]\n"
     << "max_t =              [" << agile_env.max_t_ << "]\n"
     << "act_mean =           [" << agile_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << agile_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << agile_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << agile_env.obs_std_.transpose() << "]"
     << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib