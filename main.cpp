#include <iostream>
#include <fstream>
#include <iomanip>

#include <thread>

#include <map>

#include "ceres_params.hpp"

#include "utils.hpp"


int main(int argc, char** argv)
{
  std::cout << "\n*************************\n";
  std::cout << "AISGLA -- Rotation Graph!";
  std::cout << "\n*************************\n";

  if (argc < 2)
  {
    std::cerr << "Usage : ./rot_graph graph_matches.txt out_evo_file.txt" << std::endl;
    return -1;
  }

  // 1. Load graph of matches file
  // =============================================
  std::string graph_matches_path = argv[1];
  std::string out_evo_file_path = argv[2];

  std::ifstream graph_matches_file(graph_matches_path);
  
  // Skip 1st line
  std::string first_line;
  std::getline(graph_matches_file, first_line);

  // Set to store all poses and their matches
  std::map<int, Pose> map_poses;

  // Init 1st pose to id=0 / Rwi = I_3x3
  map_poses.emplace(0, Pose(0, Sophus::SO3d(), true));

  // To propagate rot if missing
  Sophus::SO3d prev_Rij;

  int img_i_id, img_j_id;
  int nb_inliers;
  double r00, r01, r02, r10, r11, r12, r20, r21, r22;

  while (graph_matches_file >> img_i_id >> img_j_id >> nb_inliers 
            >> r00 >> r01 >> r02
            >> r10 >> r11 >> r12
            >> r20 >> r21 >> r22)
  {
    std::cout << "\n I id #" << img_i_id << " / J id #" << img_j_id << " / nb_inliers = " << nb_inliers << "\n";
    Eigen::Matrix3d rot_ij;
    rot_ij << r00, r01, r02,
              r10, r11, r12,
              r20, r21, r22;
  
    Sophus::SO3d Rij(rot_ij);

    // Check if we have not jumped images
    auto it = map_poses.end();
    --it;
    if (img_i_id - it->first > 1)
    {
      std::cerr << "\nWarning!  Last added image was #" << it->first;
      std::cerr << "\nNew image to add is #" << img_i_id << "!";
      std::cerr << "\nAdding in between images...\n\n";

      for (int id=it->first; id < img_i_id; ++id)
      {
        const Sophus::SO3d Rwj = map_poses.at(id-1).m_Rwi * prev_Rij;
        const bool hold_cst = true;
        map_poses.emplace(id, Pose(id, Rwj, hold_cst));

        std::cout << "\nAdding image #" << id << " with rot : " << Rwj.log().transpose() << "\n";
      }
    }

    // Non-initialized pose! Need to initialize it from prev_Rij
    if (!map_poses.count(img_i_id))
    {
      const Sophus::SO3d Rwj = map_poses.at(img_i_id-1).m_Rwi * prev_Rij;

      const bool hold_cst = true;
      map_poses.emplace(img_i_id, Pose(img_i_id, Rwj, hold_cst));

      std::cout << "\nAdding new image #" << img_i_id << " with rot : " << Rwj.log().transpose() << "\n";
    }

    // We got next pose so we initialize it!
    if (img_j_id - img_i_id == 1)
    {
      const Sophus::SO3d Rwj = map_poses.at(img_i_id).m_Rwi * Rij;

      map_poses.emplace(img_j_id, Pose(img_j_id, Rwj));

      std::cout << "\nAdding next image #" << img_j_id << " with rot : " << Rwj.log().transpose() << "\n";

      prev_Rij = Rij;
    }

    std::cout << "\n img_j_id - img_i_id = " << img_j_id - img_i_id;

    if (img_j_id - img_i_id < 11 || nb_inliers > 150)
      map_poses.at(img_i_id).m_v_pose_matches.emplace_back(img_j_id, Rij, nb_inliers);
  }

  graph_matches_file.close();

  std::cout << "\nNumber of images added to the Rot graph problem: " << map_poses.size() << "\n";
  
  // 2. Create ceres problem!
  // ==============================
  ceres::Problem problem;

  ceres::LossFunctionWrapper* loss_function;
  loss_function = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(0.25), ceres::TAKE_OWNERSHIP);

  // Add SO3 pose to optimize (i.e add nodes)
  for (auto& pose_el : map_poses)
  {
    Pose& pose = pose_el.second;

    ceres::LocalParameterization *local_param = new AutoDiffLocalLeftSO3();
    problem.AddParameterBlock(pose.m_Rwi.data(), 4, local_param);

    if (pose.m_id == 0)
    // if (pose.m_hold_cst)
    {
      problem.SetParameterBlockConstant(pose.m_Rwi.data());
    }
  }

  // Add edges
  for (auto& pose_el : map_poses)
  {
    Pose& pose_i = pose_el.second;
    auto pose_j_it = map_poses.find(pose_i.m_id+1);

    // Add an edge with next pose in any case
    if (pose_j_it != map_poses.end())
    {
      const Sophus::SO3d Rij = pose_i.m_Rwi.inverse() * pose_j_it->second.m_Rwi;

      double std = 10.;
      for (const auto& rel_pose_j : pose_i.m_v_pose_matches)
      {
        if (rel_pose_j.m_id == pose_j_it->first)
        {
          std = 1.;
          break;
        }
      }

      ceres::CostFunction* f = new AutoDiffLogSO3(new AutoDiffLogSO3_Kernel(Rij, std));

      problem.AddResidualBlock(f, loss_function, pose_i.m_Rwi.data(), pose_j_it->second.m_Rwi.data());
    }

    // Add other edges
    for (const auto& rel_pose_j : pose_i.m_v_pose_matches)
    {
      {
        if (rel_pose_j.m_nb_inliers < 100)
          continue;

        pose_j_it = map_poses.find(rel_pose_j.m_id);

        if (pose_j_it == map_poses.end())
        {
          std::cerr << "\npose_j_it = map_poses.find(rel_pose_j.m_id); RETURNED map_poses.end()!\n";
          exit(-1);
        }

        const double std = 1.;

        ceres::CostFunction* f = new AutoDiffLogSO3(new AutoDiffLogSO3_Kernel(rel_pose_j.m_Rij, std));

        problem.AddResidualBlock(f, loss_function, pose_i.m_Rwi.data(), pose_j_it->second.m_Rwi.data());
      }
    }
  }

  // Write out initial poses
  std::ofstream out_file_ini("ini_rot_poses.txt");
  out_file_ini << std::fixed;
  out_file_ini << std::setprecision(9);

  Eigen::Vector3d ini_t(0.,0.,0.);

  for (const auto& pose : map_poses)
  {
    Eigen::Quaterniond q = pose.second.m_Rwi.unit_quaternion();

    out_file_ini << pose.first << " "  
             << ini_t.x() << " "
             << ini_t.y() << " "
             << ini_t.z() << " "
             << q.x() << " "
             << q.y() << " "
             << q.z() << " "
             << q.w() << "\n";
    
    out_file_ini.flush();
  }

  out_file_ini.close();


  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

  options.num_threads = 8;
  options.max_num_iterations = 10000;

  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  
  std::cout << "\n\n=========================================\n";
  std::cout << "     Ceres based Rotational Graph Optim";
  std::cout << "\n=========================================\n";

  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;

  // 3. Write solution!
  // ==============================
  std::ofstream out_file(out_evo_file_path);
  out_file << std::fixed;
  out_file << std::setprecision(9);

  Eigen::Vector3d t(0.,0.,0.);

  for (const auto& pose : map_poses)
  {
    Eigen::Quaterniond q = pose.second.m_Rwi.unit_quaternion();

    out_file << pose.first << " "  
             << t.x() << " "
             << t.y() << " "
             << t.z() << " "
             << q.x() << " "
             << q.y() << " "
             << q.z() << " "
             << q.w() << "\n";
    
    out_file.flush();
  }

  out_file.close();

  return 0;
}