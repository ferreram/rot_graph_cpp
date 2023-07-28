#pragma once

#include <vector>

#include <sophus/so3.hpp>

struct PoseMatches
{
  PoseMatches(const int _id, const Sophus::SO3d& _Rij, const int _nb_inliers)
    : m_id(_id), m_Rij(_Rij), m_nb_inliers(_nb_inliers)
  {}

  int m_id;
  Sophus::SO3d m_Rij;
  int m_nb_inliers;
};


struct Pose
{
  Pose(const int _id, const Sophus::SO3d& _Rwi, const bool _hold_cst=false)
    : m_id(_id), m_Rwi(_Rwi), m_hold_cst(_hold_cst)
  {}

  int m_id;
  Sophus::SO3d m_Rwi;
  bool m_hold_cst;

  std::vector<PoseMatches> m_v_pose_matches;
};