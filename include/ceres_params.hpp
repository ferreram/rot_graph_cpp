#pragma once

#include <eigen3/Eigen/Core>

#include <sophus/so3.hpp>

#include <ceres/ceres.h>


// ===================================
//    AutoDiff Parametrization!
// ===================================

/*
    SO(3) Parametrization such as: R + dR = Exp(dR) * R
*/
struct AutoDiffLocalLeftSO3_Kernel
{
  template<typename T>
  bool operator()(const T* _x, const T* _delta, T* _x_plus_delta) const 
  {
    Eigen::Map<const Sophus::SO3<T>> ini_R(_x);
    Eigen::Map<const Eigen::Matrix<T,3,1>> lie_delta(_delta);

    Eigen::Map<Sophus::SO3<T>> opt_R(_x_plus_delta);
    opt_R = Sophus::SO3<T>::exp(lie_delta) * ini_R;

    return true;
  }
};

using AutoDiffLocalLeftSO3 = 
  ceres::AutoDiffLocalParameterization<AutoDiffLocalLeftSO3_Kernel,4,3>;


struct AutoDiffLogSO3_Kernel
{
  AutoDiffLogSO3_Kernel(const Sophus::SO3d& _Rij, const double _std=1.)
      : m_Rij(_Rij)
  {
    m_sqrt_info = (1./_std) * Eigen::Matrix3d::Identity();
  }

  template <typename T>
  bool operator()(const T *const _Rwi_param,
                  const T *const _Rwj_param,
                  T *_err) const
  {
    Eigen::Map<const Sophus::SO3<T>> Rwi(_Rwi_param);
    Eigen::Map<const Sophus::SO3<T>> Rwj(_Rwj_param);

    const Sophus::SO3<T> Rji = Rwj.inverse() * Rwi;

    Eigen::Map<Eigen::Matrix<T,3,1>> rot_err(_err);
    rot_err = m_sqrt_info.cast<T>() * (Rji * m_Rij).log();

    return true;
  }

  Sophus::SO3d m_Rij;

  Eigen::Matrix3d m_sqrt_info;
};

using AutoDiffLogSO3 = 
  ceres::AutoDiffCostFunction<AutoDiffLogSO3_Kernel, 3, 4, 4>;
