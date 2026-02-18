/**
 * @file SunSensorFactor.h
 * @brief A factor that processes sun sensor measurements
 * @date March 2023
 * @author Travis Driver
 */

/**
 * A factor that relates sun vector measurements in the camera frame to the 
 * estimated sun vector in the body-fixed frame.
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <math.h>

#include <boost/optional/optional_io.hpp>


namespace phomo {

class SunSensorFactor
    : public gtsam::NoiseModelFactor2<gtsam::Unit3,  // sun vector
                                      gtsam::Pose3   // camera pose
                                      > {
 private:
  // measurement information
  gtsam::Unit3 sbar_;  // sun direction expressed in camera frame

 public:
  /// shorthand for base class type
  using Base = gtsam::NoiseModelFactor2<gtsam::Unit3, gtsam::Pose3>;

  /**
   * @brief Constructor
   * @param sKey    associated sun vector variable key
   * @param TKey    associated camera pose variable key
   * @param sbar    sun vector measurement
   * @param model   noise model for sbar
   */
  SunSensorFactor(gtsam::Key sKey, gtsam::Key TKey, const gtsam::Unit3 sbar,
                  gtsam::SharedNoiseModel model)
      : Base(model, sKey, TKey), sbar_(sbar) {}

  /** 
   * @brief error function
   * @param s    the sun vector in body-fixed frame as Unit3
   * @param H1    optional 2x2 Jacobian matrix wrt s
   * @param H2    optional 2x6 Jacobian matrix wrt T_AC
   */
  gtsam::Vector evaluateError(const gtsam::Unit3 &s, const gtsam::Pose3 &T_AC,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2) const override {
    // mission angle from camera pose and landmark position
    gtsam::Matrix36 dR_AC_dT;
    gtsam::Matrix23 dstilde_dR_CA;
    gtsam::Matrix22 dstilde_ds;

    // Rotate sun vector into camera frame and get Jacobians.
    gtsam::Rot3 R_AC = T_AC.rotation(dR_AC_dT);
    gtsam::Rot3 R_CA = R_AC.inverse();
    gtsam::Unit3 stilde = R_CA.rotate(s, dstilde_dR_CA, dstilde_ds);

    // Need to account for inversion of R_AC, i.e., 
    // dstilde/dR_CA = -B^T * R_CA * [s]^, but we need 
    // dstilde/dR_AC = B^T * R_CA * [s]^ * R_AC (see Appendix A)
    gtsam::Matrix23 dstilde_dR_AC = -dstilde_dR_CA * R_AC.matrix();

    // Chain rule to get dstilde/dT
    gtsam::Matrix26 dstilde_dT = dstilde_dR_AC * dR_AC_dT;

    // Compute error and Jacobians.
    gtsam::Matrix22 dF_dstilde;
    gtsam::Matrix22 dF_dsbar;
    gtsam::Vector2 F = stilde.errorVector(sbar_, dF_dstilde, dF_dsbar);

    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    if (H1)
      *H1 = dF_dstilde * dstilde_ds;  // dF/ds
    if (H2)
      *H2 = dF_dstilde * dstilde_dT;  // dF/dT

    // return error vector
    return F;
  }
};

}  // namespace phomo
