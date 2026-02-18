/**
 * @file Unit3Point3SmoothnessFactor.h
 * @brief Enforces local smoothness constraint
 * @date June 2024
 * @author Travis Driver
 */

/**
 * This factor enforces a local smoothness constraint by encouraging the angle 
 * between a reference normal and neighboring landmarks to be ~90°.
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

class Unit3Point3SmoothnessFactor
    : public gtsam::NoiseModelFactor3<gtsam::Point3,  // reference landmark
                                      gtsam::Unit3,   // reference normal
                                      gtsam::Point3   // query landmark
                                      > {
 private:
  // measurement information
  double meas_;  // angle between normal and vector to neighboring landmark

 public:
  /// shorthand for base class type
  using Base =
      gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Unit3, gtsam::Point3>;

  /**
   * @brief Constructor
   * @param liKey   associated reference landmark variable key
   * @param niKey   associated reference normal variable key
   * @param ljKey   associated neighboring landmark variable key
   * @param meas    angle constraint in radians, i.e., pi/2 for our purposes
   * @param model   noise model for meas
   */
  Unit3Point3SmoothnessFactor(gtsam::Key liKey, gtsam::Key niKey,
                              gtsam::Key ljKey, const double meas,
                              gtsam::SharedNoiseModel model)
      : Base(model, liKey, niKey, ljKey), meas_(meas) {}

  /** 
   * @brief error function
   * @param li   reference landmark
   * @param ni   reference normal
   * @param lj   neighboring landmark
   * @param H1   optional 1x3 Jacobian matrix wrt li
   * @param H2   optional 1x2 Jacobian matrix wrt ni
   * @param H3   optional 1x3 Jacobian matrix wrt lj
   */
  gtsam::Vector evaluateError(const gtsam::Point3 &li, const gtsam::Unit3 &ni,
                              const gtsam::Point3 &lj,
                              gtsam::OptionalMatrixType H1,
                              gtsam::OptionalMatrixType H2,
                              gtsam::OptionalMatrixType H3) const override {
    // Compute unit direction from reference to query landmark.
    gtsam::Point3 rij = lj - li;
    gtsam::Matrix33 dr_dli = -1. * gtsam::Rot3::Identity().matrix();
    gtsam::Matrix33 dr_dlj = gtsam::Rot3::Identity().matrix();

    gtsam::Matrix23 dd_dr;
    gtsam::Unit3 dij = gtsam::Unit3::FromPoint3(rij, dd_dr);

    // Compute angle between direction and normal.
    gtsam::Matrix12 dc_dn;
    gtsam::Matrix12 dc_dd;
    double c = ni.dot(dij, dc_dn, dc_dd);
    double a = acos(c);  // in radians
    double da_dc = -1 / sqrt(1 - c * c);

    // Compute error and Jacobians.
    gtsam::Matrix13 da_dli = da_dc * dc_dd * dd_dr * dr_dli;
    gtsam::Matrix12 da_dni = da_dc * dc_dn;
    gtsam::Matrix13 da_dlj = da_dc * dc_dd * dd_dr * dr_dlj;

    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    if (H1) *H1 = da_dli;
    if (H2) *H2 = da_dni;
    if (H3) *H3 = da_dlj;

    // return error vector
    return (gtsam::Vector1() << a - meas_).finished();
  }
};

}  // namespace phomo
