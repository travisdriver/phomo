/**
 * @file McEwenFactor2.h
 * @brief A photometric factor related to the prediction of image intensity
 * @date March 2023
 * @author Travis Driver
 */

/**
 * A photometric factor related to the prediction of image intensity
 * measurements given the incidence, emission, and phase angles of the
 * associated landmark.
 *
 * This factor leverages the photometric model proposed by McEwen [1] with the 
 * exponential weighting function proposed by Gaskell et al. [2].
 *
 * Refs:
 * [1] A. S. McEwen, “Photometric functions for photoclinometry and other 
 *       applications,” Icarus, 1991. 
 *       https://doi.org/10.1016/0019-1035(91)90053-V
 * [2] R. W. Gaskell et al. "Characterizing and navigating small bodies with
 *       imaging data," Meteorics & Planetary Science, 2008.
 *       https://doi.org/10.1111/j.1945-5100.2008.tb00692.x
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

class McEwenFactor2
    : public gtsam::NoiseModelFactorN<gtsam::Unit3,   // sun vector
                                      gtsam::Unit3,   // surface normal
                                      gtsam::Pose3,   // camera pose
                                      gtsam::Point3,  // landmark position
                                      double,         // albedo
                                      double,         // image scale factor
                                      double          // image bias factor
                                      > {
 private:
  // measurement information
  double Ik_;  // pixel intensity at keypoint

 public:
  /// shorthand for base class type
  using Base = gtsam::NoiseModelFactorN<gtsam::Unit3, gtsam::Unit3, gtsam::Pose3,
                            gtsam::Point3, double, double, double>;

  /**
   * Constructor
   * @param sKey    associated sun vector variable key
   * @param nKey    associated surface normal variable key
   * @param TKey    associated camera pose variable key
   * @param aKey    associated albedo variable key
   * @param Ik      pixel intensity measurement
   * @param model   noise model for Ik
   */
  McEwenFactor2(gtsam::Key sKey, gtsam::Key nKey, gtsam::Key TKey,
                      gtsam::Key lKey, gtsam::Key aKey, gtsam::Key lmdaKey,
                      gtsam::Key xiKey, const double Ik,
                      gtsam::SharedNoiseModel model)
      : Base(model, sKey, nKey, TKey, lKey, aKey, lmdaKey, xiKey), Ik_(Ik) {}

  /** 
   * @brief error function
   * @param s     the sun vector in body frame in Unit3
   * @param n     the surface normal in body frame in Unit3
   * @param T_AC  the camera pose in Pose3
   * @param l     the landmark position in body frame in Point3
   * @param a     the albedo scalar
   * @param lmda  the image scale factor scalar
   * @param xi    the image bias factor scalar
   * @param H1    optional 1x2 Jacobian matrix wrt s
   * @param H2    optional 1x2 Jacobian matrix wrt n
   * @param H3    optional 1x6 Jacobian matrix wrt T_AC
   * @param H4    optional 1x3 Jacobian matrix wrt l
   * @param H5    optional 1x1 Jacobian matrix wrt a
   * @param H6    optional 1x1 Jacobian matrix wrt lmda
   * @param H7    optional 1x1 Jacobian matrix wrt xi
   */
  gtsam::Vector evaluateError(
      const gtsam::Unit3 &s, const gtsam::Unit3 &n, const gtsam::Pose3 &T_AC,
      const gtsam::Point3 &l,  // r_A_LA
      const double &a, const double &lmda, const double &xi,
      gtsam::OptionalMatrixType H1, gtsam::OptionalMatrixType H2,
      gtsam::OptionalMatrixType H3, gtsam::OptionalMatrixType H4,
      gtsam::OptionalMatrixType H5, gtsam::OptionalMatrixType H6,
      gtsam::OptionalMatrixType H7) const override {
    // Compute emission angle from camera pose and landmark position and define
    // Jacobians.
    gtsam::Matrix36 dr_dT;
    gtsam::Point3 r = T_AC.translation(dr_dT);  // r_A_CA
    gtsam::Point3 e = r - l;                    // r_A_CL

    gtsam::Matrix3 de_dr = gtsam::I_3x3;
    gtsam::Matrix3 de_dl = -gtsam::I_3x3;

    // Convert e to Unit3 and get Jacobian.
    gtsam::Matrix23 dd_de;
    gtsam::Unit3 d = gtsam::Unit3::FromPoint3(e, dd_de);

    // Incidence angle and Jacobians.
    gtsam::Matrix12 df_ds; /* df/ds */
    gtsam::Matrix12 df_dn; /* df/dn */
    double f = s.dot(n, df_ds, df_dn);

    // Emission angle and Jacobians.
    gtsam::Matrix12 dg_dd; /* dg/dd */
    gtsam::Matrix12 dg_dn; /* dg/dn */
    double g = d.dot(n, dg_dd, dg_dn);

    // Phase angle and Jacobians.
    gtsam::Matrix12 dh_ds; /* dh/ds */
    gtsam::Matrix12 dh_dd; /* dh/dd */
    double h = s.dot(d, dh_ds, dh_dd);

    // Phase weighting function and jacobians.
    double b = -3. * acos(h) / M_PI;
    double L = exp(b);

    double db_dh = 3. * (1. / sqrt(1. - h * h)) / M_PI;
    double dL_db = exp(b);
    gtsam::Matrix12 dL_ds = dL_db * db_dh * dh_ds;
    gtsam::Matrix12 dL_dd = dL_db * db_dh * dh_dd;

    // Compute estimated intensity.
    double Ik_hat = lmda * a * ((1. - L) * f + L * 2. * f / (f + g)) + xi;

    // Compute variable Jacobians.
    gtsam::Matrix12 dI_ds;
    gtsam::Matrix12 dI_dn;
    gtsam::Matrix13 dI_de;
    gtsam::Matrix16 dI_dT;
    gtsam::Matrix13 dI_dl;
    double dI_da;
    double dI_dlmda;
    double dI_dxi;

    dI_ds =
        lmda * a *
        ((1. - L) * df_ds - f * dL_ds + 2. * L * df_ds / (f + g) +
         2. * f * dL_ds / (f + g) - 2. * L * f * df_ds / ((f + g) * (f + g)));
    dI_dn = lmda * a *
            ((1. - L) * df_dn + 2. * L * df_dn / (f + g) +
             2. * (-df_dn - dg_dn) * L * f / ((f + g) * (f + g)));
    dI_de = lmda * a *
            (-f * dL_dd + 2.0 * f * dL_dd / (f + g) -
             2.0 * L * f * dg_dd / ((f + g) * (f + g))) *
            dd_de;
    dI_dT = dI_de * de_dr * dr_dT;
    dI_dl = dI_de * de_dl;
    dI_da = lmda * ((1. - L) * f + L * 2. * f / (f + g));
    dI_dlmda = a * ((1. - L) * f + L * 2. * f / (f + g));
    dI_dxi = 1.0;

    // dI_ds
    if (H1) *H1 = dI_ds;
    // dI_dn
    if (H2) *H2 = dI_dn;
    // dI_dT
    if (H3) *H3 = dI_dT;
    // dI_dl
    if (H4) *H4 = dI_dl;
    // dI_da
    if (H5) *H5 = (gtsam::Vector1() << dI_da).finished();
    // dI_dlmda
    if (H6) *H6 = (gtsam::Vector1() << dI_dlmda).finished();
    // dI_dxi
    if (H7) *H7 = (gtsam::Vector1() << dI_dxi).finished();

    // Return error vector.
    return (gtsam::Vector1() << Ik_hat - Ik_).finished();
  }
};

}  // namespace gtsam_example
