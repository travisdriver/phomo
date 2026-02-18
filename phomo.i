/**
 * @file     phomo.h
 * @brief    Example wrapper interface file for Python
 * @author   Travis Driver
 */

// This is an interface file for automatic Python wrapper generation.
// See gtsam.h for full documentation and more examples.

#include <cpp/LunarLambert_AffinePoly3Factor2.h>
#include <cpp/LunarLambert_AffinePoly4Factor2.h>
#include <cpp/McEwenFactor2.h>
#include <cpp/SunSensorFactor.h>
#include <cpp/Unit3Point3SmoothnessFactor.h>


namespace phomo {

class LunarLambert_AffinePoly3Factor2 : gtsam::NoiseModelFactor {
  LunarLambert_AffinePoly3Factor2(size_t sKey, size_t nKey, size_t Tkey,
                                  size_t lKey, size_t aKey, size_t wKey,
                                  size_t cKey, size_t lmdaKey, size_t xiKey,
                                  const double &Ik,
                                  gtsam::noiseModel::Base *model);
};

class LunarLambert_AffinePoly4Factor2 : gtsam::NoiseModelFactor {
  LunarLambert_AffinePoly4Factor2(size_t sKey, size_t nKey, size_t Tkey,
                                  size_t lKey, size_t aKey, size_t wKey,
                                  size_t cKey, size_t lmdaKey, size_t xiKey,
                                  const double &Ik,
                                  gtsam::noiseModel::Base *model);
};

class McEwenFactor2 : gtsam::NoiseModelFactor {
  McEwenFactor2(size_t sKey, size_t nKey, size_t Tkey, size_t lKey,
                      size_t aKey, size_t lmdaKey, size_t xiKey,
                      const double &Ik, gtsam::noiseModel::Base *model);
};

class SunSensorFactor : gtsam::NoiseModelFactor {
  SunSensorFactor(size_t sKey, size_t Tkey, const gtsam::Unit3 &sbar,
                  gtsam::noiseModel::Base *model);
};

class Unit3Point3SmoothnessFactor : gtsam::NoiseModelFactor {
  Unit3Point3SmoothnessFactor(size_t liKey, size_t niKey, size_t ljKey,
                              const double meas,
                              gtsam::noiseModel::Base *model);
};

}  // namespace phomo
