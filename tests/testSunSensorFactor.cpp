#include <CppUnitLite/TestHarness.h>
#include <cpp/SunSensorFactor.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>

using namespace std;
using namespace phomo;
using namespace gtsam;


double get_random(double min, double max) {
  return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

/* ************************************************************************* */
TEST(SunSensorFactor, Jacobians) {
  // Define measurement.
  Unit3 meas(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));

  // Create the factor.
  Key sKey(0), TKey(0);
  SharedNoiseModel model = noiseModel::Isotropic::Sigma(2, 1e-2);
  SunSensorFactor factor(sKey, TKey, meas, model);

  // Define linearization point.
  std::random_device rd;   // Non-deterministic random seed
  std::mt19937 rng(rd());  // Mersenne Twister PRNG
  Unit3 s(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));
  Pose3 T = Pose3(Rot3::Random(rng), 
            Point3(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1)));

  // Calculate numerical derivatives.
  auto f = [&factor](const Unit3& s, const Pose3& T) {
    Matrix actualH1, actualH2;
    return factor.evaluateError(s, T, &actualH1, &actualH2);
  };
  Matrix numericalH1 =
      numericalDerivative21<Vector, Unit3, Pose3>(f, s, T);
  Matrix numericalH2 =
      numericalDerivative22<Vector, Unit3, Pose3>(f, s, T);

  // Calculate actual derivatives.
  Matrix actualH1, actualH2;
  factor.evaluateError(s, T, &actualH1, &actualH2);

  // Compare!
  EXPECT(assert_equal(numericalH1, actualH1, 1e-8));
  EXPECT(assert_equal(numericalH2, actualH2, 1e-8));
}

/* ************************************************************************* */
int main() {
  srand(time(nullptr));
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */