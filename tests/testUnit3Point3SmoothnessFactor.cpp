#include <CppUnitLite/TestHarness.h>
#include <cpp/Unit3Point3SmoothnessFactor.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>

using namespace std;
using namespace phomo;
using namespace gtsam;

double get_random(double min, double max) {
  return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

/* ************************************************************************* */
TEST(Unit3Point3SmoothnessFactor, Jacobians) {
  // Define measurement.
  double meas = M_1_PI / 2.;

  // Create the factor.
  Key liKey(0), niKey(0), ljKey(1);
  SharedNoiseModel model = noiseModel::Isotropic::Sigma(1, 1e-2);
  Unit3Point3SmoothnessFactor factor(liKey, niKey, ljKey, meas, model);

  // Define linearization point.
  Point3 li(get_random(-100, 100), get_random(-100, 100), get_random(-100, 100));
  Unit3 ni(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));
  Point3 lj(get_random(-100, 100), get_random(-100, 100), get_random(-100, 100));

  // Calculate numerical derivatives.
  auto f = [&factor](const Point3& li, const Unit3& ni, const Point3& lj) {
    Matrix actualH1, actualH2, actualH3;
    return factor.evaluateError(li, ni, lj, &actualH1, &actualH2, &actualH3);
  };
  Matrix numericalH1 =
      numericalDerivative31<Vector, Point3, Unit3, Point3>(f, li, ni, lj);
  Matrix numericalH2 =
      numericalDerivative32<Vector, Point3, Unit3, Point3>(f, li, ni, lj);
  Matrix numericalH3 =
      numericalDerivative33<Vector, Point3, Unit3, Point3>(f, li, ni, lj);

  // Calculate actual derivatives.
  Matrix actualH1, actualH2, actualH3;
  factor.evaluateError(li, ni, lj, &actualH1, &actualH2, &actualH3);

  // Compare!
  EXPECT(assert_equal(numericalH1, actualH1, 1e-8));
  EXPECT(assert_equal(numericalH2, actualH2, 1e-8));
  EXPECT(assert_equal(numericalH3, actualH3, 1e-8));
}

/* ************************************************************************* */
int main() {
  srand(time(nullptr));
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */