#include <CppUnitLite/TestHarness.h>
#include <cpp/McEwenFactor2.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>

using namespace std;
using namespace phomo;
using namespace gtsam;

double get_random(double min, double max) {
  return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

Vector factorError(const Unit3& s, const Unit3& n, const Pose3& T,
                   const Point3& l, double a, double lmda, double xi,
                   const McEwenFactor2 &factor) {
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5,
      actualH6, actualH7;
  return factor.evaluateError(s, n, T, l, a, lmda, xi, &actualH1, &actualH2,
                              &actualH3, &actualH4, &actualH5, &actualH6, 
                              &actualH7);
}

/* ************************************************************************* */
TEST(McEwenFactor2, Jacobians) {
  // Define measurement.
  double meas = 0.1;

  // Create the factor.
  Key sKey(0), nKey(1), TKey(2), lKey(3), aKey(4), lmdaKey(5), xiKey(6);
  SharedNoiseModel model = noiseModel::Isotropic::Sigma(1, 0.5);
  McEwenFactor2 factor(sKey, nKey, TKey, lKey, aKey, 
                       lmdaKey, xiKey, meas, model);

  // Define linearization point.
  std::random_device rd;   // Non-deterministic random seed
  std::mt19937 rng(rd());  // Mersenne Twister PRNG
  Unit3 s(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));
  Unit3 n(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));
  Pose3 T = Pose3(
    Rot3::Random(rng), Point3(get_random(-100, 100), get_random(-100, 100), get_random(-100, 100)));
  Point3 l(get_random(-100, 100), get_random(-100, 100), get_random(-100, 100));
  double a = 0.5;
  double lmda = 0.8;
  double xi = 0.1;

  // Calculate numerical derivatives.
  Matrix numericalH1 = numericalDerivative11<Vector, Unit3>(
    std::bind(&factorError, std::placeholders::_1, n, T, l, a, lmda, xi, factor), s);
  Matrix numericalH2 = numericalDerivative11<Vector, Unit3>(
    std::bind(&factorError, s, std::placeholders::_1, T, l, a, lmda, xi, factor), n);
  Matrix numericalH3 = numericalDerivative11<Vector, Pose3>(
    std::bind(&factorError, s, n, std::placeholders::_1, l, a, lmda, xi, factor), T);
  Matrix numericalH4 = numericalDerivative11<Vector, Point3>(
    std::bind(&factorError, s, n, T, std::placeholders::_1, a, lmda, xi, factor), l);
  Matrix numericalH5 = numericalDerivative11<Vector, double>(
    std::bind(&factorError, s, n, T, l, std::placeholders::_1, lmda, xi, factor), a);
  Matrix numericalH6 = numericalDerivative11<Vector, double>(
    std::bind(&factorError, s, n, T, l, a, std::placeholders::_1, xi, factor), lmda);
  Matrix numericalH7 = numericalDerivative11<Vector, double>(
    std::bind(&factorError, s, n, T, l, a, lmda, std::placeholders::_1, factor), xi);

  // Calculate actual derivatives.
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5,
      actualH6, actualH7;
  factor.evaluateError( s, n, T, l, a, lmda, xi,
                        &actualH1, &actualH2, &actualH3, &actualH4,
                        &actualH5, &actualH6, &actualH7);

  // Compare!
  EXPECT(assert_equal(numericalH1, actualH1, 1e-8));
  EXPECT(assert_equal(numericalH2, actualH2, 1e-8));
  EXPECT(assert_equal(numericalH3, actualH3, 1e-8));
  EXPECT(assert_equal(numericalH4, actualH4, 1e-8));
  EXPECT(assert_equal(numericalH5, actualH5, 1e-8));
  EXPECT(assert_equal(numericalH6, actualH6, 1e-8));
  EXPECT(assert_equal(numericalH7, actualH7, 1e-8));
}

/* ************************************************************************* */
int main() {
  srand(time(nullptr));
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */