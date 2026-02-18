#include <CppUnitLite/TestHarness.h>
#include <cpp/LunarLambert_AffinePoly3Factor2.h>
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
                   const Point3& l, double a, const Vector2& w, const Vector4& c,
                   double lmda, double xi,
                   const LunarLambert_AffinePoly3Factor2 &factor) {
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5,
      actualH6, actualH7, actualH8, actualH9;
  return factor.evaluateError(s, n, T, l, a, w, c, lmda, xi, &actualH1, &actualH2,
                              &actualH3, &actualH4, &actualH5, &actualH6,
                              &actualH7, &actualH8, &actualH9);
}

/* ************************************************************************* */
TEST(LunarLambert_AffinePoly3Factor2, Jacobians) {
  // Define measurement.
  double meas = 0.1;

  // Create the factor.
  Key sKey(0), nKey(1), TKey(2), lKey(3), aKey(4), wKey(5), cKey(6), lmdaKey(7), xiKey(8);
  SharedNoiseModel model = noiseModel::Isotropic::Sigma(1, 0.5);
  LunarLambert_AffinePoly3Factor2 factor(sKey, nKey, TKey, lKey, aKey, wKey, 
                                         cKey, lmdaKey, xiKey, meas, model);

  // Define linearization point.
  std::random_device rd;   // Non-deterministic random seed
  std::mt19937 rng(rd());  // Mersenne Twister PRNG
  Unit3 s(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));
  Unit3 n(get_random(-1, 1), get_random(-1, 1), get_random(-1, 1));
  Pose3 T = Pose3(
    Rot3::Random(rng), Point3(get_random(-100, 100), get_random(-100, 100), get_random(-100, 100)));
  Point3 l(get_random(-100, 100), get_random(-100, 100), get_random(-100, 100));
  double a = 0.5;
  double w0 = 0.896;
  double w1 = -8.87e-3 * 180. / M_PI;
  Vector2 w(w0, w1);

  double c0 = 0.0746;
  double c1 = -1.65e-3 * 180. / M_PI;
  double c2 = 1.56e-5 * pow(180. / M_PI, 2);
  double c3 = -4.79e-8 * pow(180. / M_PI, 3);
  Vector4 c(c0, c1, c2, c3);

  double lmda = 0.8;
  double xi = 0.1;

  // Calculate numerical derivatives.
  Matrix numericalH1 = numericalDerivative11<Vector, Unit3>(
    std::bind(&factorError, std::placeholders::_1, n, T, l, a, w, c, lmda, xi, factor), s);
  Matrix numericalH2 = numericalDerivative11<Vector, Unit3>(
    std::bind(&factorError, s, std::placeholders::_1, T, l, a, w, c, lmda, xi, factor), n);
  Matrix numericalH3 = numericalDerivative11<Vector, Pose3>(
    std::bind(&factorError, s, n, std::placeholders::_1, l, a, w, c, lmda, xi, factor), T);
  Matrix numericalH4 = numericalDerivative11<Vector, Point3>(
    std::bind(&factorError, s, n, T, std::placeholders::_1, a, w, c, lmda, xi, factor), l);
  Matrix numericalH5 = numericalDerivative11<Vector, double>(
    std::bind(&factorError, s, n, T, l, std::placeholders::_1, w, c, lmda, xi, factor), a);
  Matrix numericalH6 = numericalDerivative11<Vector, Vector2>(
    std::bind(&factorError, s, n, T, l, a, std::placeholders::_1, c, lmda, xi, factor), w);
  Matrix numericalH7 = numericalDerivative11<Vector, Vector4>(
    std::bind(&factorError, s, n, T, l, a, w, std::placeholders::_1, lmda, xi, factor), c);
  Matrix numericalH8 = numericalDerivative11<Vector, double>(
    std::bind(&factorError, s, n, T, l, a, w, c, std::placeholders::_1, xi, factor), lmda);
  Matrix numericalH9 = numericalDerivative11<Vector, double>(
    std::bind(&factorError, s, n, T, l, a, w, c, lmda, std::placeholders::_1, factor), xi);

  // Calculate actual derivatives.
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5,
      actualH6, actualH7, actualH8, actualH9;
  factor.evaluateError( s, n, T, l, a, w, c, lmda, xi,
                        &actualH1, &actualH2, &actualH3, &actualH4,
                        &actualH5, &actualH6, &actualH7, &actualH8,
                        &actualH9);

  // Compare!
  EXPECT(assert_equal(numericalH1, actualH1, 1e-8));
  EXPECT(assert_equal(numericalH2, actualH2, 1e-8));
  EXPECT(assert_equal(numericalH3, actualH3, 1e-8));
  EXPECT(assert_equal(numericalH4, actualH4, 1e-8));
  EXPECT(assert_equal(numericalH5, actualH5, 1e-8));
  EXPECT(assert_equal(numericalH6, actualH6, 1e-8));
  EXPECT(assert_equal(numericalH7, actualH7, 1e-8));
  EXPECT(assert_equal(numericalH8, actualH8, 1e-8));
  EXPECT(assert_equal(numericalH9, actualH9, 1e-8));
}

/* ************************************************************************* */
int main() {
  srand(time(nullptr));
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */