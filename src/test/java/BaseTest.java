import com.jace.math.Matrix;
import com.jace.math.Vector;
import org.junit.ComparisonFailure;

import java.util.Random;

abstract class BaseTest {
  private Random random = new Random();

  Matrix getSampleMatrix(int rows, int cols) {
    Matrix matrix = new Matrix(rows, cols);

    int i = 0;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        matrix.set(r, c, i++);
      }
    }

    return matrix;
  }

  Vector getSampleVector(int length) {
    Vector vector = new Vector(length);
    for (int i = 0; i < length; i++) {
      vector.set(i, i);
    }
    return vector;
  }

  Vector getRandomVector(int length) {
    Vector vector = new Vector(length);
    for (int i = 0; i < length; i++) {
      vector.set(i, random.nextDouble());
    }

    return vector;
  }

  Matrix getRandomMatrix(int rows, int cols) {
    Matrix matrix = new Matrix(rows, cols);

    for (int r = 0; r < rows; r++) {
      matrix.setRow(r, getRandomVector(cols));
    }

    return matrix;
  }

  void addRandomNoiseToVector(Vector target, double standardDeviations) {
    for (int i = 0; i < target.size(); i++) {
      double value = target.get(i);
      double gaussianShift = random.nextGaussian() * standardDeviations;

      target.set(i, value + gaussianShift);
    }
  }

  void addRandomNoiseToMatrix(Matrix target, double standardDeviations) {
    for (int r = 0; r < target.rows(); r++) {
      addRandomNoiseToVector(target.row(r), standardDeviations);
    }
  }

  void assertVectorEquals(Vector expected, Vector actual) {
    assertVectorEquals(expected, actual, 1e-4);
  }

  void assertVectorEquals(Vector expected, Vector actual, double tolerance)
      throws AssertionError {
    if (expected.size() != actual.size()) {
      throw new ComparisonFailure(
          "Sizes mismatch.", Integer.toString(expected.size()), Integer.toString(actual.size()));
    }

    for (int i = 0; i < expected.size(); i++) {
      if (Math.abs(expected.get(i) - actual.get(i)) > tolerance) {
        throw new ComparisonFailure("Values do not match", expected.toString(), actual.toString());
      }
    }
  }
}
