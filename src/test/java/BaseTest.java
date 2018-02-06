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
    for (int i = 0; i < target.len; i++) {
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
}
