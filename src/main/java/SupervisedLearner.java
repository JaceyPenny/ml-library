// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.Random;

abstract class SupervisedLearner {
  /**
   * Return the name of this learner
   */
  abstract String name();

  /**
   * Train this supervised learner
   */
  abstract void train(Matrix features, Matrix labels);

  /**
   * Make a prediction
   */
  abstract Vector predict(Vector in);

  /**
   * Computes the sum squared error for this learner.
   */
  double computeSumSquaredError(Matrix testFeatures, Matrix expectedLabels) {
    double sumSquaredError = 0;

    for (int i = 0; i < testFeatures.rows(); i++) {
      Vector x_i = testFeatures.row(i);
      Vector expected_y = expectedLabels.row(i);
      Vector calculated_y = predict(x_i);

      // Calculate squared error
      calculated_y.scale(-1);
      calculated_y.add(expected_y);
      double squaredError = calculated_y.squaredMagnitude();

      sumSquaredError += squaredError;
    }

    return sumSquaredError;
  }

  protected void shuffleData(Matrix features, Matrix labels) {
    Random random = new Random();
    for (int i = features.rows(); i >= 2; i--) {
      int r = random.nextInt(i);
      features.swapRows(i - 1, r);
      labels.swapRows(i - 1, r);
    }
  }

  abstract double crossValidation(int folds, int repetitions, Matrix features, Matrix labels);

  /**
   * Measures the misclassificaions with the provided test data
   */
  int countMisclassifications(Matrix features, Matrix labels) {
    if (features.rows() != labels.rows())
      throw new IllegalArgumentException("Mismatching number of rows");
    int mis = 0;
    for (int i = 0; i < features.rows(); i++) {
      Vector feature = features.row(i);
      Vector prediction = predict(feature);
      Vector lab = labels.row(i);
      for (int j = 0; j < lab.size(); j++) {
        if (prediction.get(j) != lab.get(j))
          mis++;
      }
    }
    return mis;
  }
}
