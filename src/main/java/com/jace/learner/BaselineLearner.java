package com.jace.learner;// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import com.jace.math.Matrix;
import com.jace.math.Vector;

public class BaselineLearner implements SupervisedLearner {
  double[] mode;

  @Override
  public String name() {
    return "Baseline";
  }

  @Override
  public boolean isValid() {
    return true;
  }

  void train(Matrix features, Matrix labels) {
    mode = new double[labels.cols()];
    for (int i = 0; i < labels.cols(); i++) {
      if (labels.getMetadata().attributeIsContinuous(i))
        mode[i] = labels.columnMean(i);
      else
        mode[i] = labels.mostCommonValue(i);
    }
  }

  @Override
  public Vector predict(Vector in) {
    return new Vector(mode);
  }
}
