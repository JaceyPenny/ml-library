interface SupervisedLearner {
  /**
   * Return the name of this learner
   */
  String name();

  /**
   * Make a prediction
   */
  Vector predict(Vector in);

  /**
   * Determines whether or not this Learner is valid (and therefore trainable).
   */
  boolean isValid();
}
