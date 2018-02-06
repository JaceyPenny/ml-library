// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main {
  static void test(SupervisedLearner learner, String challenge) {
    // Load the training data
    String fn = "data/" + challenge;
    Matrix features = new Matrix();
    features.loadARFF(fn + "_features.arff");
    Matrix labels = new Matrix();
    labels.loadARFF(fn + "_labels.arff");

    // Measure and report accuracy
    double finalError = learner.crossValidation(10, 5, features, labels);
    System.out.println(finalError);
  }

  public static void main(String[] args) {
    test(new NeuralNetwork(), "housing");
  }
}
