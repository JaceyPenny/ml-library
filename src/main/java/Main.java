// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main {
  public static int EPOCHS = 10;

  static void test(NeuralNetwork neuralNetwork) {
    neuralNetwork.addFirstLayer(LayerType.LINEAR, 784, 80);
    neuralNetwork.addLayer(LayerType.TANH, 80);
    neuralNetwork.addLayer(LayerType.LINEAR, 30);
    neuralNetwork.addLayer(LayerType.TANH, 30);
    neuralNetwork.addLayer(LayerType.LINEAR, 10);
    neuralNetwork.addLayer(LayerType.TANH, 10);

    // Load the training data
    long startMillis = System.currentTimeMillis();

    Matrix trainingFeatures = Matrix.fromARFF("data/train_feat.arff");
    Matrix trainingLabels_temp = Matrix.fromARFF("data/train_lab.arff");

    Matrix testingFeatures = Matrix.fromARFF("data/test_feat.arff");
    Matrix testingLabels_temp = Matrix.fromARFF("data/test_lab.arff");

    long endMillis = System.currentTimeMillis();
    System.out.printf("Loaded all data in %d ms\n", endMillis - startMillis);

    trainingFeatures.scale(1.0 / 256.0);
    testingFeatures.scale(1.0 / 256.0);

    Matrix trainingLabels = new Matrix(trainingLabels_temp.rows(), 10);
    Matrix testingLabels = new Matrix(testingLabels_temp.rows(), 10);

    for (int i = 0; i < trainingLabels.rows(); i++) {
      trainingLabels.set(i, (int) Math.round(trainingLabels_temp.get(i, 0)), 1);
    }

    for (int i = 0; i < testingLabels.rows(); i++) {
      testingLabels.set(i, (int) Math.round(testingLabels_temp.get(i, 0)), 1);
    }

    neuralNetwork.initializeWeights();

    // Run a test to show baseline accuracy
    int misclassifications = neuralNetwork.countMisclassifications(testingFeatures, testingLabels);
    System.out.printf("Before training. Misclassifications: %d\n", misclassifications);

    // Measure and report accuracy
    for (int i = 0; i < EPOCHS; i++) {
      long startTime = System.currentTimeMillis();
      neuralNetwork.train(trainingFeatures, trainingLabels);
      long endTime = System.currentTimeMillis();

      System.out.printf("Finished training epoch %d: %d ms. Misclassifications: ", i + 1, endTime - startTime);

      misclassifications = neuralNetwork.countMisclassifications(testingFeatures, testingLabels);
      System.out.println(misclassifications);
    }
  }

  public static void main(String[] args) {
    test(new NeuralNetwork());
  }
}
