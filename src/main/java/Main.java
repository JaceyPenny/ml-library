// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main {
  private static int EPOCHS = 10;

  private static void runAssignment1() {
    Matrix features = Matrix.fromARFF("data/housing_features.arff");
    Matrix labels = Matrix.fromARFF("data/housing_labels.arff");

    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.addFirstLayer(LayerType.LINEAR, features.cols(), labels.cols());

    double finalError = neuralNetwork.crossValidation(10, 5, features, labels);
    System.out.println(finalError);
  }

  private static void runAssignment2() {
    NeuralNetwork neuralNetwork = new NeuralNetwork();

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

  public static void runAssignment3() {
    Matrix dataSet = Matrix.fromARFF("data/hypothyroid.arff");
    dataSet.shuffleRows();

    int trainingRows = Math.round(0.8f * dataSet.rows());
    int testingRows = dataSet.rows() - trainingRows;

    Matrix trainingFeatures = dataSet.copyBlock(0, 0, trainingRows, 29);
    Matrix trainingLabels = dataSet.copyBlock(0, 29, trainingRows, 1).toOneHot();

    Matrix testingFeatures = dataSet.copyBlock(trainingRows, 0, testingRows, 29);
    Matrix testingLabels = dataSet.copyBlock(trainingRows, 29, testingRows, 1).toOneHot();

    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.addFirstLayer(LayerType.LINEAR, 29, 20);
    neuralNetwork.addLayer(LayerType.TANH, 20);
    neuralNetwork.addLayer(LayerType.LINEAR, 4);
    neuralNetwork.addLayer(LayerType.TANH, 4);

    neuralNetwork.initializeWeights();

    int misclassifications = neuralNetwork.countMisclassifications(testingFeatures, testingLabels);

    System.out.printf("Baseline; Misclassifications: %d / %d\n", misclassifications, testingLabels.rows());

    for (int i = 0; i < EPOCHS; i++) {
      neuralNetwork.trainMiniBatch(trainingFeatures, trainingLabels, 20);
      misclassifications = neuralNetwork.countMisclassifications(testingFeatures, testingLabels);
      System.out.printf("Epoch %d; Misclassifications: %d / %d\n", i + 1, misclassifications, testingLabels.rows());
    }
  }

  public static void main(String[] args) {
    runAssignment3();
  }
}
