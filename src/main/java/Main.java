import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

class Main {
  public static final Random RANDOM = new Random();

  private static int EPOCHS = 10;
  private static long EXECUTION_TIMESTAMP = System.currentTimeMillis();

  private static int BATCH_SIZE = 5;
  private static double MOMENTUM = 0.8;
  public static double LEARNING_RATE = 0.0001;

  private static void runAssignment1() {
    Matrix features = Matrix.fromARFF("data/housing_features.arff");
    Matrix labels = Matrix.fromARFF("data/housing_labels.arff");

    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.addLayer(new LinearLayer(features.cols(), labels.cols()));

    LearnerEvaluator<NeuralNetwork> evaluator = new LearnerEvaluator<>(neuralNetwork);
    double finalError = evaluator.crossValidation(features, labels, 10, 5);
    System.out.println(finalError);
  }

  private static void runAssignment2() {
    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.setLearningRate(0.03);
    neuralNetwork.setMomentum(0);

    neuralNetwork.addLayer(new LinearLayer(784, 80));
    neuralNetwork.addLayer(new TanhLayer(80));
    neuralNetwork.addLayer(new LinearLayer(80, 30));
    neuralNetwork.addLayer(new TanhLayer(30));
    neuralNetwork.addLayer(new LinearLayer(30, 10));
    neuralNetwork.addLayer(new TanhLayer(10));

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

    LearnerEvaluator<NeuralNetwork> evaluator =
        new LearnerEvaluator<>(neuralNetwork, LearnerEvaluator.TrainingType.BASIC);

    // Run a test to show baseline accuracy
    int misclassifications = evaluator.countMisclassifications(testingFeatures, testingLabels);
    System.out.printf("Before training. Misclassifications: %d\n", misclassifications);

    // Measure and report accuracy
    for (int i = 0; i < EPOCHS; i++) {
      long startTime = System.currentTimeMillis();
      evaluator.train(trainingFeatures, trainingLabels);
      long endTime = System.currentTimeMillis();

      System.out.printf("Finished training epoch %d: %d ms. Misclassifications: ", i + 1, endTime - startTime);

      misclassifications = evaluator.countMisclassifications(testingFeatures, testingLabels);
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

    NeuralNetwork stochasticNeuralNetwork = new NeuralNetwork();
    stochasticNeuralNetwork.addLayer(new LinearLayer(29, 20));
    stochasticNeuralNetwork.addLayer(new TanhLayer(20));
    stochasticNeuralNetwork.addLayer(new LinearLayer(20, 4));
    stochasticNeuralNetwork.addLayer(new TanhLayer(4));

    stochasticNeuralNetwork.setMomentum(MOMENTUM);
    stochasticNeuralNetwork.setLearningRate(LEARNING_RATE);

    NeuralNetwork miniBatchNeuralNetwork = stochasticNeuralNetwork.copy();
    miniBatchNeuralNetwork.setMomentum(0);

    stochasticNeuralNetwork.initializeWeights();
    miniBatchNeuralNetwork.initializeWeights();

    PrintWriter stochasticMisclassificationsTraining;
    PrintWriter stochasticMisclassificationsTesting;

    PrintWriter miniBatchMisclassificationsTraining;
    PrintWriter miniBatchMisclassificationsTesting;

    try {
      stochasticMisclassificationsTraining = getPrintWriterWithName("stochastic_mc_train.csv");
      stochasticMisclassificationsTesting = getPrintWriterWithName("stochastic_mc_test.csv");

      stochasticMisclassificationsTraining.println("steps,time,misclassifications");
      stochasticMisclassificationsTesting.println("steps,time,misclassifications");

      miniBatchMisclassificationsTraining = getPrintWriterWithName("minibatch_mc_train.csv");
      miniBatchMisclassificationsTesting = getPrintWriterWithName("minibatch_mc_test.csv");
      miniBatchMisclassificationsTraining.println("steps,time,misclassifications");
      miniBatchMisclassificationsTesting.println("steps,time,misclassifications");
    } catch (IOException ioException) {
      System.err.println("There was an error creating your files.");
      return;
    }

    // Run stochastic
    System.out.println("Running stochastic process...");
    stochasticNeuralNetwork.printTopology();

    LearnerEvaluator<NeuralNetwork> stochasticEvaluator = new LearnerEvaluator<>(
        stochasticNeuralNetwork, LearnerEvaluator.TrainingType.STOCHASTIC);
    stochasticEvaluator.setTrainingPrintWriter(stochasticMisclassificationsTraining);
    stochasticEvaluator.setTestingPrintWriter(stochasticMisclassificationsTesting);
    stochasticEvaluator.resetMetrics();

    // Gather statistics before training
    stochasticEvaluator.countMisclassifications(trainingFeatures, trainingLabels, true);
    stochasticEvaluator.countMisclassifications(testingFeatures, testingLabels, false);

    // Begin training
    for (int i = 0; i < trainingFeatures.rows(); i++) {
      // Train with exactly one row
      stochasticEvaluator.trainSingleRow(trainingFeatures, trainingLabels, i);

      stochasticEvaluator.countMisclassifications(trainingFeatures, trainingLabels, true);
      stochasticEvaluator.countMisclassifications(testingFeatures, testingLabels, false);
    }

    // Run mini batch
    System.out.println("Running mini batch process...");
    miniBatchNeuralNetwork.printTopology();

    LearnerEvaluator<NeuralNetwork> miniBatchEvaluator = new LearnerEvaluator<>(
        miniBatchNeuralNetwork, LearnerEvaluator.TrainingType.MINI_BATCH);
    miniBatchEvaluator.setTrainingPrintWriter(miniBatchMisclassificationsTraining);
    miniBatchEvaluator.setTestingPrintWriter(miniBatchMisclassificationsTesting);
    miniBatchEvaluator.resetMetrics();
    miniBatchEvaluator.setBatchSize(BATCH_SIZE);

    miniBatchEvaluator.countMisclassifications(trainingFeatures, trainingLabels, true);
    miniBatchEvaluator.countMisclassifications(testingFeatures, testingLabels, false);

    for (int i = 0; i < trainingFeatures.rows() / BATCH_SIZE; i++) {
      miniBatchEvaluator.trainSingleMiniBatch(trainingFeatures, trainingLabels, BATCH_SIZE, i);

      miniBatchEvaluator.countMisclassifications(trainingFeatures, trainingLabels, true);
      miniBatchEvaluator.countMisclassifications(testingFeatures, testingLabels, false);
    }

    stochasticMisclassificationsTraining.close();
    stochasticMisclassificationsTesting.close();
    miniBatchMisclassificationsTraining.close();
    miniBatchMisclassificationsTesting.close();
  }

  private static PrintWriter getPrintWriterWithName(String fileName)
      throws IOException {
    File outputDirectory = new File("output/" + EXECUTION_TIMESTAMP);
    outputDirectory.mkdirs();
    File outputFile = new File("output/" + EXECUTION_TIMESTAMP + "/" + fileName);
    outputFile.createNewFile();

    return new PrintWriter(new FileOutputStream(outputFile));
  }

  private static void runAssignment4() {

  }

  public static void main(String[] args) {
    runAssignment4();
  }
}
