package com.jace;

import com.jace.evaluator.GradientEvaluator;
import com.jace.evaluator.LearnerEvaluator;
import com.jace.layer.*;
import com.jace.learner.GenerativeNeuralNetwork;
import com.jace.learner.NeuralNetwork;
import com.jace.math.Matrix;
import com.jace.math.Vector;
import com.jace.util.ChartMaker;
import com.jace.util.Console;
import com.jace.util.FileManager;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class Main {
  public static final Random RANDOM = new Random();

  private static int EPOCHS = 10;

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
    Console.i("%f", finalError);
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
    Matrix trainingLabels = Matrix.fromARFF("data/train_lab.arff").toOneHot();

    Matrix testingFeatures = Matrix.fromARFF("data/test_feat.arff");
    Matrix testingLabels = Matrix.fromARFF("data/test_lab.arff").toOneHot();

    long endMillis = System.currentTimeMillis();
    Console.i("Loaded all data in %d ms", endMillis - startMillis);

    trainingFeatures.scale(1.0 / 256.0);
    testingFeatures.scale(1.0 / 256.0);

    neuralNetwork.initialize();

    LearnerEvaluator<NeuralNetwork> evaluator =
        new LearnerEvaluator<>(neuralNetwork, LearnerEvaluator.TrainingType.BASIC);

    // Run a test to show baseline accuracy
    int misclassifications = evaluator.countMisclassifications(testingFeatures, testingLabels);
    Console.i("Before training. Misclassifications: %d", misclassifications);

    // Measure and report accuracy
    for (int i = 0; i < EPOCHS; i++) {
      long startTime = System.currentTimeMillis();
      evaluator.train(trainingFeatures, trainingLabels);
      long endTime = System.currentTimeMillis();

      System.out.printf("Finished training epoch %d: %d ms. Misclassifications: ", i + 1, endTime - startTime);

      misclassifications = evaluator.countMisclassifications(testingFeatures, testingLabels);
      Console.i("%d", misclassifications);
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

    stochasticNeuralNetwork.initialize();
    miniBatchNeuralNetwork.initialize();

    PrintWriter stochasticMisclassificationsTraining;
    PrintWriter stochasticMisclassificationsTesting;

    PrintWriter miniBatchMisclassificationsTraining;
    PrintWriter miniBatchMisclassificationsTesting;

    try {
      stochasticMisclassificationsTraining =
          FileManager.getPrintWriterWithName("stochastic_mc_train.csv");
      stochasticMisclassificationsTesting =
          FileManager.getPrintWriterWithName("stochastic_mc_test.csv");

      stochasticMisclassificationsTraining.println("steps,time,misclassifications");
      stochasticMisclassificationsTesting.println("steps,time,misclassifications");

      miniBatchMisclassificationsTraining =
          FileManager.getPrintWriterWithName("minibatch_mc_train.csv");
      miniBatchMisclassificationsTesting =
          FileManager.getPrintWriterWithName("minibatch_mc_test.csv");

      miniBatchMisclassificationsTraining.println("steps,time,misclassifications");
      miniBatchMisclassificationsTesting.println("steps,time,misclassifications");
    } catch (IOException ioException) {
      Console.e("There was an error creating your files.");
      return;
    }

    // Run stochastic
    Console.i("Running stochastic process...");

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
    Console.i("Running mini batch process...");
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

  private static void runMNISTwithCNN() {
    // Load the training data
    long startMillis = System.currentTimeMillis();

    Matrix trainingFeatures_temp = Matrix.fromARFF("data/train_feat.arff");
    Matrix trainingLabels_temp = Matrix.fromARFF("data/train_lab.arff");

    Matrix testingFeatures_temp = Matrix.fromARFF("data/test_feat.arff");
    Matrix testingLabels_temp = Matrix.fromARFF("data/test_lab.arff");

    Matrix.shuffleMatrices(trainingFeatures_temp, trainingLabels_temp);
    Matrix.shuffleMatrices(testingFeatures_temp, testingLabels_temp);

    Matrix trainingFeatures = trainingFeatures_temp.copyBlock(0, 0, 4500, 784);
    Matrix trainingLabels = trainingLabels_temp.copyBlock(0, 0, 4500, 1).toOneHot();

    Matrix testingFeatures = testingFeatures_temp.copyBlock(0, 0, 500, 784);
    Matrix testingLabels = testingLabels_temp.copyBlock(0, 0, 500, 1).toOneHot();

    trainingFeatures.scale(1.0 / 256.0);
    testingFeatures.scale(1.0 / 256.0);

    long endMillis = System.currentTimeMillis();
    Console.i("Loaded all data in %d ms", endMillis - startMillis);
    Console.i("Running MNIST classification example...");

    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.setLearningRate(0.01);

    neuralNetwork.addLayer(new ConvolutionLayer(new int[]{28, 28}, new int[]{5, 5, 32}, new int[]{28, 28, 32}));
    neuralNetwork.addLayer(new LeakyRectifierLayer(28 * 28 * 32));
    neuralNetwork.addLayer(new MaxPooling2DLayer(new int[]{28, 28, 32}));
    neuralNetwork.addLayer(new ConvolutionLayer(new int[]{14, 14, 32}, new int[]{5, 5, 32, 8}, new int[]{14, 14, 1, 8}));
    neuralNetwork.addLayer(new LeakyRectifierLayer(14 * 14 * 8));
    neuralNetwork.addLayer(new MaxPooling2DLayer(new int[]{14, 14, 8}));
    neuralNetwork.addLayer(new LinearLayer(7 * 7 * 8, 100));
    neuralNetwork.addLayer(new LeakyRectifierLayer(100));
    neuralNetwork.addLayer(new LinearLayer(100, 10));
    neuralNetwork.addLayer(new TanhLayer(10));

    neuralNetwork.initialize();

    LearnerEvaluator<NeuralNetwork> evaluator = new LearnerEvaluator<>(neuralNetwork, LearnerEvaluator.TrainingType.MINI_BATCH);
    evaluator.setBatchSize(10);

    int misclassifications = evaluator.countMisclassifications(testingFeatures, testingLabels);
    Console.i("Initial misclassifications: %d / %d", misclassifications, testingLabels.rows());

    for (int i = 0; i < 100; i++) {
      Console.i("Epoch %d ==============", i + 1);

      evaluator.train(trainingFeatures, trainingLabels);
      misclassifications = evaluator.countMisclassifications(testingFeatures, testingLabels);
      Console.i("Misclassifications: %d / %d", misclassifications, testingLabels.rows());
    }
  }

  private static void runAssignment4() {
    Console.i("ASSIGNMENT 4: Running the finite differencing test for the sample neural network.\n\n");

    NeuralNetwork neuralNetwork = new NeuralNetwork();

    neuralNetwork.addLayer(new ConvolutionLayer(new int[]{8, 8}, new int[]{5, 5, 4}, new int[]{8, 8, 4}));
    neuralNetwork.addLayer(new LeakyRectifierLayer(8 * 8 * 4));
    neuralNetwork.addLayer(new MaxPooling2DLayer(new int[]{8, 8, 4}));
    neuralNetwork.addLayer(new ConvolutionLayer(new int[]{4, 4, 4}, new int[]{3, 3, 4, 6}, new int[]{4, 4, 1, 6}));
    neuralNetwork.addLayer(new LeakyRectifierLayer(4 * 4 * 6));
    neuralNetwork.addLayer(new MaxPooling2DLayer(new int[]{4, 4, 6}));
    neuralNetwork.addLayer(new LinearLayer(2 * 2 * 6, 3));

    neuralNetwork.initialize();

    Vector testInput = new Vector(new double[]{
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
        3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
        4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
        5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4
    });

    Vector testOutput = new Vector(new double[]{
        1.0, 2.0, 3.0
    });

    Matrix testFeatures = new Matrix(1, 8 * 8);
    testFeatures.row(0).set(0, testInput);

    Matrix testLabels = new Matrix(1, 3);
    testLabels.row(0).set(0, testOutput);

    GradientEvaluator gradientEvaluator = new GradientEvaluator(neuralNetwork);
    gradientEvaluator.setTestData(testFeatures, testLabels);

    gradientEvaluator.checkAgainstFiniteDifferencing();
  }

  public static void runAssignment5() {
    Console.i("ASSIGNMENT 5: Run the US Dept of Labor Statistics data through a custom network.");

    Matrix features = new Matrix(357, 1);
    for (int i = 0; i < features.rows(); i++) {
      features.set(i, 0, i / 256.0);
    }

    Matrix labels = Matrix.fromARFF("data/labor_stats.arff").copyBlock(0, 0, 357, 1);

    Matrix subFeatures = features.copyBlock(0, 0, 256, 1);
    Matrix subLabels = labels.copyBlock(0, 0, 256, 1);

    LinearLayer outputLayer = new LinearLayer(101, 1);

    NeuralNetwork neuralNetwork = new NeuralNetwork();
    neuralNetwork.addLayer(new Assignment5LinearLayer());
    neuralNetwork.addLayer(new Assignment5ActivationLayer());
    neuralNetwork.addLayer(outputLayer);
    neuralNetwork.setLearningRate(0.005);

    Matrix csvOutput = new Matrix(357, 5);
    csvOutput.copyBlock(0, 0, features, 0, 0, 357, 1, false);
    csvOutput.copyBlock(0, 1, labels, 0, 0, 357, 1, false);

    LearnerEvaluator<NeuralNetwork> evaluator = new LearnerEvaluator<>(neuralNetwork);
    evaluator.setTrainingType(LearnerEvaluator.TrainingType.MINI_BATCH);
    evaluator.setBatchSize(8);

    neuralNetwork.initialize();
    evaluator.train(subFeatures, subLabels, 10);

    // predict without regularization
    Vector input = new Vector(1);
    for (int i = 0; i < 357; i++) {
      input.set(0, i / 256.0);
      Vector resultVector = neuralNetwork.predict(input);
      csvOutput.set(i, 2, resultVector.get(0));
    }

    outputLayer.setRegularizationType(ConnectedLayer.RegularizationType.L1);
    outputLayer.setRegularizationAmount(0.01);

    neuralNetwork.initialize();
    evaluator.train(subFeatures, subLabels, 10);

    // predict with L1 regularization
    for (int i = 0; i < 357; i++) {
      input.set(0, i / 256.0);
      double result = neuralNetwork.predict(input).get(0);
      csvOutput.set(i, 3, result);
    }

    outputLayer.setRegularizationType(ConnectedLayer.RegularizationType.L2);
    outputLayer.setRegularizationAmount(0.01);

    neuralNetwork.initialize();
    evaluator.train(subFeatures, subLabels, 10);

    // predict with L2 regularization
    for (int i = 0; i < 357; i++) {
      input.set(0, i / 256.0);
      double result = neuralNetwork.predict(input).get(0);
      csvOutput.set(i, 4, result);
    }

    PrintWriter writer;
    try {
      writer = FileManager.getPrintWriterWithName("assignment5output.csv");
    } catch (Exception e) {
      Console.e("Couldn't open file for output.");
      Console.exception(e);
      return;
    }

    // print values to console
    writer.println("Feature,Label,None,L1,L2");
    for (int i = 0; i < 357; i++) {
      Vector row = csvOutput.row(i);
      String output = row.get(0) + "," + row.get(1) + "," + row.get(2) + "," + row.get(3) + "," + row.get(4);
      writer.println(output);
    }

    writer.close();
  }

  private static void generateImage(GenerativeNeuralNetwork gnn, Vector inputState, String fileName) {
    Vector imageVector = new Vector(64 * 48 * 3);

    Vector state = new Vector(2 + inputState.size());
    state.set(2, inputState);

    for (int w = 0; w < 64; w++) {
      for (int h = 0; h < 48; h++) {
        state.set(0, w / 64.0);
        state.set(1, h / 48.0);

        Vector color = gnn.predict(state);

        int position = (h * 64 + w) * 3;

        imageVector.set(position, color.get(0));
        imageVector.set(position + 1, color.get(1));
        imageVector.set(position + 2, color.get(2));
      }
    }

    try {
      FileManager.writeImageFromVector(fileName, imageVector, 64, 48);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void runAssignment6() {
    Console.i("Loading data from \"%s\"...", "data/observations.arff");

    Matrix observations = Matrix.fromARFF("data/observations.arff");
    observations.scale(1 / 256.0);

    Console.i("Data finished loading");

    GenerativeNeuralNetwork observationNetwork =
        new GenerativeNeuralNetwork(64, 48, 2, observations.rows());

    observationNetwork.setLearningRate(0.1);
    observationNetwork.setMomentum(0);

    observationNetwork.addLayer(new LinearLayer(4, 12));
    observationNetwork.addLayer(new TanhLayer(12));
    observationNetwork.addLayer(new LinearLayer(12, 12));
    observationNetwork.addLayer(new TanhLayer(12));
    observationNetwork.addLayer(new LinearLayer(12, 3));
    observationNetwork.addLayer(new TanhLayer(3));

    observationNetwork.initialize();

    Console.i("Beginning training...");

    observationNetwork.trainUnsupervised(observations);

    Console.i("Finished training. Writing images...");

    Matrix estimatedStates = observationNetwork.getEstimatedStates();

    ChartMaker maker = new ChartMaker();
    maker.setData(estimatedStates);
    maker.setWidth(1000);
    maker.setPointSize(7);
    maker.setLabelFontSize(24);
    maker.setBufferPercentage(0.1);
    maker.setConnectPoints(true);

    maker.draw();

    try {
      maker.writeToFile(FileManager.getOutputFileWithName("intrinsic.png"));
    } catch (IOException e) {
      Console.exception(e);
    }

    Console.i("Loading actions...");

    Matrix actions = Matrix.fromARFF("data/actions.arff").toOneHot();
    actions.removeRow(actions.rows() - 1);

    Console.i("Finished loading actions...");

    Console.i("Creating features and labels from learned state representations...");

    double stateRangeX = estimatedStates.columnMax(0) - estimatedStates.columnMin(0);
    double stateMeanX = estimatedStates.columnMean(0);
    double stateRangeY = estimatedStates.columnMax(1) - estimatedStates.columnMin(1);
    double stateMeanY = estimatedStates.columnMean(1);

    Matrix features = estimatedStates.copy();
    features.removeRow(features.rows() - 1);

    for (int i = 0; i < features.rows(); i++) {
      features.row(i).set(0, (features.row(i).get(0) - stateMeanX) / stateRangeX);
      features.row(i).set(1, (features.row(i).get(1) - stateMeanY) / stateRangeY);
    }

    features = Matrix.joined(actions, features);

    Matrix labels = estimatedStates.copy();
    labels.removeRow(0);

    for (int i = 0; i < labels.rows(); i++) {
      labels.row(i).set(0, (labels.row(i).get(0) - stateMeanX) / stateRangeX);
      labels.row(i).set(1, (labels.row(i).get(1) - stateMeanY) / stateRangeY);
    }

    Console.i("Finished creating features and labels");

    Console.i("Creating and training state prediction network...");

    NeuralNetwork statePredictionNetwork = new NeuralNetwork();
    statePredictionNetwork.addLayer(new LinearLayer(6, 6));
    statePredictionNetwork.addLayer(new TanhLayer(6));
    statePredictionNetwork.addLayer(new LinearLayer(6, 2));
    statePredictionNetwork.addLayer(new TanhLayer(2));

    statePredictionNetwork.setLearningRate(0.1);
    statePredictionNetwork.setMomentum(0);

    statePredictionNetwork.initialize();

    LearnerEvaluator<NeuralNetwork> learnerEvaluator = new LearnerEvaluator<>(statePredictionNetwork);
    learnerEvaluator.setTrainingType(LearnerEvaluator.TrainingType.BASIC);

    for (int i = 0; i < 5000; i++) {
      learnerEvaluator.train(features, labels);
      statePredictionNetwork.setLearningRate(statePredictionNetwork.getLearningRate() * 0.99897);
    }

    Console.i("Prediction network trained");

    Console.i("Performing assignment 6 steps...");

    Matrix testActions = Matrix.deserialize(
        new Vector(new double[]{0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3}),
        11, 1).toOneHot();
    testActions.removeRow(10);

    Vector currentState = estimatedStates.row(0).copy();
    generateImage(observationNetwork, currentState, "frame0.png");
    currentState.set(0, (currentState.get(0) - stateMeanX) / stateRangeX);
    currentState.set(1, (currentState.get(1) - stateMeanY) / stateRangeY);

    for (int i = 0; i < testActions.rows(); i++) {
      Vector predictorInput = Vector.joined(testActions.row(i), currentState);
      currentState = statePredictionNetwork.predict(predictorInput);
      Vector stateForImage = currentState.copy();
      stateForImage.set(0, stateForImage.get(0) * stateRangeX + stateMeanX);
      stateForImage.set(1, stateForImage.get(1) * stateRangeY + stateMeanY);
      generateImage(observationNetwork, stateForImage, "frame" + (i + 1) + ".png");
    }
  }

  public static void main(String[] args) {
    Console.init();
    Console.setMessageLevel(Console.MessageLevel.DEBUG);

    runAssignment6();
  }
}
