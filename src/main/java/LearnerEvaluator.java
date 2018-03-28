import java.io.PrintWriter;

public class LearnerEvaluator<T extends SupervisedLearner> {

  public enum TrainingType {
    BASIC, LINEAR, STOCHASTIC, BATCH, MINI_BATCH
  }

  private T learner;
  private TrainingType trainingType;
  private int batchSize = 1;

  private PrintWriter trainingPrintWriter;
  private PrintWriter testingPrintWriter;

  private MetricTracker trainingMetricTracker;
  private MetricTracker testingMetricTracker;

  public LearnerEvaluator(T learner) {
    this(learner, TrainingType.BASIC);

    if (learner instanceof NeuralNetwork && ((NeuralNetwork) learner).isLinearNetwork()) {
      this.trainingType = TrainingType.LINEAR;
    }
  }

  public LearnerEvaluator(T learner, TrainingType trainingType) {
    this.learner = learner;
    this.trainingType = trainingType;

    this.trainingMetricTracker = new MetricTracker();
    this.testingMetricTracker = new MetricTracker();
  }

  public void setTrainingType(TrainingType trainingType) {
    this.trainingType = trainingType;
  }

  public void setTrainingPrintWriter(PrintWriter trainingPrintWriter) {
    this.trainingPrintWriter = trainingPrintWriter;
  }

  public void setTestingPrintWriter(PrintWriter testingPrintWriter) {
    this.testingPrintWriter = testingPrintWriter;
  }

  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }

  public void resetMetrics() {
    this.trainingMetricTracker.reset();
    this.testingMetricTracker.reset();
  }

  public int countMisclassifications(Matrix features, Matrix labels) {
    return countMisclassifications(features, labels, false);
  }

  public int countMisclassifications(Matrix features, Matrix labels, boolean isTraining) {
    int misclassifications = 0;

    for (int row = 0; row < features.rows(); row++) {
      if (row % 100 == 0) {
        System.out.println("Testing row " + row + "...");
      }

      Vector output = learner.predict(features.row(row));
      int predictedNumber = output.maxIndex();

      if (predictedNumber != labels.row(row).maxIndex()) {
        misclassifications++;
      }
    }

    outputMisclassifications((double) misclassifications / features.rows(), isTraining);
    return misclassifications;
  }

  private void outputMisclassifications(double misclassifications, boolean isTraining) {
    MetricTracker metricTracker = (isTraining) ? trainingMetricTracker : testingMetricTracker;

    if (trainingPrintWriter != null && testingPrintWriter != null && trainingMetricTracker != null) {
      PrintWriter writer = (isTraining) ? trainingPrintWriter : testingPrintWriter;

      writer.printf("%d,%.3f,%.5f\n",
          metricTracker.getSteps(), metricTracker.getTime() / 1000.0, misclassifications);
      metricTracker.updateSteps(batchSize);
    } else {
//      System.out.printf("%d, %.3f, %.5f\n", metricTracker.getSteps(), metricTracker.getTime() / 1000.0, misclassifications);
    }
  }

  /**
   * Computes the sum squared error for this learner.
   */
  public double computeSumSquaredError(Matrix testFeatures, Matrix expectedLabels) {
    double sumSquaredError = 0;

    for (int i = 0; i < testFeatures.rows(); i++) {
      Vector x_i = testFeatures.row(i);
      Vector expected_y = expectedLabels.row(i);
      Vector calculated_y = learner.predict(x_i);

      // Calculate squared error
      calculated_y.scale(-1);
      calculated_y.add(expected_y);
      double squaredError = calculated_y.squaredMagnitude();

      sumSquaredError += squaredError;
    }

    return sumSquaredError;
  }

  double crossValidation(Matrix features, Matrix labels, int folds, int repetitions) {
    if (!learner.isValid()) {
      throw new IllegalStateException("Your NeuralNetwork is in an invalid state");
    }

    int dataRows = features.rows();
    int[] foldSizes = Matrix.computeFoldSizes(dataRows, folds);

    double totalError = 0;

    for (int r = 0; r < repetitions; r++) {
      Matrix.shuffleMatrices(features, labels);

      for (int i = 0; i < folds; i++) {
        Matrix X = Matrix.matrixWithoutFold(foldSizes, i, features);
        Matrix Y = Matrix.matrixWithoutFold(foldSizes, i, labels);

        Matrix test_X = Matrix.matrixFold(foldSizes, i, features);
        Matrix expected_Y = Matrix.matrixFold(foldSizes, i, labels);

        train(X, Y);

        double sumSquaredError = computeSumSquaredError(test_X, expected_Y);
        totalError += sumSquaredError;
      }
    }

    return Math.sqrt(totalError / repetitions / features.rows());
  }

  void train(Matrix features, Matrix labels) {
    switch (trainingType) {
      case BASIC:
        trainBasic(features, labels);
        break;

      case LINEAR:
        trainLinear(features, labels);
        break;

      case STOCHASTIC:
        trainStochastic(features, labels);
        break;

      case BATCH:
        trainBatch(features, labels);
        break;

      case MINI_BATCH:
        trainMiniBatch(features, labels, batchSize);
        break;

      default:
        trainStochastic(features, labels);
        break;
    }
  }

  private void trainBasic(Matrix features, Matrix labels) {
    if (!(learner instanceof NeuralNetwork)) {
      throw new IllegalStateException("Your learner must be a NeuralNetwork.");
    }

    NeuralNetwork neuralNetwork = (NeuralNetwork) learner;

    Matrix.shuffleMatrices(features, labels);

    for (int i = 0; i < features.rows(); i++) {
      int row = Main.RANDOM.nextInt(features.rows());

      Vector input = features.row(row);
      Vector output = labels.row(row);

      neuralNetwork.predict(input);
      neuralNetwork.backPropagate(output);
      neuralNetwork.updateGradient(input);
      neuralNetwork.updateWeights();
    }
  }

  private void trainLinear(Matrix features, Matrix labels) {
    if (!(learner instanceof NeuralNetwork)) {
      throw new IllegalStateException(
          "To train a linear model, your learner must be a NeuralNetwork");
    }

    NeuralNetwork neuralNetwork = (NeuralNetwork) learner;
    if (!neuralNetwork.isLinearNetwork()) {
      throw new IllegalStateException("Your NeuralNetwork must have exactly one LinearLayer");
    }

    if (features.rows() != labels.rows()) {
      throw new IllegalArgumentException(
          "Your input features and labels must have the same number of rows.");
    }

    LinearLayer layer = (LinearLayer) neuralNetwork.getLayers().get(0);
    layer.ordinaryLeastSquares(features, labels);
  }

  private void trainStochastic(Matrix features, Matrix labels) {
    trainMiniBatch(features, labels, 1);
  }

  private void trainBatch(Matrix features, Matrix labels) {
    trainMiniBatch(features, labels, features.rows());
  }

  private void trainMiniBatch(Matrix features, Matrix labels, int batchSize) {
    int batches = features.rows() / batchSize;

    for (int i = 0; i < batches; i++) {
      trainSingleMiniBatch(features, labels, batchSize, i);
    }
  }

  public void trainSingleRow(Matrix features, Matrix labels, int row) {
    trainSingleMiniBatch(features, labels, 1, row);
  }

  public void trainSingleMiniBatch(Matrix features, Matrix labels, int batchSize, int currentBatch) {
    if (!(learner instanceof NeuralNetwork)) {
      throw new IllegalStateException(
          "To train a linear model, your learner must be a NeuralNetwork");
    }

    NeuralNetwork neuralNetwork = (NeuralNetwork) learner;

    trainingMetricTracker.start();
    testingMetricTracker.start();

    for (int i = currentBatch * batchSize; i < currentBatch * batchSize + batchSize; i++) {
      Vector input = features.row(i);
      Vector output = labels.row(i);

      neuralNetwork.predict(input);
      neuralNetwork.backPropagate(output);
      neuralNetwork.updateGradient(input);
    }

    neuralNetwork.updateWeights();

    trainingMetricTracker.pause();
    testingMetricTracker.pause();
  }
}
