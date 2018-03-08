import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork extends SupervisedLearner {
  public static Random RANDOM = new Random();
  public static double LEARNING_RATE = 0.0001;

  private List<Layer> layers;
  private double momentum = 0.0;
  private OutputStream outputStream;

  public NeuralNetwork() {
    this.layers = new ArrayList<>();
  }

  public void setOutputStream(OutputStream outputStream) {
    this.outputStream = outputStream;
  }

  public void setMomentum(float momentum) {
    this.momentum = momentum;
  }

  @Override
  String name() {
    return getClass().getSimpleName();
  }

  private void checkLayers() {
    if (layers.size() == 0) {
      throw new IllegalStateException("This network has no layers.");
    }
  }

  public void addFirstLayer(LayerType layerType, int inputs, int outputs) {
    this.layers.clear();
    addLayer(layerType, inputs, outputs);
  }

  private void addLayer(LayerType layerType, int inputs, int outputs) {
    switch (layerType) {
      case LINEAR:
        layers.add(new LayerLinear(inputs, outputs));
        break;
      case TANH:
        layers.add(new LayerTanh(inputs));
        break;
      default:
        throw new IllegalArgumentException("No implementation exists for this layer type.");
    }
  }

  public void addLayer(LayerType layerType, int outputs) {
    if (layers.isEmpty()) {
      throw new IllegalStateException("You must first add a layer with a specified number of inputs");
    }

    int previousOutputs = layers.get(layers.size() - 1).getOutputs();


    addLayer(layerType, previousOutputs, outputs);
  }

  public void initializeWeights() {
    for (Layer layer : layers) {
      if (layer instanceof LayerLinear) {
        ((LayerLinear) layer).initializeWeights();
      }
    }
  }

  @Override
  double crossValidation(int folds, int repetitions, Matrix features, Matrix labels) {
    if (layers.size() != 1 || layers.get(0).getLayerType() != LayerType.LINEAR) {
      throw new IllegalStateException("Your NeuralNetwork must have exactly one LayerLinear");
    }

    int dataRows = features.rows();
    int[] foldSizes = Matrix.computeFoldSizes(dataRows, folds);

    double totalError = 0;

    for (int r = 0; r < repetitions; r++) {
      shuffleData(features, labels);

      for (int i = 0; i < folds; i++) {
        Matrix X = Matrix.matrixWithoutFold(foldSizes, i, features);
        Matrix Y = Matrix.matrixWithoutFold(foldSizes, i, labels);

        Matrix test_X = Matrix.matrixFold(foldSizes, i, features);
        Matrix expected_Y = Matrix.matrixFold(foldSizes, i, labels);

        trainLinear(X, Y);

        double sumSquaredError = computeSumSquaredError(test_X, expected_Y);
        totalError += sumSquaredError;
      }
    }

    return Math.sqrt(totalError / repetitions / features.rows());
  }

  @Override
  int countMisclassifications(Matrix features, Matrix labels) {
    int misclassifications = 0;

    for (int row = 0; row < features.rows(); row++) {
      Vector output = predict(features.row(row));
      int predictedNumber = output.maxIndex();

      if (predictedNumber != labels.row(row).maxIndex()) {
        misclassifications++;
      }
    }

    return misclassifications;
  }

  private void trainLinear(Matrix features, Matrix labels) {
    if (layers.size() != 1 || layers.get(0).getLayerType() != LayerType.LINEAR) {
      throw new IllegalStateException("Your NeuralNetwork must have exactly one LayerLinear");
    }

    LayerLinear layer = (LayerLinear) layers.get(0);
    layer.ordinaryLeastSquares(features, labels);
  }

  @Override
  void train(Matrix features, Matrix labels) {
    trainStochastic(features, labels);
  }

  void trainStochastic(Matrix features, Matrix labels) {
    trainMiniBatch(features, labels, 1);
  }

  void trainMiniBatch(Matrix features, Matrix labels, int batchSize) {
    train(features, labels, features.rows() / batchSize, batchSize);
  }

  void train(Matrix features, Matrix labels, int repetitions, int rows) {
    shuffleData(features, labels);

    for (int i = 0; i < repetitions; i++) {
      for (int j = 0; j < rows; j++) {
        int row = RANDOM.nextInt(features.rows());

        Vector input = features.row(row);
        Vector output = labels.row(row);

        predict(input);
        backPropagate(output);
        updateGradient(input);
      }

      updateWeights(LEARNING_RATE);
    }
  }

  @Override
  Vector predict(Vector in) {
    checkLayers();

    layers.get(0).activate(in);

    for (int i = 1; i < layers.size(); i++) {
      Vector previousActivation = layers.get(i - 1).getActivation();
      layers.get(i).activate(previousActivation);
    }

    return layers.get(layers.size() - 1).getActivation();
  }

  private void refineWeights(Vector input, Vector target, double learningRate) {
    predict(input);
    backPropagate(target);
    updateGradient(input);
    updateWeights(learningRate);
  }

  private void updateWeights(double learningRate) {
    for (Layer layer : layers) {
      layer.applyGradient(learningRate, momentum);
    }
  }

  private void backPropagate(Vector target) {
    checkLayers();

    Vector blame = target.copy();
    blame.addScaled(layers.get(layers.size() - 1).getActivation(), -1);
    layers.get(layers.size() - 1).setBlame(blame);

    for (int i = layers.size() - 1; i >= 1; i--) {
      blame = layers.get(i).backPropagate();
      layers.get(i - 1).setBlame(blame);
    }
  }

  private void updateGradient(Vector x) {
    checkLayers();

    Vector previousActivation = x;
    for (Layer layer : layers) {
      layer.updateGradient(previousActivation);
      previousActivation = layer.getActivation();
    }
  }

  public void printTopology() {
    for (int i = 0; i < layers.size(); i++) {
      System.out.printf("%d) %s\n", i, layers.get(i).topologyString());
    }
  }
}
