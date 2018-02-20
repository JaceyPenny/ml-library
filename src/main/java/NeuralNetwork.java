import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork extends SupervisedLearner {
  public static Random RANDOM = new Random();
  public static double LEARNING_RATE = 0.03;

  private int inputs;

  private List<Layer> layers;

  public NeuralNetwork() {
    this(0);
  }

  public NeuralNetwork(int inputs) {
    this.inputs = inputs;
    this.layers = new ArrayList<>();
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

  public void addLayer(LayerType layerType, int inputs, int outputs) {
    this.inputs = inputs;

    switch (layerType) {
      case LINEAR:
        layers.add(new LayerLinear(inputs, outputs));
        break;
      case TANH:
        layers.add(new LayerTanh(inputs));
        break;
      default:
        throw new NotImplementedException();
    }
  }

  public void addLayer(LayerType layerType, int outputs) {
    if (inputs == 0) {
      throw new IllegalStateException("You must first add a layer with a specified number of inputs");
    }

    int previousOutputs =
        (layers.size() == 0) ?
            inputs :
            layers.get(layers.size() - 1).getOutputs();


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

  @Override
  void train(Matrix features, Matrix labels) {
    train(features, labels, features.rows(), 1);
  }

  void train(Matrix features, Matrix labels, int repetitions, int stochastic) {
    for (int i = 0; i < repetitions; i++) {
      for (int j = 0; j < stochastic; j++) {
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
      layer.applyGradient(learningRate);
    }
  }

  private void backPropagate(Vector target) {
    checkLayers();

    Vector blame = Vector.copy(target);
    blame.addScaled(layers.get(layers.size() - 1).getActivation(), -1);
    layers.get(layers.size() - 1).setBlame(blame);

    for (int i = layers.size() - 1; i >= 1; i--) {
      blame = layers.get(i).backPropagate();
      layers.get(i-1).setBlame(blame);
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
