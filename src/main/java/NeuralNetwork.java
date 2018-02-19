import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork extends SupervisedLearner {
  private int numLayers;

  private List<LayerLinear> layers;

  public NeuralNetwork() {
    this(1);
  }

  public NeuralNetwork(int numLayers) {
    if (numLayers < 1) {
      throw new IllegalArgumentException("You must have at least one layer in this neural network");
    }

    this.layers = new ArrayList<>();
    this.numLayers = numLayers;
  }

  @Override
  String name() {
    return getClass().getSimpleName();
  }

  /**
   * Initializes this NeuralNetwork with the proper sizes for each layer.<br>
   * For example, a simple Linear Regression learner just needs to know the number of inputs
   * and outputs, and will simple create a single layer initialized with two int values, "inputs"
   * and "outputs".<br>
   * However, more complex neural networks (say, a network with a single hidden layer) will need to
   * know the desired sized of the hidden layers. Therefore, The caller must specify three values:
   * "inputs", "outputs", and a hidden layer size "a". They would call this function in the
   * following manner:
   * <pre>{@code
   * neuralNetwork.initializeLayers(inputs, a, outputs);
   * }</pre>
   * If the user wants to initialize {@code n} hidden layers, they should pass arguments in the following
   * format:
   * <pre>{@code
   * neuralNetwork.initializeLayers(inputs, a(1), a(2), a(3), ..., a(n-1), a(n), outputs);
   * }</pre>
   */
  private void initializeLayers(int... layerSizes) {
    if (layerSizes.length - 1 != numLayers) {
      throw new IllegalArgumentException(
          "You must pass the correct number of arguments: " + (numLayers + 1));
    }

    layers.clear();

    int inputs = layerSizes[0];
    int outputs = layerSizes[1];

    // TODO Modify this for custom hidden layers sizes
    for (int i = 0; i < numLayers; i++) {
      layers.add(new LayerLinear(inputs, outputs));

      if (i < numLayers - 1) {
        inputs = outputs;
        outputs = layerSizes[i + 2];
      }
    }
  }

  @Override
  void train(Matrix features, Matrix labels) {
    initializeLayers(features.cols(), labels.cols());

    for (LayerLinear layer : layers) {
      layer.ordinaryLeastSquares(features, labels);
    }
  }

  @Override
  Vector predict(Vector in) {
    if (layers.size() == 0) {
      throw new IllegalStateException("The network has not been trained yet.");
    }

    layers.get(0).activate(in);

    for (int i = 1; i < layers.size(); i++) {
      Vector previousActivation = layers.get(i - 1).getActivation();
      layers.get(i).activate(previousActivation);
    }

    return layers.get(layers.size() - 1).getActivation();
  }

  void backPropagate(Vector weights, Vector target) {
    Vector blame = Vector.copy(target);
    blame.addScaled(layers.get(layers.size() - 1).getActivation(), -1);
    layers.get(layers.size() - 1).setBlame(blame);

    for (int i = layers.size() - 2; i > 0; i++) {
      LayerLinear layer = layers.get(i);

      layer.backPropagate(blame);
      blame = layer.getBlame();
    }
  }

  void updateGradient() {

  }
}
